"""Stage 8: Final Pre-Submission Technical Verification (P0-2, P1-1, P1-2).

Read-only with respect to all raw inputs, annotations, outputs/, frozen phases,
and Stages 0-7 artifacts. Writes strictly inside
analysis/path_b_plus/stage8_final_verification/.

Tasks:
  1. P0-2: Independent 8-dihedral orientation NCC & mask-IoU audit across all
     127 active analysis trays (67 July primary + 60 August 6-condition trays).
  2. P1-1: Protocol 2 audit (tray kernel counts, unique subset counts, exact PCA
     ranks, selected hyperparameters) and controlled factorization of
     k20->All vs. All_exact for Logistic and RBF-SVM (early1 & early2):
       - Matched PCA rank (p=30) vs. No-PCA (421 standardized bands) vs. Orig PCA
       - Matched per-tray total loss weight (sample_weight = 1/M)
       - Expanded comparable regularization grid C in {0.01, 0.1, 1, 10, 100, 1000}
         selected strictly via existing 3-fold inner tray-grouped CV.
  3. P1-2: Numerical verification of the exact algebraic SNV-pooling identity
     h(mean(x_i)) == h(sum(s_i * z_i)) where x_i = m_i + s_i * z_i.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
LONG = PROJECT / "analysis" / "long_term_validation" / "data"
OUT_DIR = ROOT / "stage8_final_verification"

sys.path.insert(0, str(PROJECT))
from night_pilot.orientation_check import FIELD_TO_NAME, TRANSFORMS, apply_transform, normalised_correlation
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE
from night_pilot.pipeline import parse_hdr, read_unicode_image, resize_analysis
from night_pilot.robustness import assert_no_group_leakage
from night_pilot.seed_lots import assert_lot_declared

LABELS = np.asarray(FROZEN_LABEL_SPACE)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
CONDITIONS = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
SEED = 20261126
EPS = 1e-12


def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)


# ============================================================================
# TASK 1 (P0-2): INDEPENDENT 8-DIHEDRAL ORIENTATION AUDIT ON ALL 127 ACTIVE TRAYS
# ============================================================================
def _audit_one_tray(rec: dict) -> dict:
    ann_dir = Path(rec["ann_dir"])
    payload = json.loads((ann_dir / "annotation.json").read_text(encoding="utf-8"))
    rgb_labels = cv2.imdecode(np.fromfile(str(ann_dir / "rgb_labels.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    hsi_labels = cv2.imdecode(np.fromfile(str(ann_dir / "hsi_labels.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    reg = payload.get("registration") or {}
    matrix = np.asarray(reg["matrix_oriented_hsi_to_rgb"], dtype=np.float64)
    rot_val = int(reg.get("rotation", 90))
    recorded = FIELD_TO_NAME.get(rot_val, f"unknown_{rot_val}")
    mirror_field = reg.get("mirror", "none")
    rgb_shape = rgb_labels.shape

    mask_scores: dict[str, float] = {}
    for name in TRANSFORMS:
        oriented = apply_transform(hsi_labels, name).astype(np.uint16)
        warped = cv2.warpPerspective(oriented, matrix, (rgb_shape[1], rgb_shape[0]), flags=cv2.INTER_NEAREST)
        union = int(((warped > 0) | (rgb_labels > 0)).sum())
        mask_scores[name] = float(((warped > 0) & (rgb_labels > 0)).sum() / union) if union else 0.0

    # Independent HSI intensity map (1000-1100 nm mean reflectance)
    hsi_raw_path = Path(rec["hsi_path"])
    hdr = parse_hdr(hsi_raw_path.with_suffix(".hdr"))
    samples, lines, bands = int(hdr["samples"]), int(hdr["lines"]), int(hdr["bands"])
    raw = np.memmap(hsi_raw_path, dtype="<u2", mode="r", shape=(lines, bands, samples))
    dark = np.fromfile(rec["hsi_dark_path"], dtype="<u2").reshape(bands, samples).astype(np.float32)
    white = np.fromfile(rec["hsi_white_path"], dtype="<u2").reshape(bands, samples).astype(np.float32)
    den = np.where((white - dark) > 0, white - dark, np.nan)
    waves = np.asarray(hdr["wavelength"], dtype=np.float32)
    bslice = np.where((waves >= 1000.0) & (waves <= 1100.0))[0]
    chunk = np.asarray(raw[:, bslice, :], dtype=np.float32)
    refl = (chunk - dark[bslice, :]) / den[bslice, :]
    intensity = np.nan_to_num(np.nanmean(refl, axis=1), nan=0.0).astype(np.float32)
    del raw

    rgb_img = read_unicode_image(Path(rec["rgb_path"]))
    rgb_gray = cv2.cvtColor(resize_analysis(rgb_img), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if rgb_gray.shape != rgb_shape:
        rgb_gray = cv2.resize(rgb_gray, (rgb_shape[1], rgb_shape[0]), interpolation=cv2.INTER_AREA)

    int_scores: dict[str, float] = {}
    for name in TRANSFORMS:
        oriented = apply_transform(intensity, name).astype(np.float32)
        warped = cv2.warpPerspective(oriented, matrix, (rgb_shape[1], rgb_shape[0]), flags=cv2.INTER_LINEAR)
        cover = cv2.warpPerspective(
            np.ones_like(oriented, np.uint8), matrix, (rgb_shape[1], rgb_shape[0]), flags=cv2.INTER_NEAREST
        ).astype(bool)
        int_scores[name] = normalised_correlation(warped, rgb_gray, cover)

    mask_best = max(mask_scores, key=mask_scores.get)
    int_best = max(int_scores, key=int_scores.get)
    int_sorted = sorted(int_scores.values(), reverse=True)
    return dict(
        cohort=rec["cohort"],
        condition=rec["condition"],
        record_id=rec["record_id"],
        variety=rec["variety"],
        rotation_field=rot_val,
        mirror_field=mirror_field,
        recorded_transform=recorded,
        mask_best=mask_best,
        mask_best_iou=mask_scores[mask_best],
        mask_recorded_iou=mask_scores.get(recorded, np.nan),
        intensity_best=int_best,
        intensity_best_corr=int_scores[int_best],
        intensity_recorded_corr=int_scores.get(recorded, np.nan),
        intensity_second_corr=int_sorted[1],
        intensity_margin=int_scores[int_best] - int_sorted[1],
        mask_agrees_with_field=(mask_best == recorded),
        intensity_agrees_with_field=(int_best == recorded),
        criteria_agree=(mask_best == int_best),
    )


def run_task1_orientation_audit(primary_rows: pd.DataFrame, ext_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    csv_p = OUT_DIR / "orientation_audit_127trays.csv"
    js_p = OUT_DIR / "orientation_audit_127trays_summary.json"
    if csv_p.exists() and js_p.exists():
        logging.info("Task 1 (P0-2): SKIP cached 127-tray orientation audit.")
        return pd.read_csv(csv_p), json.loads(js_p.read_text(encoding="utf-8"))
    logging.info("Task 1 (P0-2): Auditing 8-dihedral orientation across all 127 active trays (67 July + 60 August)...")
    tbm = pd.read_csv(PROJECT / "outputs/datasets/Maize_RGB_HSI_ThirdBatch_v1.0/third_batch_manifest.csv", dtype=str).set_index("record_id")
    jobs = []
    for rid, blk in primary_rows.groupby("record_id", sort=True):
        r0 = blk.iloc[0]
        jobs.append(dict(
            cohort="primary_july",
            condition="PRIMARY",
            record_id=str(rid),
            variety=str(r0["variety_name"]),
            ann_dir=str(PROJECT / "annotations/projects/second_batch" / str(rid)),
            rgb_path=str(r0["rgb_path"]),
            hsi_path=str(r0["hsi_path"]),
            hsi_dark_path=str(r0["hsi_dark_path"]),
            hsi_white_path=str(r0["hsi_white_path"]),
        ))
    for rid, blk in ext_rows.groupby("record_id", sort=True):
        r0 = blk.iloc[0]
        tm = tbm.loc[str(rid)]
        jobs.append(dict(
            cohort="later_date_august",
            condition=str(r0["condition"]),
            record_id=str(rid),
            variety=str(r0["variety"]),
            ann_dir=str(PROJECT / "annotations/projects/third_batch" / str(rid)),
            rgb_path=str(tm["rgb_path"]),
            hsi_path=str(tm["hsi_path"]),
            hsi_dark_path=str(r0["hsi_dark_path"]),
            hsi_white_path=str(r0["white_reference_path"]),
        ))
    assert len(jobs) == 127

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_audit_one_tray, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())

    df = pd.DataFrame(results).sort_values(["cohort", "condition", "record_id"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "orientation_audit_127trays.csv", index=False)

    summary = {}
    for name, sub in [
        ("all_127", df),
        ("primary_67", df[df["cohort"].eq("primary_july")]),
        ("august_60", df[df["cohort"].eq("later_date_august")]),
    ]:
        c = sub["intensity_recorded_corr"]
        summary[name] = dict(
            n_trays=int(len(sub)),
            mask_agrees=int(sub["mask_agrees_with_field"].sum()),
            intensity_agrees=int(sub["intensity_agrees_with_field"].sum()),
            criteria_agree=int(sub["criteria_agree"].sum()),
            rotations=sub["recorded_transform"].value_counts().to_dict(),
            corr_min=float(c.min()),
            corr_q25=float(c.quantile(0.25)),
            corr_median=float(c.median()),
            corr_q75=float(c.quantile(0.75)),
            corr_max=float(c.max()),
            margin_min=float(sub["intensity_margin"].min()),
            margin_median=float(sub["intensity_margin"].median()),
        )
    # Also by August condition
    summary["august_by_condition"] = {}
    for cond, sub in df[df["cohort"].eq("later_date_august")].groupby("condition"):
        c = sub["intensity_recorded_corr"]
        summary["august_by_condition"][cond] = dict(
            n=int(len(sub)),
            intensity_agrees=int(sub["intensity_agrees_with_field"].sum()),
            mask_agrees=int(sub["mask_agrees_with_field"].sum()),
            corr_median=float(c.median()),
            corr_min=float(c.min()),
        )

    (OUT_DIR / "orientation_audit_127trays_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return df, summary


# ============================================================================
# TASK 2 (P1-1): PROTOCOL 2 AUDIT + CONTROLLED PCA / WEIGHT / C FACTORIZATION
# ============================================================================
def make_ctrl_pipeline(clf_name: str, params: dict, pca_mode: str, n_features: int, n_train: int) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca_mode == "orig60":
        n_comp = min(60, n_features, n_train - 1)
        steps.append(("pca", PCA(n_components=n_comp, svd_solver="randomized", random_state=42)))
    elif pca_mode == "fixed30":
        n_comp = min(30, n_features, n_train - 1)
        steps.append(("pca", PCA(n_components=n_comp, svd_solver="randomized", random_state=42)))
    elif pca_mode == "none":
        pass
    else:
        raise ValueError(pca_mode)

    if clf_name == "logistic":
        steps.append((
            "clf",
            LogisticRegression(
                C=params["C"],
                max_iter=4000,
                class_weight="balanced",
                random_state=42,
            ),
        ))
    elif clf_name == "rbf_svm":
        steps.append((
            "clf",
            SVC(
                C=params["C"],
                gamma=params["gamma"],
                kernel="rbf",
                class_weight="balanced",
            ),
        ))
    else:
        raise ValueError(clf_name)
    return Pipeline(steps)


def fit_ctrl_model(
    clf_name: str,
    params: dict,
    pca_mode: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    rec_tr: np.ndarray,
    sample_weight: np.ndarray | None,
    cal_splits: int,
    seed_offset: int,
):
    if clf_name == "rbf_svm":
        sgkf = StratifiedGroupKFold(n_splits=cal_splits, shuffle=True, random_state=SEED + seed_offset)
        cv_idx = list(sgkf.split(X_tr, y_tr, rec_tr))
        for a, b in cv_idx:
            assert not (set(rec_tr[a]) & set(rec_tr[b]))
        min_cal_tr = min(len(a) for a, _ in cv_idx)
        base = make_ctrl_pipeline(clf_name, params, pca_mode, X_tr.shape[1], min_cal_tr)
        # Note: CalibratedClassifierCV passes sample_weight down to base estimator via fit(X, y, sample_weight=...)
        # Wait: Pipeline expects clf__sample_weight if sample_weight passed directly, whereas CalibratedClassifierCV
        # forwards sample_weight to base_estimator.fit(X, y, sample_weight=sw)!
        # Let's avoid Pipeline routing mismatch by wrapping cleanly if sample_weight is used.
        if sample_weight is not None:
            # Since every subset has identical constant weight w = 1/M, for SVC with class_weight="balanced"
            # scaling ALL sample_weights by a constant alpha is algebraically identical to replacing C -> alpha * C!
            # Verify: min 0.5||w||^2 + C * sum(alpha * class_w_i * xi_i) == min 0.5||w||^2 + (alpha * C) * sum(class_w_i * xi_i).
            # For LogisticRegression: class_weight='balanced' uses unscaled n/(K*n_c), and sample_weight scales loss by alpha,
            # so constant sample_weight = alpha is ALSO algebraically identical to C_eff = alpha * C!
            alpha = float(sample_weight[0])
            assert np.allclose(sample_weight, alpha), "Expected uniform per-subset weight 1/M"
            p_eff = dict(params)
            p_eff["C"] = float(params["C"]) * alpha
            base = make_ctrl_pipeline(clf_name, p_eff, pca_mode, X_tr.shape[1], min_cal_tr)
        model = CalibratedClassifierCV(base, cv=cv_idx, n_jobs=min(3, cal_splits), ensemble=True)
        model.fit(X_tr, y_tr)
        return model
    else:
        if sample_weight is not None:
            alpha = float(sample_weight[0])
            assert np.allclose(sample_weight, alpha)
            p_eff = dict(params)
            p_eff["C"] = float(params["C"]) * alpha
            model = make_ctrl_pipeline(clf_name, p_eff, pca_mode, X_tr.shape[1], len(X_tr))
        else:
            model = make_ctrl_pipeline(clf_name, params, pca_mode, X_tr.shape[1], len(X_tr))
        model.fit(X_tr, y_tr)
        return model


def predict_aligned_proba(model, X: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    order = [int(np.flatnonzero(classes == lbl).item()) for lbl in LABELS]
    probs = np.asarray(model.predict_proba(X), dtype=np.float64)[:, order]
    probs = np.clip(probs, 0.0, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def run_task2_protocol2_control(
    primary_rows: pd.DataFrame,
    ext_rows: pd.DataFrame,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    boot_weights: np.ndarray,
    entity_sampling: dict,
    entities: list[str],
) -> tuple[pd.DataFrame, dict]:
    logging.info("Task 2 (P1-1): Running Protocol 2 audit & controlled PCA/weight/C factorization...")
    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        raw_j_f32 = psp["statistics"][:, 0, :]
        X_raw_j = raw_j_f32.astype(np.float64)
        X_snv_j = snv(raw_j_f32).astype(np.float64)
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        raw_a_f32 = esp["statistics"][:, 0, :]
        X_raw_a = raw_a_f32.astype(np.float64)
        X_snv_a = snv(raw_a_f32).astype(np.float64)

    y_july_str = primary_rows["variety_name"].to_numpy()
    rec_july = primary_rows["record_id"].to_numpy()
    folds = primary_rows["fold"].to_numpy()
    tray_folds = dict(zip(primary_rows["record_id"], primary_rows["fold"]))
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]
    fold_test_trays = {f: [t for t, r in trays_df.iterrows() if tray_folds[str(r["record_id"])] == f] for f in range(4)}

    # Precompute Primary full-tray pooled spectra and August full-tray pooled spectra for early1 & early2
    prim_all = {"early1": {}, "early2": {}}
    for f in range(4):
        prim_all["early1"][f] = np.vstack([snv(X_raw_j[tray_idx_list[t]].mean(axis=0, keepdims=True)) for t in fold_test_trays[f]])
        prim_all["early2"][f] = np.vstack([X_snv_j[tray_idx_list[t]].mean(axis=0, keepdims=True) for t in fold_test_trays[f]])

    ext_rows_idx = ext_rows.copy()
    ext_rows_idx["row_index"] = np.arange(len(ext_rows_idx))
    aug_all = {"early1": {}, "early2": {}}
    aug_truth = np.empty(10, dtype=int)
    for cond in CONDITIONS:
        block = ext_rows_idx[ext_rows_idx["condition"].eq(cond)]
        e1_l, e2_l = [], []
        for t, ent in enumerate(entities):
            sids, _ = entity_sampling[ent]
            sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
            j = sub["row_index"].to_numpy(dtype=int)
            e1_l.append(snv(X_raw_a[j].mean(axis=0, keepdims=True)))
            e2_l.append(X_snv_a[j].mean(axis=0, keepdims=True))
            if cond == "BASE":
                aug_truth[t] = LABEL_TO_IDX[sub.iloc[0]["variety"]]
        aug_all["early1"][cond] = np.vstack(e1_l)
        aug_all["early2"][cond] = np.vstack(e2_l)

    # 1. Exact subset uniqueness & PCA rank audit for M=50, k=20 vs All_exact
    audit_trays = []
    for f in range(4):
        outer_idx = np.flatnonzero(folds != f)
        u_trays = sorted(np.unique(rec_july[outer_idx]).tolist())
        for ti, tid in enumerate(u_trays):
            loc_abs = outer_idx[np.flatnonzero(rec_july[outer_idx] == tid)]
            n_k = len(loc_abs)
            kk = min(20, n_k)
            rng = np.random.default_rng(SEED + f * 10000 + 20 * 100 + ti)
            draws = [tuple(sorted(rng.permutation(n_k)[:kk].tolist())) for _ in range(50)]
            audit_trays.append(dict(
                fold=f, record_id=tid, n_kernels=n_k,
                n_draws=50, n_unique_draws=len(set(draws)),
                max_possible_subsets=int(math.comb(n_k, kk)),
            ))
    aud_df = pd.DataFrame(audit_trays)
    subset_audit = dict(
        min_tray_kernels=int(aud_df["n_kernels"].min()),
        median_tray_kernels=float(aud_df["n_kernels"].median()),
        max_tray_kernels=int(aud_df["n_kernels"].max()),
        trays_with_n_eq_20=int(primary_rows.groupby("record_id").size().eq(20).sum()),
        trays_with_n_le_21=int(primary_rows.groupby("record_id").size().le(21).sum()),
        fold_tray_pairs_with_50_unique=int((aud_df["n_unique_draws"] == 50).sum()),
        total_fold_tray_pairs=int(len(aud_df)),
        non_unique_tray_details=aud_df[aud_df["n_unique_draws"] < 50].drop_duplicates("record_id")[
            ["record_id", "n_kernels", "n_draws", "n_unique_draws", "max_possible_subsets"]
        ].to_dict("records"),
    )

    # Verify exact algebraic equivalence of constant sample_weight=1/M and C_eff=C/M on LogisticRegression & SVC!
    outer0 = np.flatnonzero(folds != 0)
    sub_idx = outer0[::15]
    X_chk = X_snv_j[sub_idx]
    y_chk = y_july_str[sub_idx]
    sw_chk = np.full(len(X_chk), 1.0 / 50.0)
    m_sw = LogisticRegression(C=10.0, max_iter=4000, class_weight="balanced", random_state=42).fit(X_chk, y_chk, sample_weight=sw_chk)
    m_ce = LogisticRegression(C=10.0 / 50.0, max_iter=4000, class_weight="balanced", random_state=42).fit(X_chk, y_chk)
    p_diff_sw = float(np.max(np.abs(m_sw.predict_proba(X_chk) - m_ce.predict_proba(X_chk))))
    s_sw = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced").fit(X_chk, y_chk, sample_weight=sw_chk)
    s_ce = SVC(C=10.0 / 50.0, gamma="scale", kernel="rbf", class_weight="balanced").fit(X_chk, y_chk)
    s_diff_sw = float(np.max(np.abs(s_sw.decision_function(X_chk) - s_ce.decision_function(X_chk))))
    logging.info("  Verified sample_weight=1/50 vs C/50 equivalence: Logistic max_prob_diff=%.3g, SVC max_df_diff=%.3g", p_diff_sw, s_diff_sw)
    subset_audit["sample_weight_vs_C_scale_logistic_max_prob_diff"] = p_diff_sw
    subset_audit["sample_weight_vs_C_scale_svc_max_df_diff"] = s_diff_sw

    # Prebuild k20 (M=50) and All_exact (M=1) training sets for each outer fold & route
    train_sets = {}
    for rkey in ("early1", "early2"):
        for f in range(4):
            outer_idx = np.flatnonzero(folds != f)
            inner_vals = primary_rows[f"inner_fold_for_outer{f}"].to_numpy()[outer_idx]
            u_trays = sorted(np.unique(rec_july[outer_idx]).tolist())
            # All_exact
            Xa, ya, ra, ia = [], [], [], []
            # k20 (M=50)
            Xk, yk, rk, ik = [], [], [], []
            for ti, tid in enumerate(u_trays):
                loc_rel = np.flatnonzero(rec_july[outer_idx] == tid)
                loc_abs = outer_idx[loc_rel]
                lbl = y_july_str[loc_abs[0]]
                inf = int(inner_vals[loc_rel[0]])
                xr = X_raw_j[loc_abs]
                xs = X_snv_j[loc_abs]
                Xa.append(snv(xr.mean(axis=0, keepdims=True)) if rkey == "early1" else xs.mean(axis=0, keepdims=True))
                ya.append(lbl); ra.append(tid); ia.append(inf)

                kk = min(20, len(loc_abs))
                rng = np.random.default_rng(SEED + f * 10000 + 20 * 100 + ti)
                draws = np.stack([rng.permutation(len(loc_abs))[:kk] for _ in range(50)])
                Xk.append(snv(xr[draws].mean(axis=1)) if rkey == "early1" else xs[draws].mean(axis=1))
                yk.extend([lbl] * 50); rk.extend([tid] * 50); ik.extend([inf] * 50)
            train_sets[(rkey, f, "All_exact")] = (
                np.concatenate(Xa, axis=0), np.asarray(ya, dtype=object), np.asarray(ra, dtype=object), np.asarray(ia, dtype=int)
            )
            train_sets[(rkey, f, "k20")] = (
                np.concatenate(Xk, axis=0), np.asarray(yk, dtype=object), np.asarray(rk, dtype=object), np.asarray(ik, dtype=int)
            )

    # Late baselines for contrast
    preds_dir = ROOT / "models_and_preds"
    late_corr = {}
    for clf_name in ("logistic", "rbf_svm"):
        with np.load(preds_dir / f"S_{clf_name}_mean.npz") as a:
            op = a["outer_oof_prob"]
        late_corr[clf_name] = np.array(
            [float(int(op[tray_idx_list[t]].mean(axis=0).argmax()) == truth_arr[t]) for t in range(67)],
            dtype=np.float64,
        )

    # Define controlled experimental matrix
    # (setting_id, train_mode, pca_mode, tray_weight_matched, c_grid_name)
    settings = [
        # Baseline original Stage 6 replication (to verify exact replay)
        ("S0_orig_PCA60_origC", "All_exact", "orig60", False, "orig"),
        ("S0_orig_PCA60_origC", "k20",       "orig60", False, "orig"),
        # Control 1: Equal PCA rank (p=30) + original C grid (unweighted)
        ("S1_fixedPCA30_origC", "All_exact", "fixed30", False, "orig"),
        ("S1_fixedPCA30_origC", "k20",       "fixed30", False, "orig"),
        # Control 2: Equal PCA rank (p=30) + matched tray weight (1/M) + wide C grid
        ("S2_fixedPCA30_wt1_wideC", "All_exact", "fixed30", True, "wide"),
        ("S2_fixedPCA30_wt1_wideC", "k20",       "fixed30", True, "wide"),
        # Control 3: No PCA (421 std bands) + matched tray weight (1/M) + wide C grid
        ("S3_noPCA_wt1_wideC", "All_exact", "none", True, "wide"),
        ("S3_noPCA_wt1_wideC", "k20",       "none", True, "wide"),
        # Control 4: Original PCA (min(60, n-1)) + matched tray weight (1/M) + wide C grid
        ("S4_origPCA_wt1_wideC", "All_exact", "orig60", True, "wide"),
        ("S4_origPCA_wt1_wideC", "k20",       "orig60", True, "wide"),
    ]

    def get_grid(clf_name: str, grid_name: str):
        if clf_name == "logistic":
            cs = (0.1, 1.0, 10.0) if grid_name == "orig" else (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
            return [{"C": c} for c in cs]
        else:
            cs = (1.0, 10.0, 100.0) if grid_name == "orig" else (0.1, 1.0, 10.0, 100.0, 1000.0)
            return [{"C": c, "gamma": g} for c in cs for g in ("scale", 0.01)]

    rows_out = []
    detail_logs = []

    for clf_name in ("logistic", "rbf_svm"):
        for rkey in ("early1", "early2"):
            for setting_id, train_mode, pca_mode, wt_matched, grid_name in settings:
                # For RBF-SVM, run early1 across all controls and early2 for S0/S2/S3 to keep runtime crisp & complete
                grid = get_grid(clf_name, grid_name)
                corr_prim = np.zeros(67, dtype=np.float64)
                fold_models = []
                fold_chosen = []
                fold_pca_inner = []
                fold_pca_outer = []

                for f in range(4):
                    X_tr, y_tr, rec_tr, inf_tr = train_sets[(rkey, f, train_mode)]
                    alpha = (1.0 / 50.0) if (wt_matched and train_mode == "k20") else 1.0

                    # Record actual PCA dimensions
                    min_in_n = min(int((inf_tr != g).sum()) for g in range(3))
                    if pca_mode == "orig60":
                        in_p = min(60, X_tr.shape[1], min_in_n - 1)
                        out_p = min(60, X_tr.shape[1], len(X_tr) - 1)
                    elif pca_mode == "fixed30":
                        in_p = min(30, X_tr.shape[1], min_in_n - 1)
                        out_p = min(30, X_tr.shape[1], len(X_tr) - 1)
                    else:
                        in_p = X_tr.shape[1]
                        out_p = X_tr.shape[1]
                    fold_pca_inner.append(in_p)
                    fold_pca_outer.append(out_p)

                    def _eval_cand(params, g):
                        tr_m = inf_tr != g
                        va_m = inf_tr == g
                        p_eff = dict(params)
                        p_eff["C"] = float(params["C"]) * alpha
                        est = make_ctrl_pipeline(clf_name, p_eff, pca_mode, X_tr.shape[1], int(tr_m.sum()))
                        est.fit(X_tr[tr_m], y_tr[tr_m])
                        return va_m, est.predict(X_tr[va_m])

                    tasks = [(pi, g) for pi in range(len(grid)) for g in range(3)]
                    outs = Parallel(n_jobs=14, backend="loky")(
                        delayed(_eval_cand)(grid[pi], g) for pi, g in tasks
                    )
                    scores = []
                    for pi in range(len(grid)):
                        pred_in = np.empty(len(y_tr), dtype=object)
                        for g in range(3):
                            va_m, pr = outs[pi * 3 + g]
                            pred_in[va_m] = pr
                        scores.append(float(balanced_accuracy_score(y_tr, pred_in)))
                    best_i = int(np.argmax(scores))
                    chosen = grid[best_i]
                    fold_chosen.append(chosen)

                    sw = np.full(len(X_tr), alpha) if (wt_matched and train_mode == "k20") else None
                    mdl = fit_ctrl_model(
                        clf_name, chosen, pca_mode, X_tr, y_tr, rec_tr, sw,
                        cal_splits=3, seed_offset=f * 100 + 7,
                    )
                    fold_models.append(mdl)

                    t_list = fold_test_trays[f]
                    pa = predict_aligned_proba(mdl, prim_all[rkey][f])
                    for lt, t in enumerate(t_list):
                        corr_prim[t] = float(int(pa[lt].argmax()) == truth_arr[t])

                prim_ba = float(corr_prim @ class_w)
                prim_trays = int(corr_prim.sum())
                db = (corr_prim - late_corr[clf_name]) @ boot_weights.T
                delta_late = float(prim_ba - float(late_corr[clf_name] @ class_w))

                # Evaluate August 6 conditions
                aug_hits = np.empty((len(CONDITIONS), 10), dtype=np.float64)
                for ci, cond in enumerate(CONDITIONS):
                    pp = sum(predict_aligned_proba(fold_models[f], aug_all[rkey][cond]) for f in range(4)) / 4.0
                    aug_hits[ci] = (pp.argmax(axis=1) == aug_truth).astype(np.float64)

                rows_out.append(dict(
                    classifier=clf_name,
                    route=rkey,
                    setting=setting_id,
                    train_mode="k20->All" if train_mode == "k20" else "All_exact",
                    pca_mode=pca_mode,
                    pca_dim_inner_min=min(fold_pca_inner),
                    pca_dim_outer_min=min(fold_pca_outer),
                    tray_weight_matched=wt_matched,
                    c_grid=grid_name,
                    prim_all_ba=prim_ba,
                    prim_correct_trays=prim_trays,
                    delta_vs_late_pp=100.0 * delta_late,
                    ci_low_pp=100.0 * float(np.quantile(db, 0.025)),
                    ci_high_pp=100.0 * float(np.quantile(db, 0.975)),
                    aug_base_all=float(aug_hits[0].mean()),
                    aug_6cond_mean_all=float(aug_hits.mean()),
                    aug_lamp_loss_mean=float(aug_hits[[4, 5]].mean()),
                    chosen_params=json.dumps(fold_chosen),
                ))
                detail_logs.append((clf_name, rkey, setting_id, train_mode, prim_ba, float(aug_hits.mean())))
            logging.info("  Completed P1-1 controls for %s / %s", clf_name, rkey)

    res_df = pd.DataFrame(rows_out)
    res_df.to_csv(OUT_DIR / "protocol2_pca_weight_c_controls.csv", index=False)
    (OUT_DIR / "protocol2_subset_audit.json").write_text(json.dumps(subset_audit, indent=2) + "\n", encoding="utf-8")
    return res_df, subset_audit


# ============================================================================
# TASK 3 (P1-2): EXACT ALGEBRAIC SNV-POOLING IDENTITY VERIFICATION
# ============================================================================
def run_task3_snv_algebra_check(primary_rows: pd.DataFrame) -> dict:
    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        X_raw = psp["statistics"][:, 0, :].astype(np.float64)
    rec = primary_rows["record_id"].to_numpy()
    max_err = 0.0
    for tid in np.unique(rec):
        loc = np.flatnonzero(rec == tid)
        xr = X_raw[loc]
        # Direct early1 = h(mean(x_i))
        e1 = snv(xr.mean(axis=0, keepdims=True))[0]
        # Weighted SNV sum: x_i = m_i + s_i * z_i -> mean(x_i) = mean(m_i) + (1/k) sum(s_i * z_i)
        # Since h(c*1 + v) == h(v) for any scalar offset c, h(mean(x_i)) == h(sum(s_i * z_i))!
        s_i = np.maximum(xr.std(axis=1, keepdims=True), 1e-6)
        z_i = snv(xr)
        e1_alg = snv((s_i * z_i).sum(axis=0, keepdims=True))[0]
        max_err = max(max_err, float(np.max(np.abs(e1 - e1_alg))))
    # Also check correlation between s_i (across-wavelength SD) and mean reflectance m_i across 3,627 kernels
    m_all = X_raw.mean(axis=1)
    s_all = X_raw.std(axis=1)
    corr_ms = float(np.corrcoef(m_all, s_all)[0, 1])
    out = dict(
        identity="h((1/k) sum x_i) == h(sum s_i z_i) for x_i = m_i*1 + s_i*z_i",
        max_abs_diff_across_67_trays=max_err,
        corr_kernel_mean_refl_vs_wavelength_sd=corr_ms,
        s_i_cv_across_kernels=float(s_all.std() / s_all.mean()),
    )
    (OUT_DIR / "snv_pooling_algebra_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT_DIR / "stage8.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    saved_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    rowpositions = pd.Series(np.arange(len(primary_rows)), index=primary_rows["seed_id"]).loc[saved_rows["seed_id"]].to_numpy()
    trays_df = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    boot_weights = np.load(LONG / "batch2_bootstrap.npz")["weights"]

    base_ext = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    entities = sorted(base_ext["entity_tray_id"].unique())
    entity_sampling = {}
    for t, ent in enumerate(entities):
        sids = base_ext.loc[base_ext["entity_tray_id"].eq(ent), "entity_seed_id"].tolist()
        rng = np.random.default_rng(20261114 + t)
        entity_sampling[ent] = (sids, np.stack([rng.permutation(len(sids)) for _ in range(500)]))

    ori_df, ori_sum = run_task1_orientation_audit(primary_rows, ext_rows)
    ctrl_df, sub_aud = run_task2_protocol2_control(
        primary_rows, ext_rows, trays_df, saved_rows, rowpositions, boot_weights, entity_sampling, entities
    )
    alg_chk = run_task3_snv_algebra_check(primary_rows)

    print("\n=== TASK 1 (P0-2): 127-TRAY INDEPENDENT ORIENTATION AUDIT ===")
    print(json.dumps(ori_sum, indent=2))

    print("\n=== TASK 2 (P1-1): SUBSET AUDIT ===")
    print(json.dumps(sub_aud, indent=2))

    print("\n=== TASK 2 (P1-1): CONTROLLED PROTOCOL 2 FACTORIZATION (k20->All vs All_exact) ===")
    cols = [
        "classifier", "route", "setting", "train_mode", "pca_dim_inner_min", "pca_dim_outer_min",
        "tray_weight_matched", "prim_all_ba", "prim_correct_trays", "delta_vs_late_pp",
        "aug_base_all", "aug_6cond_mean_all",
    ]
    print(ctrl_df[cols].round(4).to_string(index=False))

    print("\n=== TASK 3 (P1-2): EXACT ALGEBRAIC SNV-POOLING IDENTITY ===")
    print(json.dumps(alg_chk, indent=2))

    logging.info("Stage 8 COMPLETE in %.1fs.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
