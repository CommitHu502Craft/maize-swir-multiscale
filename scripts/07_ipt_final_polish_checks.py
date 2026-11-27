"""Stage 7 (Final Pre-Submission Verification & Polish — zero heavy retraining):
1. Complete Table 4 metrics (Group BA, Group NLL, Group Brier, Group ECE) for ALL
   7 pooling/calibration strategies across Primary (k=3, All) and August (6-cond k=3,
   6-cond All, BASE All) so Table 4 has zero unnecessary blank cells.
2. Early-pooling mechanism check (Major 1):
   - Measure exact PCA-space feature variance ratio and Euclidean distance shift of
     k-pooled spectra (early1, early2, early3 = h(mean(h(x)))) vs single-kernel training
     spectra across k in {1, 3, 5, 10, 20, All}.
   - Evaluate early3 = f(h(mean(h(x)))) across all 4 fixed single-kernel classifiers
     to separate variance restoration (re-SNV) from raw-amplitude weighting.
3. Subset-count sensitivity (Major 2):
   - Evaluate Logistic early1 (k=20 -> All) across M in {1, 5, 10, 25, 50} pooled
     training subsets per outer-training tray (Primary & August 6-cond mean).
4. Per-entity-tray paired uncertainty on August 6 conditions (Major 5):
   - Aggregate each of the 10 August entity trays across its 6 imaging conditions first,
     then compute exact per-entity-tray paired differences (wins/ties/losses, mean +/- SD,
     exact sign/Wilcoxon test, and 10-entity-tray bootstrap 95% CI) for:
       (a) Logistic scale-matched early1 (k20->All) vs late_prob_pool
       (b) SNV vs non-SNV across all 4 classifiers (SWIR-only and MSD_raw)
       (c) MSD_raw + 1D-CNN (SNV) vs MD_raw.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
LONG = PROJECT / "analysis" / "long_term_validation" / "data"
S6 = ROOT / "stage6_ipt_revision"
OUT_DIR = ROOT / "stage7_final_polish"

sys.path.insert(0, str(PROJECT))
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE

LABELS = np.asarray(FROZEN_LABEL_SPACE)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
KS = [1, 3, 5, 10, 20, "All"]
CONDITIONS = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
SEED = 20261126
EPS = 1e-12

FACTORIAL_SWIR = [
    f"S_{clf}_{rep}"
    for clf in ("logistic", "rbf_svm", "extra_trees")
    for rep in ("median", "mean", "mean_std", "distribution")
] + ["S_cnn1d_median", "S_cnn1d_mean"]


def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)


def softmax_T(scores: np.ndarray, T: float) -> np.ndarray:
    z = scores / float(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def group_metrics_from_q(q_arr: np.ndarray, truth_vec: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    S, nt, _ = q_arr.shape
    ar_t = np.arange(nt)
    correct = (q_arr.argmax(axis=2) == truth_vec[None, :]).astype(np.float64)
    p_true = np.clip(q_arr[:, ar_t, truth_vec], EPS, 1.0)
    nll = -np.log(p_true)
    onehot = np.zeros_like(q_arr)
    onehot[:, ar_t, truth_vec] = 1.0
    brier = np.sum((q_arr - onehot) ** 2, axis=2)

    ba = float(correct.mean(axis=0) @ weights)
    gnll = float(nll.mean(axis=0) @ weights)
    gbrier = float(brier.mean(axis=0) @ weights)

    conf = q_arr.max(axis=2).reshape(-1)
    corr_f = correct.reshape(-1)
    w_f = np.tile(weights, S) / float(S)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bins[b], bins[b + 1]
        m = (conf >= lo) & (conf <= hi if b == 9 else conf < hi)
        wb = w_f[m].sum()
        if wb > 0:
            ece += wb * abs(np.sum(w_f[m] * corr_f[m]) / wb - np.sum(w_f[m] * conf[m]) / wb)
    return dict(group_ba=ba, group_nll=gnll, group_brier=gbrier, group_ece=float(ece))


# ============================================================================
# 1. COMPLETE TABLE 4 METRICS FOR ALL 7 POOLING/CALIBRATION STRATEGIES
# ============================================================================
def run_complete_table4(
    primary_rows: pd.DataFrame,
    ext_rows: pd.DataFrame,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
    entity_sampling: dict,
    entities: list[str],
) -> pd.DataFrame:
    logging.info("1. Computing complete Group BA/NLL/Brier/ECE for all 7 strategies (Table 4)...")
    preds_dir = ROOT / "models_and_preds"
    temps_s3 = pd.read_csv(ROOT / "stage3_calibration" / "fitted_temperatures.csv")
    temps_p1 = pd.read_csv(ROOT / "stage5_strengthening" / "p1_post_temp_bound_audit.csv")

    tray_folds = dict(zip(primary_rows["record_id"], primary_rows["fold"]))
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]
    folds_july = primary_rows["fold"].to_numpy()

    ext_rows_idx = ext_rows.copy()
    ext_rows_idx["row_index"] = np.arange(len(ext_rows_idx))
    w_aug = np.full(10, 0.1, dtype=np.float64)

    rows = []
    for cid in FACTORIAL_SWIR:
        with np.load(preds_dir / f"{cid}.npz") as a:
            outer_p = a["outer_oof_prob"]
            outer_s = a["outer_oof_score"]
            ext_fold_p = a["ext_fold_prob"]  # (4, 1680, 12)
            ext_fold_s = a["ext_fold_score"]

        ts3 = temps_s3[temps_s3["config_id"].eq(cid)].set_index("fold")
        tp1 = temps_p1[temps_p1["config_id"].eq(cid)].set_index(["fold", "k"])

        # Pre-pooling calibrated probabilities (Primary fold-wise, August fold-wise before averaging)
        outer_p_tk = np.empty_like(outer_p)
        outer_p_tg = np.empty_like(outer_p)
        ext_p_tk = np.zeros_like(ext_fold_p[0])
        ext_p_tg = np.zeros_like(ext_fold_p[0])
        for f in range(4):
            tk = float(ts3.loc[f, "T_kernel"])
            tg = float(ts3.loc[f, "T_group"])
            m = folds_july == f
            outer_p_tk[m] = softmax_T(outer_s[m], tk)
            outer_p_tg[m] = softmax_T(outer_s[m], tg)
            ext_p_tk += softmax_T(ext_fold_s[f], tk) / 4.0
            ext_p_tg += softmax_T(ext_fold_s[f], tg) / 4.0
        ext_p_raw = ext_fold_p.mean(axis=0)

        t_post_aug = {
            kv: float(np.exp(np.mean([np.log(float(tp1.loc[(f, kv), "T_post"])) for f in range(4)])))
            for kv in ("3", "All")
        }

        def pool_all_strats_primary(k_val):
            S = 1 if k_val == "All" else 500
            q_map = {
                s: np.empty((S, 67, 12), dtype=np.float64)
                for s in ("arith_raw", "majority_vote", "log_prob_pool", "cal_kernel_pre", "cal_group_pre", "post_cal_AllT", "post_cal_matched_kT")
            }
            for t, trow in trays_df.iterrows():
                idx = tray_idx_list[t]
                f = tray_folds[str(trow["record_id"])]
                t_all = float(tp1.loc[(f, "All"), "T_post"])
                t_k = float(tp1.loc[(f, str(k_val)), "T_post"])
                if k_val == "All":
                    qr = outer_p[idx].mean(axis=0)[None, :]
                    qv = np.eye(12)[outer_p[idx].argmax(axis=1)].mean(axis=0)[None, :]
                    ql_raw = np.log(np.clip(outer_p[idx], EPS, 1.0)).mean(axis=0)[None, :]
                    ql = softmax_T(ql_raw, 1.0)
                    qk = outer_p_tk[idx].mean(axis=0)[None, :]
                    qg = outer_p_tg[idx].mean(axis=0)[None, :]
                else:
                    pm = perms[:, t, : len(idx)][:, : int(k_val)]
                    p_sub = outer_p[rowpositions[pm]]  # (500, k, 12)
                    qr = p_sub.mean(axis=1)
                    qv = np.eye(12)[p_sub.argmax(axis=2)].mean(axis=1)
                    # Break ties in majority vote by tiny tie-breaker ONLY in argmax? Wait: check Brier/ECE on raw vote fractions
                    ql = softmax_T(np.log(np.clip(p_sub, EPS, 1.0)).mean(axis=1), 1.0)
                    qk = outer_p_tk[rowpositions[pm]].mean(axis=1)
                    qg = outer_p_tg[rowpositions[pm]].mean(axis=1)
                lq = np.log(np.clip(qr, EPS, 1.0))
                q_map["arith_raw"][:, t] = qr
                q_map["majority_vote"][:, t] = qv
                q_map["log_prob_pool"][:, t] = ql
                q_map["cal_kernel_pre"][:, t] = qk
                q_map["cal_group_pre"][:, t] = qg
                q_map["post_cal_AllT"][:, t] = softmax_T(lq, t_all)
                q_map["post_cal_matched_kT"][:, t] = softmax_T(lq, t_k)
            return q_map

        for kv in (3, "All"):
            q_map = pool_all_strats_primary(kv)
            for st, qa in q_map.items():
                if st == "majority_vote":
                    # Deterministic tie-breaking via arith_raw for BA (exact Stage 3 rule); raw vote fractions for Brier/ECE
                    qr = q_map["arith_raw"]
                    m = group_metrics_from_q(qa, truth_arr, class_w)
                    # Recompute BA with exact tie-breaking
                    tie_q = qa + 1e-9 * qr
                    corr_tb = (tie_q.argmax(axis=2) == truth_arr[None, :]).astype(np.float64)
                    m["group_ba"] = float(corr_tb.mean(axis=0) @ class_w)
                    m["group_nll"] = np.nan
                else:
                    m = group_metrics_from_q(qa, truth_arr, class_w)
                rows.append(dict(cohort="primary", condition="PRIMARY", config_id=cid, k=str(kv), strategy=st, **m))

        # August 6 conditions (k=3 and All)
        for cond in CONDITIONS:
            block = ext_rows_idx[ext_rows_idx["condition"].eq(cond)]
            tr_aug = np.empty(10, dtype=int)
            j_list = []
            pm_list = []
            for t, ent in enumerate(entities):
                sids, pm_e = entity_sampling[ent]
                sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
                j_list.append(sub["row_index"].to_numpy(dtype=int))
                pm_list.append(pm_e)
                tr_aug[t] = LABEL_TO_IDX[sub.iloc[0]["variety"]]

            for kv in (3, "All"):
                S = 1 if kv == "All" else 500
                q_map = {
                    s: np.empty((S, 10, 12), dtype=np.float64)
                    for s in ("arith_raw", "majority_vote", "log_prob_pool", "cal_kernel_pre", "cal_group_pre", "post_cal_AllT", "post_cal_matched_kT")
                }
                for t in range(10):
                    j = j_list[t]
                    if kv == "All":
                        qr = ext_p_raw[j].mean(axis=0)[None, :]
                        qv = np.eye(12)[ext_p_raw[j].argmax(axis=1)].mean(axis=0)[None, :]
                        ql = softmax_T(np.log(np.clip(ext_p_raw[j], EPS, 1.0)).mean(axis=0)[None, :], 1.0)
                        qk = ext_p_tk[j].mean(axis=0)[None, :]
                        qg = ext_p_tg[j].mean(axis=0)[None, :]
                    else:
                        pm = pm_list[t][:, : int(kv)]
                        p_sub = ext_p_raw[j[pm]]
                        qr = p_sub.mean(axis=1)
                        qv = np.eye(12)[p_sub.argmax(axis=2)].mean(axis=1)
                        ql = softmax_T(np.log(np.clip(p_sub, EPS, 1.0)).mean(axis=1), 1.0)
                        qk = ext_p_tk[j[pm]].mean(axis=1)
                        qg = ext_p_tg[j[pm]].mean(axis=1)
                    lq = np.log(np.clip(qr, EPS, 1.0))
                    q_map["arith_raw"][:, t] = qr
                    q_map["majority_vote"][:, t] = qv
                    q_map["log_prob_pool"][:, t] = ql
                    q_map["cal_kernel_pre"][:, t] = qk
                    q_map["cal_group_pre"][:, t] = qg
                    q_map["post_cal_AllT"][:, t] = softmax_T(lq, t_post_aug["All"])
                    q_map["post_cal_matched_kT"][:, t] = softmax_T(lq, t_post_aug[str(kv)])

                for st, qa in q_map.items():
                    if st == "majority_vote":
                        qr = q_map["arith_raw"]
                        m = group_metrics_from_q(qa, tr_aug, w_aug)
                        tie_q = qa + 1e-9 * qr
                        m["group_ba"] = float((tie_q.argmax(axis=2) == tr_aug[None, :]).mean(axis=0) @ w_aug)
                        m["group_nll"] = np.nan
                    else:
                        m = group_metrics_from_q(qa, tr_aug, w_aug)
                    rows.append(dict(cohort="august", condition=cond, config_id=cid, k=str(kv), strategy=st, **m))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table4_complete_all7.csv", index=False)
    return df


# ============================================================================
# 2. EARLY-POOLING MECHANISM CHECK: early3 = h(mean(h(x))) + PCA DISPERSION SHIFT
# ============================================================================
def run_early3_and_pca_shift(
    primary_rows: pd.DataFrame,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("2. Running early3 = h(mean(h(x))) control & PCA variance/distance diagnostics...")
    from analysis.path_b_plus import __name__ as _unused  # noqa: F401
    sys.path.insert(0, str(ROOT))
    import importlib
    s6 = importlib.import_module("06_ipt_signal_processing_revision")

    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        raw_j_f32 = psp["statistics"][:, 0, :]
        X_raw = raw_j_f32.astype(np.float64)
        X_snv = snv(raw_j_f32).astype(np.float64)

    y_july_str = primary_rows["variety_name"].to_numpy()
    y_july_idx = np.array([LABEL_TO_IDX[v] for v in y_july_str], dtype=int)
    rec_july = primary_rows["record_id"].to_numpy()
    folds = primary_rows["fold"].to_numpy()
    tray_folds = dict(zip(primary_rows["record_id"], primary_rows["fold"]))
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)

    tray_idx_list = []
    tray_local_perms = []
    for t, trow in trays_df.iterrows():
        local = np.flatnonzero(saved_rows["record_id"].eq(trow["record_id"]))
        idx = rowpositions[local]
        tray_idx_list.append(idx)
        loc_map = {int(g): i for i, g in enumerate(local)}
        tray_local_perms.append(np.vectorize(loc_map.__getitem__)(perms[:, t, : len(idx)]))

    fold_test_trays = {f: [t for t, r in trays_df.iterrows() if tray_folds[str(r["record_id"])] == f] for f in range(4)}

    # Build pooled arrays for early1, early2, early3 at k in {1, 3, 5, 10, 20, All} (using 100 schedules for speed on diagnostics, 500 for early3 BA!)
    pooled = {r: {} for r in ("early1", "early2", "early3")}
    for f in range(4):
        for k in (1, 3, 5, 10, 20, "All"):
            b1, b2, b3 = [], [], []
            for t in fold_test_trays[f]:
                xr = X_raw[tray_idx_list[t]]
                xs = X_snv[tray_idx_list[t]]
                if k == "All":
                    m_r = xr.mean(axis=0, keepdims=True)
                    m_s = xs.mean(axis=0, keepdims=True)
                    b1.append(snv(m_r))
                    b2.append(m_s)
                    b3.append(snv(m_s))
                else:
                    pm = tray_local_perms[t][:, : int(k)]
                    m_r = xr[pm].mean(axis=1)
                    m_s = xs[pm].mean(axis=1)
                    b1.append(snv(m_r))
                    b2.append(m_s)
                    b3.append(snv(m_s))
            pooled["early1"][(f, k)] = np.concatenate(b1, axis=0)
            pooled["early2"][(f, k)] = np.concatenate(b2, axis=0)
            pooled["early3"][(f, k)] = np.concatenate(b3, axis=0)

    # 2A: PCA-space variance ratio, within-spectrum SD, and distance to training mean across folds
    dist_rows = []
    for f in range(4):
        outer_idx = np.flatnonzero(folds != f)
        sc = StandardScaler().fit(X_snv[outer_idx])
        pca = PCA(n_components=60, svd_solver="randomized", random_state=42).fit(sc.transform(X_snv[outer_idx]))
        Z_tr = pca.transform(sc.transform(X_snv[outer_idx]))
        tr_total_var = float(np.var(Z_tr, axis=0).sum())
        tr_mean_norm = float(np.mean(np.linalg.norm(Z_tr, axis=1)))
        for rt in ("early1", "early2", "early3"):
            for k in (1, 3, 5, 10, 20, "All"):
                Xp = pooled[rt][(f, k)]
                spec_sd = float(np.mean(np.std(Xp, axis=1)))
                Zp = pca.transform(sc.transform(Xp))
                p_total_var = float(np.var(Zp, axis=0).sum())
                p_norm = float(np.mean(np.linalg.norm(Zp, axis=1)))
                dist_rows.append(dict(
                    fold=f, route=rt, k=str(k),
                    within_spectrum_sd=spec_sd,
                    pca60_var_ratio_vs_train=p_total_var / tr_total_var,
                    pca60_mean_dist_ratio_vs_train=p_norm / tr_mean_norm,
                ))
    dist_df = pd.DataFrame(dist_rows).groupby(["route", "k"], as_index=False)[
        ["within_spectrum_sd", "pca60_var_ratio_vs_train", "pca60_mean_dist_ratio_vs_train"]
    ].mean()
    dist_df.to_csv(OUT_DIR / "early_pooling_pca_shift.csv", index=False)

    # 2B: Evaluate early3 = f(h(mean(h(x)))) across all 4 fixed single-kernel classifiers (Primary)
    preds_dir = ROOT / "models_and_preds"
    e3_rows = []
    for clf_name in ("logistic", "rbf_svm", "extra_trees", "cnn1d"):
        meta = json.loads((preds_dir / f"S_{clf_name}_mean_meta.json").read_text(encoding="utf-8"))
        corr_e3 = np.empty((5, 500, 67), dtype=bool)  # k in {3, 5, 10, 20, All}
        for f in range(4):
            outer_idx = np.flatnonzero(folds != f)
            if clf_name == "cnn1d":
                sel_ep = int(meta["folds"][f]["selected_epochs"])
                sc_m, mdl = s6.train_cnn1d_single(X_snv[outer_idx], y_july_idx[outer_idx], epochs=sel_ep, seed=SEED + f * 1000 + 99)
            else:
                chosen = meta["folds"][f]["chosen"]
                mdl = s6.fit_classical_model(clf_name, chosen, X_snv[outer_idx], y_july_str[outer_idx], rec_july[outer_idx], cal_splits=3, seed_offset=f * 100 + 7)
            t_list = fold_test_trays[f]
            for ki, k in enumerate((3, 5, 10, 20, "All")):
                Xp = pooled["early3"][(f, k)]
                if clf_name == "cnn1d":
                    pp = s6.predict_cnn1d_proba(sc_m, mdl, Xp)
                elif clf_name == "rbf_svm" and len(Xp) >= 1000:
                    chunks = np.array_split(Xp, 14)
                    pp = np.concatenate(Parallel(n_jobs=14, prefer="threads")(delayed(s6.predict_aligned_proba)(mdl, ch) for ch in chunks), axis=0)
                else:
                    pp = s6.predict_aligned_proba(mdl, Xp)
                S_k = 1 if k == "All" else 500
                pp_r = pp.reshape(len(t_list), S_k, 12)
                for lt, t in enumerate(t_list):
                    corr_e3[ki, :, t] = pp_r[lt].argmax(axis=1) == truth_arr[t]
        gba = corr_e3.mean(axis=1) @ class_w
        e3_rows.append(dict(
            classifier=clf_name, route="early3_snv_pool_resnv",
            group_ba_k3=float(gba[0]), group_ba_k5=float(gba[1]),
            group_ba_k10=float(gba[2]), group_ba_k20=float(gba[3]),
            group_ba_kAll=float(gba[4]), correct_trays=int(corr_e3[4, 0, :].sum()),
        ))
    e3_df = pd.DataFrame(e3_rows)
    e3_df.to_csv(OUT_DIR / "early3_resnv_control.csv", index=False)
    return dist_df, e3_df


# ============================================================================
# 3. SUBSET-COUNT SENSITIVITY: M in {1, 5, 10, 25, 50} for Logistic early1 (k=20)
# ============================================================================
def run_subset_count_sensitivity(
    primary_rows: pd.DataFrame,
    ext_rows: pd.DataFrame,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
    entity_sampling: dict,
    entities: list[str],
) -> pd.DataFrame:
    logging.info("3. Running training subset-count sensitivity M in {1, 5, 10, 25, 50} (Logistic early1, k=20)...")
    sys.path.insert(0, str(ROOT))
    import importlib
    s6 = importlib.import_module("06_ipt_signal_processing_revision")

    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        X_raw_j = psp["statistics"][:, 0, :].astype(np.float64)
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        X_raw_a = esp["statistics"][:, 0, :].astype(np.float64)

    y_july_str = primary_rows["variety_name"].to_numpy()
    rec_july = primary_rows["record_id"].to_numpy()
    folds = primary_rows["fold"].to_numpy()
    tray_folds = dict(zip(primary_rows["record_id"], primary_rows["fold"]))
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)

    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]
    tray_local_perms = []
    for t, trow in trays_df.iterrows():
        local = np.flatnonzero(saved_rows["record_id"].eq(trow["record_id"]))
        loc_map = {int(g): i for i, g in enumerate(local)}
        tray_local_perms.append(np.vectorize(loc_map.__getitem__)(perms[:, t, : len(local)]))

    ext_rows_idx = ext_rows.copy()
    ext_rows_idx["row_index"] = np.arange(len(ext_rows_idx))
    w_aug = np.full(10, 0.1, dtype=np.float64)

    # Precompute test k=20 & All pooled spectra
    fold_test_trays = {f: [t for t, r in trays_df.iterrows() if tray_folds[str(r["record_id"])] == f] for f in range(4)}
    prim_k20 = {f: np.concatenate([snv(X_raw_j[tray_idx_list[t]][tray_local_perms[t][:, :20]].mean(axis=1)) for t in fold_test_trays[f]], axis=0) for f in range(4)}
    prim_all = {f: np.vstack([snv(X_raw_j[tray_idx_list[t]].mean(axis=0, keepdims=True)) for t in fold_test_trays[f]]) for f in range(4)}

    aug_all = {}
    aug_truth = np.empty(10, dtype=int)
    for cond in CONDITIONS:
        block = ext_rows_idx[ext_rows_idx["condition"].eq(cond)]
        arrs = []
        for t, ent in enumerate(entities):
            sids, _ = entity_sampling[ent]
            sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
            j = sub["row_index"].to_numpy(dtype=int)
            arrs.append(snv(X_raw_a[j].mean(axis=0, keepdims=True)))
            if cond == "BASE":
                aug_truth[t] = LABEL_TO_IDX[sub.iloc[0]["variety"]]
        aug_all[cond] = np.vstack(arrs)

    grid = [{"C": c} for c in (0.1, 1.0, 10.0)]
    rows = []
    for M in (1, 5, 10, 25, 50):
        corr_k20 = np.empty((500, 67), dtype=bool)
        corr_all = np.empty(67, dtype=bool)
        fold_models = []
        for f in range(4):
            outer_idx = np.flatnonzero(folds != f)
            inner_vals = primary_rows[f"inner_fold_for_outer{f}"].to_numpy()[outer_idx]
            u_trays = sorted(np.unique(rec_july[outer_idx]).tolist())
            X_l, y_l, r_l, i_l = [], [], [], []
            for ti, tid in enumerate(u_trays):
                loc_rel = np.flatnonzero(rec_july[outer_idx] == tid)
                loc_abs = outer_idx[loc_rel]
                xr = X_raw_j[loc_abs]
                kk = min(20, len(loc_abs))
                rng = np.random.default_rng(SEED + f * 10000 + 20 * 100 + ti)
                draws = np.stack([rng.permutation(len(loc_abs))[:kk] for _ in range(M)])
                X_l.append(snv(xr[draws].mean(axis=1)))
                y_l.extend([y_july_str[loc_abs[0]]] * M)
                r_l.extend([tid] * M)
                i_l.extend([int(inner_vals[loc_rel[0]])] * M)
            X_tr_m = np.concatenate(X_l, axis=0)
            y_tr_m = np.asarray(y_l, dtype=object)
            rec_tr_m = np.asarray(r_l, dtype=object)
            inf_tr_m = np.asarray(i_l, dtype=int)

            best_ba, chosen = -1.0, grid[0]
            for params in grid:
                pred_in = np.empty(len(y_tr_m), dtype=object)
                for g in range(3):
                    tr_m, va_m = inf_tr_m != g, inf_tr_m == g
                    est = s6.make_classical_estimator("logistic", params, X_tr_m.shape[1], int(tr_m.sum()))
                    est.fit(X_tr_m[tr_m], y_tr_m[tr_m])
                    pred_in[va_m] = est.predict(X_tr_m[va_m])
                sc = float(balanced_accuracy_score(y_tr_m, pred_in))
                if sc > best_ba:
                    best_ba, chosen = sc, params
            mdl = s6.fit_classical_model("logistic", chosen, X_tr_m, y_tr_m, rec_tr_m, cal_splits=3, seed_offset=f * 100 + 7)
            fold_models.append(mdl)

            t_list = fold_test_trays[f]
            p20 = s6.predict_aligned_proba(mdl, prim_k20[f]).reshape(len(t_list), 500, 12)
            pa = s6.predict_aligned_proba(mdl, prim_all[f])
            for lt, t in enumerate(t_list):
                corr_k20[:, t] = p20[lt].argmax(axis=1) == truth_arr[t]
                corr_all[t] = int(pa[lt].argmax()) == truth_arr[t]

        # August 6-cond (save per-entity-tray hits for M=50 too!)
        aug_tray_hits = np.empty((len(CONDITIONS), 10), dtype=np.float64)
        for ci, cond in enumerate(CONDITIONS):
            pp = sum(s6.predict_aligned_proba(fold_models[f], aug_all[cond]) for f in range(4)) / 4.0
            aug_tray_hits[ci] = (pp.argmax(axis=1) == aug_truth).astype(np.float64)

        rows.append(dict(
            M_subsets_per_tray=M,
            total_train_spectra_per_fold=int(M * 50),
            prim_k20_ba=float(corr_k20.mean(axis=0) @ class_w),
            prim_k20toAll_ba=float(corr_all.astype(np.float64) @ class_w),
            prim_correct_trays=int(corr_all.sum()),
            aug_base_all=float(aug_tray_hits[0].mean()),
            aug_6cond_mean_all=float(aug_tray_hits.mean()),
        ))
        if M == 50:
            np.save(OUT_DIR / "aug_tray_hits_logistic_early1_k20toAll.npy", aug_tray_hits)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "subset_count_sensitivity_logistic_k20.csv", index=False)
    return df


# ============================================================================
# 4. PER-ENTITY-TRAY PAIRED UNCERTAINTY ON AUGUST 6 CONDITIONS (n = 10 entities)
# ============================================================================
def run_august_entity_tray_uncertainty(ext_rows: pd.DataFrame, entity_sampling: dict, entities: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("4. Computing per-entity-tray paired uncertainty across the 10 August entity trays...")
    preds_dir = ROOT / "models_and_preds"
    snv_dir = S6 / "snv_ablation_preds"
    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rf:
        d_aug = rf["d_prob_aug"]
    with np.load(preds_dir / "M_raw.npz") as a:
        m_raw_aug = a["ext_fold_prob"].mean(axis=0)

    ext_rows_idx = ext_rows.copy()
    ext_rows_idx["row_index"] = np.arange(len(ext_rows_idx))
    aug_truth = np.empty(10, dtype=int)
    tray_j = {}
    for cond in CONDITIONS:
        block = ext_rows_idx[ext_rows_idx["condition"].eq(cond)]
        tray_j[cond] = []
        for t, ent in enumerate(entities):
            sids, _ = entity_sampling[ent]
            sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
            tray_j[cond].append(sub["row_index"].to_numpy(dtype=int))
            if cond == "BASE":
                aug_truth[t] = LABEL_TO_IDX[sub.iloc[0]["variety"]]

    def tray_hits_6cond(p_aug: np.ndarray) -> np.ndarray:
        out = np.empty((6, 10), dtype=np.float64)
        for ci, cond in enumerate(CONDITIONS):
            for t in range(10):
                out[ci, t] = float(int(p_aug[tray_j[cond][t]].mean(axis=0).argmax()) == aug_truth[t])
        return out

    hits = {}
    hits["MD_raw"] = tray_hits_6cond((m_raw_aug + d_aug) / 2.0)
    hits["M_raw"] = tray_hits_6cond(m_raw_aug)
    hits["D_phys"] = tray_hits_6cond(d_aug)
    for clf in ("logistic", "rbf_svm", "extra_trees", "cnn1d"):
        with np.load(preds_dir / f"S_{clf}_mean.npz") as a:
            ps = a["ext_fold_prob"].mean(axis=0)
        with np.load(snv_dir / f"S_noSNV_{clf}_mean.npz") as a:
            pr = a["ext_fold_prob"].mean(axis=0)
        hits[f"SWIR_{clf}_SNV"] = tray_hits_6cond(ps)
        hits[f"SWIR_{clf}_nonSNV"] = tray_hits_6cond(pr)
        hits[f"MSD_raw_{clf}_SNV"] = tray_hits_6cond((m_raw_aug + d_aug + ps) / 3.0)
        hits[f"MSD_raw_{clf}_nonSNV"] = tray_hits_6cond((m_raw_aug + d_aug + pr) / 3.0)
    hits["SWIR_logistic_early1_k20toAll"] = np.load(OUT_DIR / "aug_tray_hits_logistic_early1_k20toAll.npy")

    # Per-entity-tray 6-condition mean table
    ent_rows = []
    for t, ent in enumerate(entities):
        rec = dict(entity_tray_id=ent, cultivar=str(LABELS[aug_truth[t]]))
        for nm in [
            "MD_raw", "SWIR_logistic_SNV", "SWIR_logistic_early1_k20toAll",
            "SWIR_cnn1d_nonSNV", "SWIR_cnn1d_SNV", "MSD_raw_cnn1d_nonSNV", "MSD_raw_cnn1d_SNV",
        ]:
            rec[nm] = float(hits[nm][:, t].mean())
        ent_rows.append(rec)
    ent_df = pd.DataFrame(ent_rows)
    ent_df.to_csv(OUT_DIR / "august_per_entity_tray_6cond.csv", index=False)

    # Paired contrasts across the 10 entity trays (exact entity-tray cluster bootstrap, 10,000 draws + Wilcoxon)
    pairs = [
        ("SWIR_logistic_early1_k20toAll", "SWIR_logistic_SNV", "6-cond mean (All)"),
        ("SWIR_logistic_SNV", "SWIR_logistic_nonSNV", "Lamp-loss (LEFT+RIGHT)"),
        ("SWIR_rbf_svm_SNV", "SWIR_rbf_svm_nonSNV", "Lamp-loss (LEFT+RIGHT)"),
        ("SWIR_extra_trees_SNV", "SWIR_extra_trees_nonSNV", "Lamp-loss (LEFT+RIGHT)"),
        ("SWIR_cnn1d_SNV", "SWIR_cnn1d_nonSNV", "Lamp-loss (LEFT+RIGHT)"),
        ("MSD_raw_cnn1d_SNV", "MD_raw", "6-cond mean (All)"),
        ("MSD_raw_cnn1d_SNV", "MSD_raw_cnn1d_nonSNV", "Lamp-loss (LEFT+RIGHT)"),
    ]
    rng = np.random.default_rng(20261126)
    boot_idx = rng.integers(0, 10, size=(10000, 10))
    summary_rows = []
    for a_name, b_name, scope in pairs:
        c_slice = [4, 5] if "Lamp-loss" in scope else [0, 1, 2, 3, 4, 5]
        va = hits[a_name][c_slice].mean(axis=0)  # (10,)
        vb = hits[b_name][c_slice].mean(axis=0)  # (10,)
        d = va - vb
        b_means = d[boot_idx].mean(axis=1)
        wins = int((d > 1e-12).sum())
        ties = int((np.abs(d) <= 1e-12).sum())
        losses = int((d < -1e-12).sum())
        if np.any(np.abs(d) > 1e-12):
            pval = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
        else:
            pval = 1.0
        summary_rows.append(dict(
            model_a=a_name, model_b=b_name, scope=scope,
            mean_a=float(va.mean()), mean_b=float(vb.mean()),
            paired_delta_mean=float(d.mean()), paired_delta_sd=float(d.std(ddof=1)),
            median_delta=float(np.median(d)),
            iqr_low=float(np.quantile(d, 0.25)), iqr_high=float(np.quantile(d, 0.75)),
            boot_ci_low=float(np.quantile(b_means, 0.025)), boot_ci_high=float(np.quantile(b_means, 0.975)),
            wins_ties_losses=f"{wins}/{ties}/{losses}",
            wilcoxon_p=pval,
        ))
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUT_DIR / "august_entity_tray_paired_uncertainty.csv", index=False)
    return ent_df, sum_df


def main() -> None:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    saved_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    rowpositions = pd.Series(np.arange(len(primary_rows)), index=primary_rows["seed_id"]).loc[saved_rows["seed_id"]].to_numpy()
    trays_df = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    perms = np.load(LONG / "batch2_schedules.npz")["permutations"]

    base_ext = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    entities = sorted(base_ext["entity_tray_id"].unique())
    entity_sampling = {}
    for t, ent in enumerate(entities):
        sids = base_ext.loc[base_ext["entity_tray_id"].eq(ent), "entity_seed_id"].tolist()
        rng = np.random.default_rng(20261114 + t)
        entity_sampling[ent] = (sids, np.stack([rng.permutation(len(sids)) for _ in range(500)]))

    t4 = run_complete_table4(primary_rows, ext_rows, trays_df, saved_rows, rowpositions, perms, entity_sampling, entities)
    dist_df, e3_df = run_early3_and_pca_shift(primary_rows, trays_df, saved_rows, rowpositions, perms)
    m_df = run_subset_count_sensitivity(primary_rows, ext_rows, trays_df, saved_rows, rowpositions, perms, entity_sampling, entities)
    ent_df, unc_df = run_august_entity_tray_uncertainty(ext_rows, entity_sampling, entities)

    print("\n=== 1. COMPLETE TABLE 4 SUMMARY (14 SWIR MEAN) ===")
    for lbl, sub in [
        ("Primary k=3", t4[(t4["cohort"] == "primary") & (t4["k"] == "3")]),
        ("Primary All", t4[(t4["cohort"] == "primary") & (t4["k"] == "All")]),
        ("Aug 6-cond k=3", t4[(t4["cohort"] == "august") & (t4["k"] == "3")]),
        ("Aug 6-cond All", t4[(t4["cohort"] == "august") & (t4["k"] == "All")]),
        ("Aug BASE All", t4[(t4["cohort"] == "august") & (t4["condition"] == "BASE") & (t4["k"] == "All")]),
    ]:
        print(f"-- {lbl} --")
        print(sub.groupby("strategy", as_index=False)[["group_ba", "group_nll", "group_brier", "group_ece"]].mean().round(4).to_string(index=False))

    print("\n=== 2A. PCA-SPACE DISPERSION & DISTANCE SHIFT (EARLY1 / EARLY2 / EARLY3) ===")
    print(dist_df.round(4).to_string(index=False))

    print("\n=== 2B. EARLY3 = h(mean(h(x))) CONTROL (PRIMARY) ===")
    print(e3_df.round(4).to_string(index=False))

    print("\n=== 3. SUBSET-COUNT SENSITIVITY M in {1,5,10,25,50} (Logistic early1, k=20) ===")
    print(m_df.round(4).to_string(index=False))

    print("\n=== 4. AUGUST PER-ENTITY-TRAY PAIRED UNCERTAINTY (n=10 entity trays) ===")
    print(unc_df[["model_a", "model_b", "scope", "mean_a", "mean_b", "paired_delta_mean", "boot_ci_low", "boot_ci_high", "wins_ties_losses", "wilcoxon_p"]].round(4).to_string(index=False))

    logging.info("Stage 7 Final Polish COMPLETE in %.1fs.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
