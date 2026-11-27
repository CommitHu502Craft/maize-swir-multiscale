"""Stage 6: Bounded IPT Signal-Processing Revision Experiments (P0 + P1).

Implements strictly the finalized 15-item IPT revision plan without modifying any
frozen data, annotations, outputs/, or Stages 0-5 artifacts. All outputs are
written under analysis/path_b_plus/stage6_ipt_revision/.

Blocks:
  1. P0.1: Unified tray-equal single-kernel BA ranking analysis (2,000 paired
           tray-cluster bootstrap draws) + exact nested SVM split audit.
  2. P1.B: Wavelength-dependent paired affine residual curves across all 421
           SWIR bands (951.5-1650.0 nm) aggregated by entity tray.
  3. P1.A: Controlled SNV vs. non-SNV ablation on mean421 reflectance across all
           4 classifiers (Logistic, RBF-SVM, ExtraTrees, 1D-CNN), Primary +
           August 6 conditions (single-modality SWIR and MSD_raw fusion).
  4. P1.C & P1.D: Spectral-level pooling (early1: raw pool -> SNV -> classify;
           early2: per-kernel SNV -> pool -> classify) vs. probability-level
           pooling (late), under Protocol 1 (fixed single-kernel classifier,
           4 classifiers) and Protocol 2 (scale-matched multi-kernel spectral
           training inside outer training folds, Logistic & RBF-SVM).
  5. P1.5: Unified RGB comparator suite (M_raw, D, MD_raw, MD_jpeg) vs.
           representative SWIR and MSD_raw schemes across k and 6 conditions.
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
from scipy.stats import kendalltau, spearmanr
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
OUT_DIR = ROOT / "stage6_ipt_revision"

sys.path.insert(0, str(PROJECT))
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE
from night_pilot.robustness import assert_no_group_leakage
from night_pilot.seed_lots import assert_lot_declared

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


def make_classical_estimator(clf_name: str, params: dict, n_features: int, n_train: int):
    if clf_name == "logistic":
        n_comp = min(60, n_features, n_train - 1)
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, svd_solver="randomized", random_state=42),
            LogisticRegression(
                C=params["C"],
                max_iter=4000,
                class_weight="balanced",
                random_state=42,
            ),
        )
    if clf_name == "rbf_svm":
        n_comp = min(60, n_features, n_train - 1)
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, svd_solver="randomized", random_state=42),
            SVC(
                C=params["C"],
                gamma=params["gamma"],
                kernel="rbf",
                class_weight="balanced",
            ),
        )
    if clf_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=params["leaf"],
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=14,
        )
    raise ValueError(clf_name)


def fit_classical_model(
    clf_name: str,
    params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    rec_tr: np.ndarray,
    cal_splits: int,
    seed_offset: int,
):
    if clf_name == "rbf_svm":
        sgkf = StratifiedGroupKFold(n_splits=cal_splits, shuffle=True, random_state=SEED + seed_offset)
        cv_idx = list(sgkf.split(X_tr, y_tr, rec_tr))
        for a, b in cv_idx:
            assert not (set(rec_tr[a]) & set(rec_tr[b]))
        min_cal_tr = min(len(a) for a, _ in cv_idx)
        base = make_classical_estimator(clf_name, params, X_tr.shape[1], min_cal_tr)
        model = CalibratedClassifierCV(base, cv=cv_idx, n_jobs=min(3, cal_splits), ensemble=True)
        model.fit(X_tr, y_tr)
    else:
        base = make_classical_estimator(clf_name, params, X_tr.shape[1], len(X_tr))
        model = base
        model.fit(X_tr, y_tr)
    return model


def predict_aligned_proba(model, X: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    order = [int(np.flatnonzero(classes == lbl).item()) for lbl in LABELS]
    probs = np.asarray(model.predict_proba(X), dtype=np.float64)[:, order]
    probs = np.clip(probs, 0.0, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def train_cnn1d_single(
    tr_x: np.ndarray,
    tr_y_idx: np.ndarray,
    epochs: int,
    seed: int,
    val_x: np.ndarray | None = None,
    val_y_idx: np.ndarray | None = None,
):
    import torch
    from torch import nn

    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    class CNN1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Conv1d(16, 32, 5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 32, 5, stride=2, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten(),
                nn.Linear(512, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 12),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x[:, None, :])

    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(tr_x)
    tr_t = torch.tensor(scaler.transform(tr_x), dtype=torch.float32)
    yt = torch.tensor(tr_y_idx, dtype=torch.long)
    val_t = torch.tensor(scaler.transform(val_x), dtype=torch.float32) if val_x is not None else None

    counts = np.bincount(tr_y_idx, minlength=12)
    weights = torch.tensor(len(tr_y_idx) / (12.0 * counts), dtype=torch.float32)
    model = CNN1D()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_ba = -1.0
    best_epoch = 1
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(tr_y_idx))
        for j in range(0, len(order), 64):
            b = order[j : j + 64]
            opt.zero_grad()
            loss = loss_fn(model(tr_t[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_t is not None and val_y_idx is not None:
            model.eval()
            with torch.no_grad():
                vl = torch.cat([model(val_t[j : j + 256]) for j in range(0, len(val_t), 256)]).numpy()
            ba = float(balanced_accuracy_score(val_y_idx, vl.argmax(1)))
            if ba > best_ba:
                best_ba = ba
                best_epoch = ep

    if val_t is not None:
        return best_epoch, best_ba

    model.eval()
    return scaler, model


def predict_cnn1d_proba(scaler: StandardScaler, model, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    import torch

    model.eval()
    xt = torch.tensor(scaler.transform(X), dtype=torch.float32)
    outs = []
    with torch.no_grad():
        for j in range(0, len(xt), batch_size):
            logits = model(xt[j : j + batch_size]).numpy().astype(np.float64)
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            outs.append(p)
    return np.concatenate(outs, axis=0)


def compute_tray_mean_mu_star(p: np.ndarray, tray_idx_list: list[np.ndarray], truth_arr: np.ndarray, class_w: np.ndarray) -> float:
    mus = np.empty(len(tray_idx_list), dtype=np.float64)
    for t, (idx, tr) in enumerate(zip(tray_idx_list, truth_arr)):
        pt = p[idx]
        mu_all = (pt[:, [tr]] - pt).mean(axis=0)
        rivals = np.arange(12) != tr
        mus[t] = float(np.min(mu_all[rivals]))
    return float(mus @ class_w)


# ============================================================================
# BLOCK 1: P0 Unified Tray-Equal Single-Kernel Ranking + SVM Fold Audit
# ============================================================================
def run_block1_p0_unified_rankings_and_svm_audit(
    primary_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
    boot_draws: np.ndarray,
    boot_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rk_p = OUT_DIR / "p0_unified_tray_equal_rankings.csv"
    cfg_p = OUT_DIR / "p0_swir14_unified_summary.csv"
    svm_p = OUT_DIR / "p0_nested_svm_split_audit.json"
    if rk_p.exists() and cfg_p.exists() and svm_p.exists():
        logging.info("Block 1: SKIP cached unified rankings & SVM audit.")
        return pd.read_csv(rk_p), pd.read_csv(cfg_p), json.loads(svm_p.read_text(encoding="utf-8"))
    logging.info("Block 1: P0 unified tray-equal rankings & nested SVM split audit...")
    preds_dir = ROOT / "models_and_preds"
    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rf:
        d_july = rf["d_prob_july"]
    with np.load(preds_dir / "M_jpeg.npz") as a:
        m_jpeg = a["outer_oof_prob"]
    with np.load(preds_dir / "M_raw.npz") as a:
        m_raw = a["outer_oof_prob"]

    # Exact nested SVM split audit
    y_str = primary_rows["variety_name"].to_numpy()
    rec_str = primary_rows["record_id"].to_numpy()
    folds = primary_rows["fold"].to_numpy()
    svm_audit = dict(
        min_outer_train_trays_per_cultivar=999,
        min_inner_train_trays_per_cultivar=999,
        outer_cal3_min_train_trays=999,
        outer_cal3_min_val_trays=999,
        inner_cal2_min_train_trays=999,
        inner_cal2_min_val_trays=999,
        all_12_classes_every_split=True,
        zero_tray_overlap_every_split=True,
    )
    for f in range(4):
        outer_idx = np.flatnonzero(folds != f)
        inner_vals = primary_rows[f"inner_fold_for_outer{f}"].to_numpy()[outer_idx]
        c_ot = primary_rows.iloc[outer_idx].groupby("variety_name")["record_id"].nunique()
        svm_audit["min_outer_train_trays_per_cultivar"] = min(svm_audit["min_outer_train_trays_per_cultivar"], int(c_ot.min()))
        sg3 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + f * 100 + 7)
        for a, b in sg3.split(outer_idx, y_str[outer_idx], rec_str[outer_idx]):
            assert not (set(rec_str[outer_idx[a]]) & set(rec_str[outer_idx[b]]))
            c_tr = primary_rows.iloc[outer_idx[a]].groupby("variety_name")["record_id"].nunique()
            c_va = primary_rows.iloc[outer_idx[b]].groupby("variety_name")["record_id"].nunique()
            assert len(c_tr) == 12 and len(c_va) == 12
            svm_audit["outer_cal3_min_train_trays"] = min(svm_audit["outer_cal3_min_train_trays"], int(c_tr.min()))
            svm_audit["outer_cal3_min_val_trays"] = min(svm_audit["outer_cal3_min_val_trays"], int(c_va.min()))
        for g in range(3):
            tr_abs = outer_idx[inner_vals != g]
            c_it = primary_rows.iloc[tr_abs].groupby("variety_name")["record_id"].nunique()
            svm_audit["min_inner_train_trays_per_cultivar"] = min(svm_audit["min_inner_train_trays_per_cultivar"], int(c_it.min()))
            sg2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED + f * 10 + g)
            for a, b in sg2.split(tr_abs, y_str[tr_abs], rec_str[tr_abs]):
                assert not (set(rec_str[tr_abs[a]]) & set(rec_str[tr_abs[b]]))
                c_tr = primary_rows.iloc[tr_abs[a]].groupby("variety_name")["record_id"].nunique()
                c_va = primary_rows.iloc[tr_abs[b]].groupby("variety_name")["record_id"].nunique()
                assert len(c_tr) == 12 and len(c_va) == 12
                svm_audit["inner_cal2_min_train_trays"] = min(svm_audit["inner_cal2_min_train_trays"], int(c_tr.min()))
                svm_audit["inner_cal2_min_val_trays"] = min(svm_audit["inner_cal2_min_val_trays"], int(c_va.min()))
    (OUT_DIR / "p0_nested_svm_split_audit.json").write_text(json.dumps(svm_audit, indent=2) + "\n", encoding="utf-8")

    # Unified tray-equal single-kernel BA & cluster-bootstrap rank correlations
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]
    n_boot = len(boot_draws)

    def eval_unified(p: np.ndarray):
        pred = p.argmax(axis=1)
        right = (pred == y_july_idx).astype(np.float64)
        kba_kernel_eq = float(np.mean([right[y_july_idx == c].mean() for c in range(12)]))
        tray_k1_acc = np.array([right[idx].mean() for idx in tray_idx_list], dtype=np.float64)
        kba_tray_eq = float(tray_k1_acc @ class_w)
        kba_tray_eq_boot = tray_k1_acc @ boot_weights.T  # (2000,)

        correct = np.empty((6, 500, 67), dtype=bool)
        for t, trow in trays_df.iterrows():
            idx = tray_idx_list[t]
            n_k = len(idx)
            tr = truth_arr[t]
            perm = perms[:, t, :n_k]
            cum = np.cumsum(p[rowpositions[perm]], axis=1)
            p_all = p[idx].mean(axis=0)
            for ki, k in enumerate(KS):
                correct[ki, :, t] = (int(p_all.argmax()) == tr) if k == "All" else (cum[:, k - 1, :].argmax(axis=1) == tr)
        gmean = correct.mean(axis=1)  # (6, 67)
        gba_pt = gmean @ class_w
        bg = gmean @ boot_weights.T  # (6, 2000)
        return kba_kernel_eq, kba_tray_eq, kba_tray_eq_boot, gba_pt, bg

    swir_p = {}
    for cid in FACTORIAL_SWIR:
        with np.load(preds_dir / f"{cid}.npz") as a:
            swir_p[cid] = a["outer_oof_prob"]

    ensembles = {
        "SWIR_factorial_14": {cid: swir_p[cid] for cid in FACTORIAL_SWIR},
        "MSD_jpeg_factorial_14": {cid: (m_jpeg + d_july + swir_p[cid]) / 3.0 for cid in FACTORIAL_SWIR},
        "MSD_raw_factorial_14": {cid: (m_raw + d_july + swir_p[cid]) / 3.0 for cid in FACTORIAL_SWIR},
        "SWIR_fixed_mean421_4": {cid: swir_p[cid] for cid in ("S_logistic_mean", "S_rbf_svm_mean", "S_extra_trees_mean", "S_cnn1d_mean")},
    }

    rank_rows = []
    cfg_rows = []
    for ens_name, pmap in ensembles.items():
        names = list(pmap.keys())
        n_cfg = len(names)
        k_tray_pts = np.empty(n_cfg)
        k_tray_boots = np.empty((n_cfg, n_boot))
        g_pts = np.empty((n_cfg, 6))
        g_boots = np.empty((n_cfg, 6, n_boot))
        for i, nm in enumerate(names):
            kk, kt, ktb, gp, gb = eval_unified(pmap[nm])
            k_tray_pts[i] = kt
            k_tray_boots[i] = ktb
            g_pts[i] = gp
            g_boots[i] = gb
            if ens_name == "SWIR_factorial_14":
                cfg_rows.append(
                    dict(
                        configuration=nm,
                        kernel_ba_traditional=kk,
                        tray_equal_k1_ba=kt,
                        group_ba_k1_sched=float(gp[0]),
                        group_ba_k3=float(gp[1]),
                        group_ba_k10=float(gp[3]),
                        group_ba_k20=float(gp[4]),
                        group_ba_kAll=float(gp[5]),
                    )
                )

        for ki, k in enumerate(KS):
            # Round to 9 decimals to prevent 1-ULP floating-point noise from breaking exact rational BA ties
            xk_pt = np.round(k_tray_pts, 9)
            yk_pt = np.round(g_pts[:, ki], 9)
            rho_pt = float(spearmanr(xk_pt, yk_pt)[0])
            tau_pt = float(kendalltau(xk_pt, yk_pt)[0])
            n_ties = int(n_cfg - len(np.unique(np.round(g_pts[:, ki], 6))))
            rhos = np.empty(n_boot)
            taus = np.empty(n_boot)
            for b in range(n_boot):
                xb = np.round(k_tray_boots[:, b], 9)
                yb = np.round(g_boots[:, ki, b], 9)
                if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
                    rhos[b] = np.nan
                    taus[b] = np.nan
                else:
                    rhos[b] = spearmanr(xb, yb)[0]
                    taus[b] = kendalltau(xb, yb)[0]
            ok = np.isfinite(rhos)
            rank_rows.append(
                dict(
                    ensemble=ens_name,
                    reference="tray_equal_single_kernel_ba",
                    n_configs=n_cfg,
                    endpoint=f"k{k}",
                    n_tied_values_point=n_ties,
                    spearman_rho=rho_pt,
                    rho_ci_low=float(np.quantile(rhos[ok], 0.025)),
                    rho_ci_high=float(np.quantile(rhos[ok], 0.975)),
                    kendall_tau=tau_pt,
                    tau_ci_low=float(np.quantile(taus[ok], 0.025)),
                    tau_ci_high=float(np.quantile(taus[ok], 0.975)),
                    valid_boot_draws=int(ok.sum()),
                )
            )

    rk_df = pd.DataFrame(rank_rows)
    cfg_df = pd.DataFrame(cfg_rows)
    rk_df.to_csv(OUT_DIR / "p0_unified_tray_equal_rankings.csv", index=False)
    cfg_df.to_csv(OUT_DIR / "p0_swir14_unified_summary.csv", index=False)
    return rk_df, cfg_df, svm_audit


# ============================================================================
# BLOCK 2: P1.B Wavelength-Dependent Paired Affine Residual Curves
# ============================================================================
def run_block2_wavelength_affine_residuals(ext_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Block 2: Wavelength-dependent paired affine residual analysis (421 bands)...")
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        waves = esp["waves"].astype(np.float64)
        e_mean = esp["statistics"][:, 0, :].astype(np.float64)

    base_df = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    base_idx = base_df.index.to_numpy()
    sids_ordered = base_df["entity_seed_id"].to_numpy()
    ent_trays = base_df["entity_tray_id"].to_numpy()
    unique_trays = sorted(np.unique(ent_trays).tolist())

    X_base = e_mean[base_idx]  # (280, 421)
    wave_rows = []
    region_rows = []
    regions = [
        ("950-1100 nm", 950.0, 1100.0),
        ("1100-1250 nm", 1100.0, 1250.0),
        ("1250-1400 nm", 1250.0, 1400.0),
        ("1400-1550 nm", 1400.0, 1550.0),
        ("1550-1650 nm", 1550.0, 1650.0),
        ("Full 950-1650 nm", 950.0, 1651.0),
    ]

    for cond in ["REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]:
        c_df = ext_rows[ext_rows["condition"].eq(cond)].set_index("entity_seed_id").loc[sids_ordered]
        # Need original row positions in ext_rows
        c_lookup = dict(zip(ext_rows.loc[ext_rows["condition"].eq(cond), "entity_seed_id"],
                            ext_rows.index[ext_rows["condition"].eq(cond)].to_numpy()))
        ci = np.array([c_lookup[s] for s in sids_ordered], dtype=int)
        X_pert = e_mean[ci]

        res = np.empty_like(X_base)  # e_{i,lambda} = r_BASE - (a_i * r_pert + b_i)
        raw_diff = X_pert - X_base
        for i in range(len(sids_ordered)):
            xb = X_base[i]
            xp = X_pert[i]
            cb = xb - xb.mean()
            cp = xp - xp.mean()
            a = float(np.dot(cb, cp) / max(float(np.dot(cp, cp)), 1e-12))
            b = float(xb.mean() - a * xp.mean())
            res[i] = xb - (a * xp + b)

        # Aggregate by entity tray (10 trays)
        tray_mean_signed = np.empty((len(unique_trays), len(waves)), dtype=np.float64)
        tray_rms = np.empty((len(unique_trays), len(waves)), dtype=np.float64)
        tray_raw_rms = np.empty((len(unique_trays), len(waves)), dtype=np.float64)
        for ti, tid in enumerate(unique_trays):
            m = ent_trays == tid
            tray_mean_signed[ti] = res[m].mean(axis=0)
            tray_rms[ti] = np.sqrt(np.mean(res[m] ** 2, axis=0))
            tray_raw_rms[ti] = np.sqrt(np.mean(raw_diff[m] ** 2, axis=0))

        # Sign consistency and inter-kernel profile correlation with cohort mean residual profile
        mean_profile = tray_mean_signed.mean(axis=0)
        mp_c = mean_profile - mean_profile.mean()
        mp_norm = max(float(np.linalg.norm(mp_c)), 1e-12)
        kernel_corrs = np.empty(len(sids_ordered), dtype=np.float64)
        for i in range(len(sids_ordered)):
            ri = res[i] - res[i].mean()
            kernel_corrs[i] = float(np.dot(ri, mp_c) / max(float(np.linalg.norm(ri)) * mp_norm, 1e-12))

        tray_rms_mean = tray_rms.mean(axis=0)
        tray_rms_sd = tray_rms.std(axis=0, ddof=1)
        tray_signed_mean = tray_mean_signed.mean(axis=0)
        tray_signed_sd = tray_mean_signed.std(axis=0, ddof=1)
        frac_pos = (res > 0).mean(axis=0)
        sign_consistency = np.maximum(frac_pos, 1.0 - frac_pos)

        for wi, wv in enumerate(waves):
            wave_rows.append(
                dict(
                    condition=cond,
                    wavelength_nm=float(wv),
                    tray_mean_signed_resid=float(tray_signed_mean[wi]),
                    tray_sd_signed_resid=float(tray_signed_sd[wi]),
                    tray_mean_rms_resid=float(tray_rms_mean[wi]),
                    tray_sd_rms_resid=float(tray_rms_sd[wi]),
                    tray_q25_rms_resid=float(np.quantile(tray_rms[:, wi], 0.25)),
                    tray_q75_rms_resid=float(np.quantile(tray_rms[:, wi], 0.75)),
                    tray_mean_raw_rms=float(tray_raw_rms[:, wi].mean()),
                    frac_kernels_pos_resid=float(frac_pos[wi]),
                    sign_consistency=float(sign_consistency[wi]),
                )
            )

        peak_idx = int(np.argmax(tray_rms_mean))
        for rname, lo, hi in regions:
            rmask = (waves >= lo) & (waves <= hi)
            raw_r = float(np.sqrt(np.mean(tray_raw_rms[:, rmask] ** 2)))
            aff_r = float(np.sqrt(np.mean(tray_rms[:, rmask] ** 2)))
            region_rows.append(
                dict(
                    condition=cond,
                    region=rname,
                    mean_tray_rms_resid=float(tray_rms_mean[rmask].mean()),
                    max_tray_rms_resid=float(tray_rms_mean[rmask].max()),
                    peak_wavelength_nm=float(waves[rmask][int(np.argmax(tray_rms_mean[rmask]))]),
                    mean_raw_rms=raw_r,
                    affine_reduction_pct=float(100.0 * (1.0 - aff_r / max(raw_r, 1e-12))),
                    mean_abs_signed_resid=float(np.mean(np.abs(tray_signed_mean[rmask]))),
                    mean_sign_consistency=float(sign_consistency[rmask].mean()),
                    cohort_peak_nm=float(waves[peak_idx]),
                    mean_kernel_profile_corr=float(kernel_corrs.mean()),
                    median_kernel_profile_corr=float(np.median(kernel_corrs)),
                )
            )

    wdf = pd.DataFrame(wave_rows)
    rdf = pd.DataFrame(region_rows)
    wdf.to_csv(OUT_DIR / "p1_wavelength_affine_residuals.csv", index=False)
    rdf.to_csv(OUT_DIR / "p1_wavelength_affine_summary.csv", index=False)
    return wdf, rdf


# ============================================================================
# BLOCK 3: P1.A Controlled SNV vs. non-SNV Classification & Perturbation Ablation
# ============================================================================
def run_block3_snv_ablation(
    primary_rows: pd.DataFrame,
    ext_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
    boot_draws: np.ndarray,
    boot_weights: np.ndarray,
    entity_sampling: dict,
    entities: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Block 3: Controlled SNV vs. non-SNV ablation across 4 classifiers (mean421)...")
    pred_cache = OUT_DIR / "snv_ablation_preds"
    pred_cache.mkdir(parents=True, exist_ok=True)

    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        X_raw_july = psp["statistics"][:, 0, :].astype(np.float64)
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        X_raw_aug = esp["statistics"][:, 0, :].astype(np.float64)

    y_july_str = primary_rows["variety_name"].to_numpy()
    rec_july = primary_rows["record_id"].to_numpy()
    folds = primary_rows["fold"].to_numpy()

    grids = {
        "logistic": [{"C": c} for c in (0.1, 1.0, 10.0)],
        "rbf_svm": [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01)],
        "extra_trees": [{"leaf": l, "max_features": f} for l in (2, 5) for f in ("sqrt", 0.3)],
    }

    # Train & cache non-SNV models with exact Stage 1 nested tray-grouped CV protocol
    for clf_name, grid in grids.items():
        npz_path = pred_cache / f"S_noSNV_{clf_name}_mean.npz"
        meta_path = pred_cache / f"S_noSNV_{clf_name}_mean_meta.json"
        if npz_path.exists() and meta_path.exists():
            logging.info("  SKIP cached non-SNV %s", clf_name)
            continue
        t0 = time.monotonic()
        outer_oof_prob = np.zeros((len(primary_rows), 12), dtype=np.float64)
        ext_fold_prob = np.zeros((4, len(X_raw_aug), 12), dtype=np.float64)
        fold_meta = []
        for fold in range(4):
            outer_idx = np.flatnonzero(folds != fold)
            test_idx = np.flatnonzero(folds == fold)
            assert_no_group_leakage(set(rec_july[outer_idx]), set(rec_july[test_idx]))
            assert_lot_declared(set(rec_july[outer_idx]), set(rec_july[test_idx]))
            inner_vals = primary_rows[f"inner_fold_for_outer{fold}"].to_numpy()[outer_idx]

            best_ba = -1.0
            chosen = grid[0]
            for params in grid:
                pred_inner = np.empty(len(outer_idx), dtype=object)
                for g in range(3):
                    tr_abs = outer_idx[inner_vals != g]
                    va_abs = outer_idx[inner_vals == g]
                    est = make_classical_estimator(clf_name, params, X_raw_july.shape[1], len(tr_abs))
                    est.fit(X_raw_july[tr_abs], y_july_str[tr_abs])
                    pred_inner[inner_vals == g] = est.predict(X_raw_july[va_abs])
                sc = float(balanced_accuracy_score(y_july_str[outer_idx], pred_inner))
                if sc > best_ba:
                    best_ba = sc
                    chosen = params

            model = fit_classical_model(
                clf_name, chosen, X_raw_july[outer_idx], y_july_str[outer_idx], rec_july[outer_idx],
                cal_splits=3, seed_offset=fold * 100 + 7,
            )
            outer_oof_prob[test_idx] = predict_aligned_proba(model, X_raw_july[test_idx])
            ext_fold_prob[fold] = predict_aligned_proba(model, X_raw_aug)
            fold_meta.append(dict(fold=fold, chosen=chosen, inner_oof_ba=best_ba))

        overall_ba = float(balanced_accuracy_score(y_july_str, LABELS[outer_oof_prob.argmax(1)]))
        np.savez_compressed(npz_path, outer_oof_prob=outer_oof_prob, ext_fold_prob=ext_fold_prob)
        meta_path.write_text(json.dumps(dict(clf=clf_name, preprocessing="non_SNV", overall_oof_ba=overall_ba, folds=fold_meta), indent=2) + "\n", encoding="utf-8")
        logging.info("  DONE non-SNV %-12s OOF BA=%.4f (%.1fs)", clf_name, overall_ba, time.monotonic() - t0)

    # Train & cache non-SNV 1D-CNN
    cnn_npz = pred_cache / "S_noSNV_cnn1d_mean.npz"
    cnn_meta = pred_cache / "S_noSNV_cnn1d_mean_meta.json"
    if not (cnn_npz.exists() and cnn_meta.exists()):
        t0 = time.monotonic()
        outer_oof_prob = np.zeros((len(primary_rows), 12), dtype=np.float64)
        ext_fold_prob = np.zeros((4, len(X_raw_aug), 12), dtype=np.float64)
        fold_meta = []
        for fold in range(4):
            outer_idx = np.flatnonzero(folds != fold)
            test_idx = np.flatnonzero(folds == fold)
            inner_vals = primary_rows[f"inner_fold_for_outer{fold}"].to_numpy()[outer_idx]
            chosen_epochs = []
            for g in range(3):
                tr_abs = outer_idx[inner_vals != g]
                va_abs = outer_idx[inner_vals == g]
                be, _ = train_cnn1d_single(
                    X_raw_july[tr_abs], y_july_idx[tr_abs], epochs=80,
                    seed=SEED + fold * 100 + g * 10, val_x=X_raw_july[va_abs], val_y_idx=y_july_idx[va_abs],
                )
                chosen_epochs.append(be)
            sel_ep = max(10, min(80, int(np.round(np.mean(chosen_epochs)))))
            scaler, model = train_cnn1d_single(
                X_raw_july[outer_idx], y_july_idx[outer_idx], epochs=sel_ep, seed=SEED + fold * 1000 + 99
            )
            outer_oof_prob[test_idx] = predict_cnn1d_proba(scaler, model, X_raw_july[test_idx])
            ext_fold_prob[fold] = predict_cnn1d_proba(scaler, model, X_raw_aug)
            fold_meta.append(dict(fold=fold, inner_best_epochs=chosen_epochs, selected_epochs=sel_ep))
        overall_ba = float(balanced_accuracy_score(y_july_idx, outer_oof_prob.argmax(1)))
        np.savez_compressed(cnn_npz, outer_oof_prob=outer_oof_prob, ext_fold_prob=ext_fold_prob)
        cnn_meta.write_text(json.dumps(dict(clf="cnn1d", preprocessing="non_SNV", overall_oof_ba=overall_ba, folds=fold_meta), indent=2) + "\n", encoding="utf-8")
        logging.info("  DONE non-SNV cnn1d        OOF BA=%.4f (%.1fs)", overall_ba, time.monotonic() - t0)
    else:
        logging.info("  SKIP cached non-SNV cnn1d")

    # Evaluate SNV vs non-SNV across Primary + August 6 conditions (both SWIR-only and MSD_raw)
    preds_dir = ROOT / "models_and_preds"
    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rf:
        d_july, d_aug = rf["d_prob_july"], rf["d_prob_aug"]
    with np.load(preds_dir / "M_raw.npz") as a:
        m_raw_july = a["outer_oof_prob"]
        m_raw_aug = a["ext_fold_prob"].mean(axis=0)

    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]

    def eval_primary_full(p: np.ndarray):
        pred = p.argmax(axis=1)
        right = (pred == y_july_idx).astype(np.float64)
        kba = float(np.mean([right[y_july_idx == c].mean() for c in range(12)]))
        tray_k1 = np.array([right[idx].mean() for idx in tray_idx_list], dtype=np.float64)
        kba_te = float(tray_k1 @ class_w)
        kba_te_boot = tray_k1 @ boot_weights.T

        correct = np.empty((6, 500, 67), dtype=bool)
        for t, idx in enumerate(tray_idx_list):
            n_k = len(idx)
            tr = truth_arr[t]
            perm = perms[:, t, :n_k]
            cum = np.cumsum(p[rowpositions[perm]], axis=1)
            p_all = p[idx].mean(axis=0)
            for ki, k in enumerate(KS):
                correct[ki, :, t] = (int(p_all.argmax()) == tr) if k == "All" else (cum[:, k - 1, :].argmax(axis=1) == tr)
        gmean = correct.mean(axis=1)
        gba = gmean @ class_w
        bg = gmean @ boot_weights.T
        n_corr = int(correct[5, 0, :].sum())
        mu_star = compute_tray_mean_mu_star(p, tray_idx_list, truth_arr, class_w)
        return dict(
            kernel_ba=kba, tray_equal_k1_ba=kba_te,
            group_ba_k1=float(gba[0]), group_ba_k3=float(gba[1]), group_ba_k5=float(gba[2]),
            group_ba_k10=float(gba[3]), group_ba_k20=float(gba[4]), group_ba_kAll=float(gba[5]),
            correct_trays=n_corr, mean_mu_star=mu_star,
        ), kba_te_boot, bg

    ext_rows_idx = ext_rows.copy()
    ext_rows_idx["row_index"] = np.arange(len(ext_rows_idx))

    def eval_aug_6cond(p_aug: np.ndarray):
        cond_records = {}
        w_aug = np.full(10, 0.1, dtype=np.float64)
        for cond in CONDITIONS:
            block = ext_rows_idx[ext_rows_idx["condition"].eq(cond)]
            correct = np.empty((6, 500, 10), dtype=bool)
            tray_idxs = []
            tr_aug = np.empty(10, dtype=int)
            k_accs = []
            for t, ent in enumerate(entities):
                sids, pm_e = entity_sampling[ent]
                sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
                j = sub["row_index"].to_numpy(dtype=int)
                tr = LABEL_TO_IDX[sub.iloc[0]["variety"]]
                tr_aug[t] = tr
                tray_idxs.append(j)
                k_accs.append(float((p_aug[j].argmax(axis=1) == tr).mean()))
                cum = np.cumsum(p_aug[j[pm_e]], axis=1)
                p_all = p_aug[j].mean(axis=0)
                for ki, k in enumerate(KS):
                    correct[ki, :, t] = (int(p_all.argmax()) == tr) if k == "All" else (cum[:, k - 1, :].argmax(axis=1) == tr)
            gba = correct.mean(axis=1) @ w_aug
            mu_star = compute_tray_mean_mu_star(p_aug, tray_idxs, tr_aug, w_aug)
            cond_records[cond] = dict(
                kernel_ba=float(np.mean(k_accs)),
                group_ba_k1=float(gba[0]), group_ba_k3=float(gba[1]), group_ba_k5=float(gba[2]),
                group_ba_k10=float(gba[3]), group_ba_k20=float(gba[4]), group_ba_kAll=float(gba[5]),
                correct_trays=int(correct[5, 0, :].sum()),
                mean_mu_star=mu_star,
            )
        return cond_records

    prim_rows = []
    aug_rows = []
    contrast_rows = []

    for clf_name in ("logistic", "rbf_svm", "extra_trees", "cnn1d"):
        with np.load(preds_dir / f"S_{clf_name}_mean.npz") as a:
            p_snv_j = a["outer_oof_prob"]
            p_snv_a = a["ext_fold_prob"].mean(axis=0)
        with np.load(pred_cache / f"S_noSNV_{clf_name}_mean.npz") as a:
            p_raw_j = a["outer_oof_prob"]
            p_raw_a = a["ext_fold_prob"].mean(axis=0)

        for mode, pj_snv, pa_snv, pj_raw, pa_raw in [
            ("SWIR_only", p_snv_j, p_snv_a, p_raw_j, p_raw_a),
            ("MSD_raw", (m_raw_july + d_july + p_snv_j) / 3.0, (m_raw_aug + d_aug + p_snv_a) / 3.0,
                        (m_raw_july + d_july + p_raw_j) / 3.0, (m_raw_aug + d_aug + p_raw_a) / 3.0),
        ]:
            m_snv, kbt_snv, bg_snv = eval_primary_full(pj_snv)
            m_raw, kbt_raw, bg_raw = eval_primary_full(pj_raw)
            prim_rows.append(dict(mode=mode, classifier=clf_name, preprocessing="SNV", **m_snv))
            prim_rows.append(dict(mode=mode, classifier=clf_name, preprocessing="non_SNV", **m_raw))

            for ep_name, diff_boot, diff_pt in [
                ("tray_equal_k1", kbt_snv - kbt_raw, m_snv["tray_equal_k1_ba"] - m_raw["tray_equal_k1_ba"]),
                ("k3", bg_snv[1] - bg_raw[1], m_snv["group_ba_k3"] - m_raw["group_ba_k3"]),
                ("kAll", bg_snv[5] - bg_raw[5], m_snv["group_ba_kAll"] - m_raw["group_ba_kAll"]),
            ]:
                contrast_rows.append(
                    dict(
                        mode=mode, classifier=clf_name, endpoint=ep_name,
                        delta_snv_minus_non_snv=float(diff_pt),
                        ci_low=float(np.quantile(diff_boot, 0.025)),
                        ci_high=float(np.quantile(diff_boot, 0.975)),
                    )
                )

            for prep_lbl, pa in [("SNV", pa_snv), ("non_SNV", pa_raw)]:
                crec = eval_aug_6cond(pa)
                base_r = crec["BASE"]
                for cond in CONDITIONS:
                    r = crec[cond]
                    aug_rows.append(
                        dict(
                            mode=mode, classifier=clf_name, preprocessing=prep_lbl, condition=cond,
                            **r,
                            delta_vs_base_kernel=r["kernel_ba"] - base_r["kernel_ba"],
                            delta_vs_base_k3=r["group_ba_k3"] - base_r["group_ba_k3"],
                            delta_vs_base_kAll=r["group_ba_kAll"] - base_r["group_ba_kAll"],
                            delta_vs_base_mu_star=r["mean_mu_star"] - base_r["mean_mu_star"],
                        )
                    )

    df_p = pd.DataFrame(prim_rows)
    df_a = pd.DataFrame(aug_rows)
    df_c = pd.DataFrame(contrast_rows)
    df_p.to_csv(OUT_DIR / "p1_snv_ablation_primary.csv", index=False)
    df_a.to_csv(OUT_DIR / "p1_snv_ablation_august.csv", index=False)
    df_c.to_csv(OUT_DIR / "p1_snv_ablation_contrasts.csv", index=False)
    return df_p, df_a, df_c


# ============================================================================
# BLOCK 4: P1.C & P1.D Spectral-Level vs. Probability-Level Pooling
# ============================================================================
def _build_pooled_arrays_for_tray(
    X_raw: np.ndarray, X_snv: np.ndarray, idx: np.ndarray, perm_local: np.ndarray
) -> tuple[dict, dict]:
    """Build exact pooled spectra for early1 (raw pool -> SNV) and early2 (SNV -> pool)
    across k in {3, 5, 10, 20, All}.
    perm_local has shape (S, n_k) indexing into idx (0..n_k-1).
    """
    S = perm_local.shape[0]
    xr_t = X_raw[idx]  # (n_k, 421)
    xs_t = X_snv[idx]  # (n_k, 421)
    cum_r = np.cumsum(xr_t[perm_local], axis=1)  # (S, n_k, 421)
    cum_s = np.cumsum(xs_t[perm_local], axis=1)  # (S, n_k, 421)
    e1, e2 = {}, {}
    for k in (3, 5, 10, 20):
        e1[k] = snv(cum_r[:, k - 1, :] / float(k))
        e2[k] = cum_s[:, k - 1, :] / float(k)
    e1["All"] = snv(xr_t.mean(axis=0, keepdims=True))
    e2["All"] = xs_t.mean(axis=0, keepdims=True)
    return e1, e2


def run_block4_early_vs_late_pooling(
    primary_rows: pd.DataFrame,
    ext_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
    boot_draws: np.ndarray,
    boot_weights: np.ndarray,
    entity_sampling: dict,
    entities: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Block 4: Spectral-level (early1, early2) vs. probability-level (late) pooling...")
    cache_p1 = OUT_DIR / "p1_early_vs_late_primary.csv"
    cache_a1 = OUT_DIR / "p1_early_vs_late_august.csv"
    cache_c1 = OUT_DIR / "p1_early_vs_late_contrasts.csv"
    if cache_p1.exists() and cache_a1.exists() and cache_c1.exists():
        logging.info("  SKIP cached Block 4 outputs.")
        return pd.read_csv(cache_p1), pd.read_csv(cache_a1), pd.read_csv(cache_c1)

    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        raw_j_f32 = psp["statistics"][:, 0, :]
        X_raw_july = raw_j_f32.astype(np.float64)
        X_snv_july = snv(raw_j_f32).astype(np.float64)
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        raw_a_f32 = esp["statistics"][:, 0, :]
        X_raw_aug = raw_a_f32.astype(np.float64)
        X_snv_aug = snv(raw_a_f32).astype(np.float64)

    y_july_str = primary_rows["variety_name"].to_numpy()
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
        n_k = len(idx)
        tray_idx_list.append(idx)
        # Map global saved_rows permutation indices back to local 0..n_k-1 within this tray
        loc_map = {int(g): i for i, g in enumerate(local)}
        perm_global = perms[:, t, :n_k]
        tray_local_perms.append(np.vectorize(loc_map.__getitem__)(perm_global))

    # Precompute pooled primary spectra grouped by fold for fast batched prediction!
    # For each fold f, concatenate test trays' pooled spectra per k
    fold_test_trays = {f: [t for t, r in trays_df.iterrows() if tray_folds[str(r["record_id"])] == f] for f in range(4)}
    prim_pooled = {"early1": {}, "early2": {}}  # [route][(f, k)] = array (n_trays_f * S_k, 421)
    for f in range(4):
        for k in (3, 5, 10, 20, "All"):
            b1, b2 = [], []
            for t in fold_test_trays[f]:
                xr_t = X_raw_july[tray_idx_list[t]]
                xs_t = X_snv_july[tray_idx_list[t]]
                if k == "All":
                    b1.append(snv(xr_t.mean(axis=0, keepdims=True)))
                    b2.append(xs_t.mean(axis=0, keepdims=True))
                else:
                    pm = tray_local_perms[t][:, :k]  # (500, k)
                    b1.append(snv(xr_t[pm].mean(axis=1)))
                    b2.append(xs_t[pm].mean(axis=1))
            prim_pooled["early1"][(f, k)] = np.concatenate(b1, axis=0)
            prim_pooled["early2"][(f, k)] = np.concatenate(b2, axis=0)

    # Precompute pooled August spectra across all 6 conditions
    ext_rows_idx = ext_rows.copy()
    ext_rows_idx["row_index"] = np.arange(len(ext_rows_idx))
    aug_pooled = {"early1": {}, "early2": {}}  # [route][(cond, k)] = array (10 * S_k, 421)
    aug_truth = np.empty(10, dtype=int)
    aug_tray_idxs = {}
    for cond in CONDITIONS:
        block = ext_rows_idx[ext_rows_idx["condition"].eq(cond)]
        aug_tray_idxs[cond] = []
        for t, ent in enumerate(entities):
            sids, _ = entity_sampling[ent]
            sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
            aug_tray_idxs[cond].append(sub["row_index"].to_numpy(dtype=int))
            if cond == "BASE":
                aug_truth[t] = LABEL_TO_IDX[sub.iloc[0]["variety"]]
        for k in (3, 5, 10, 20, "All"):
            b1, b2 = [], []
            for t, ent in enumerate(entities):
                _, pm_e = entity_sampling[ent]
                j = aug_tray_idxs[cond][t]
                xr_t = X_raw_aug[j]
                xs_t = X_snv_aug[j]
                if k == "All":
                    b1.append(snv(xr_t.mean(axis=0, keepdims=True)))
                    b2.append(xs_t.mean(axis=0, keepdims=True))
                else:
                    pm = pm_e[:, :k]
                    b1.append(snv(xr_t[pm].mean(axis=1)))
                    b2.append(xs_t[pm].mean(axis=1))
            aug_pooled["early1"][(cond, k)] = np.concatenate(b1, axis=0)
            aug_pooled["early2"][(cond, k)] = np.concatenate(b2, axis=0)

    def predict_batched_parallel(model, X: np.ndarray, is_svm: bool) -> np.ndarray:
        if not is_svm or len(X) < 1000:
            return predict_aligned_proba(model, X)
        chunks = np.array_split(X, 14)
        res = Parallel(n_jobs=14, prefer="threads")(delayed(predict_aligned_proba)(model, ch) for ch in chunks)
        return np.concatenate(res, axis=0)

    def summarize_primary_from_correct(corr_mat: np.ndarray, nll_mat: np.ndarray):
        # corr_mat, nll_mat: (6, 500, 67)
        gmean = corr_mat.mean(axis=1)  # (6, 67)
        gba = gmean @ class_w
        gnll = nll_mat.mean(axis=1) @ class_w
        bg = gmean @ boot_weights.T  # (6, 2000)
        return gba, gnll, int(corr_mat[5, 0, :].sum()), bg

    prim_rows = []
    aug_rows = []
    contrast_rows = []
    preds_dir = ROOT / "models_and_preds"
    w_aug = np.full(10, 0.1, dtype=np.float64)

    # ---- PROTOCOL 1: Fixed single-kernel classifiers (all 4 families) ----
    late_boots_by_clf = {}
    late_gba_by_clf = {}

    for clf_name in ("logistic", "rbf_svm", "extra_trees", "cnn1d"):
        t0 = time.monotonic()
        meta = json.loads((preds_dir / f"S_{clf_name}_mean_meta.json").read_text(encoding="utf-8"))
        with np.load(preds_dir / f"S_{clf_name}_mean.npz") as a:
            ref_oof_p = a["outer_oof_prob"]
            ref_ext_p = a["ext_fold_prob"].mean(axis=0)

        # Refit exact 4 outer-fold models on SNV(mean421)
        fold_models = []
        check_oof = np.zeros_like(ref_oof_p)
        for fold in range(4):
            outer_idx = np.flatnonzero(folds != fold)
            test_idx = np.flatnonzero(folds == fold)
            if clf_name == "cnn1d":
                sel_ep = int(meta["folds"][fold]["selected_epochs"])
                sc, mdl = train_cnn1d_single(X_snv_july[outer_idx], y_july_idx[outer_idx], epochs=sel_ep, seed=SEED + fold * 1000 + 99)
                check_oof[test_idx] = predict_cnn1d_proba(sc, mdl, X_snv_july[test_idx])
                fold_models.append((sc, mdl))
            else:
                chosen = meta["folds"][fold]["chosen"]
                mdl = fit_classical_model(
                    clf_name, chosen, X_snv_july[outer_idx], y_july_str[outer_idx], rec_july[outer_idx],
                    cal_splits=3, seed_offset=fold * 100 + 7,
                )
                check_oof[test_idx] = predict_aligned_proba(mdl, X_snv_july[test_idx])
                fold_models.append(mdl)

        max_diff = float(np.max(np.abs(check_oof - ref_oof_p)))
        ref_ba = float(balanced_accuracy_score(y_july_idx, ref_oof_p.argmax(1)))
        chk_ba = float(balanced_accuracy_score(y_july_idx, check_oof.argmax(1)))
        logging.info("  Protocol 1 refit %-12s max_prob_diff=%.3g (ref_ba=%.4f chk_ba=%.4f)", clf_name, max_diff, ref_ba, chk_ba)
        if clf_name != "cnn1d":
            assert max_diff < 1e-6, (clf_name, max_diff)

        # Evaluate LATE (from ref_oof_p / ref_ext_p)
        corr_late = np.empty((6, 500, 67), dtype=bool)
        nll_late = np.empty((6, 500, 67), dtype=np.float64)
        for t, idx in enumerate(tray_idx_list):
            tr = truth_arr[t]
            pm = tray_local_perms[t]
            cum = np.cumsum(ref_oof_p[idx][pm], axis=1)
            p_all = ref_oof_p[idx].mean(axis=0)
            for ki, k in enumerate(KS):
                if k == "All":
                    corr_late[ki, :, t] = int(p_all.argmax()) == tr
                    nll_late[ki, :, t] = -np.log(max(float(p_all[tr]), EPS))
                else:
                    q = cum[:, k - 1, :] / float(k)
                    corr_late[ki, :, t] = q.argmax(axis=1) == tr
                    nll_late[ki, :, t] = -np.log(np.clip(q[:, tr], EPS, 1.0))
        gba_l, gnll_l, ntr_l, bg_l = summarize_primary_from_correct(corr_late, nll_late)
        late_boots_by_clf[clf_name] = bg_l
        late_gba_by_clf[clf_name] = gba_l
        prim_rows.append(dict(
            protocol="fixed_single_kernel", classifier=clf_name, route="late_prob_pool",
            **{f"group_ba_k{k}": float(gba_l[ki]) for ki, k in enumerate(KS)},
            **{f"group_nll_k{k}": float(gnll_l[ki]) for ki, k in enumerate(KS)},
            correct_trays=ntr_l,
        ))

        # Evaluate August LATE
        for cond in CONDITIONS:
            corr_a = np.empty((6, 500, 10), dtype=bool)
            nll_a = np.empty((6, 500, 10), dtype=np.float64)
            for t, ent in enumerate(entities):
                _, pm_e = entity_sampling[ent]
                j = aug_tray_idxs[cond][t]
                tr = aug_truth[t]
                cum = np.cumsum(ref_ext_p[j[pm_e]], axis=1)
                p_all = ref_ext_p[j].mean(axis=0)
                for ki, k in enumerate(KS):
                    if k == "All":
                        corr_a[ki, :, t] = int(p_all.argmax()) == tr
                        nll_a[ki, :, t] = -np.log(max(float(p_all[tr]), EPS))
                    else:
                        q = cum[:, k - 1, :] / float(k)
                        corr_a[ki, :, t] = q.argmax(axis=1) == tr
                        nll_a[ki, :, t] = -np.log(np.clip(q[:, tr], EPS, 1.0))
            gba_a = corr_a.mean(axis=1) @ w_aug
            gnll_a = nll_a.mean(axis=1) @ w_aug
            aug_rows.append(dict(
                protocol="fixed_single_kernel", classifier=clf_name, route="late_prob_pool", condition=cond,
                **{f"group_ba_k{k}": float(gba_a[ki]) for ki, k in enumerate(KS)},
                **{f"group_nll_k{k}": float(gnll_a[ki]) for ki, k in enumerate(KS)},
                correct_trays=int(corr_a[5, 0, :].sum()),
            ))

        # Evaluate EARLY1 & EARLY2 (Protocol 1 fixed single-kernel model)
        route_boots = {"late_prob_pool": bg_l}
        route_gbas = {"late_prob_pool": gba_l}
        for route in ("early1_raw_pool_snv", "early2_snv_pool"):
            rkey = "early1" if route.startswith("early1") else "early2"
            corr_e = corr_late.copy()  # k=1 is identical!
            nll_e = nll_late.copy()
            for f in range(4):
                t_list = fold_test_trays[f]
                for ki, k in [(1, 3), (2, 5), (3, 10), (4, 20), (5, "All")]:
                    Xp = prim_pooled[rkey][(f, k)]
                    if clf_name == "cnn1d":
                        sc, mdl = fold_models[f]
                        pp = predict_cnn1d_proba(sc, mdl, Xp)
                    else:
                        pp = predict_batched_parallel(fold_models[f], Xp, is_svm=(clf_name == "rbf_svm"))
                    S_k = 1 if k == "All" else 500
                    pp_r = pp.reshape(len(t_list), S_k, 12)
                    for lt, t in enumerate(t_list):
                        tr = truth_arr[t]
                        q = pp_r[lt]  # (S_k, 12)
                        corr_e[ki, :, t] = q.argmax(axis=1) == tr
                        nll_e[ki, :, t] = -np.log(np.clip(q[:, tr], EPS, 1.0))
            gba_e, gnll_e, ntr_e, bg_e = summarize_primary_from_correct(corr_e, nll_e)
            route_boots[route] = bg_e
            route_gbas[route] = gba_e
            prim_rows.append(dict(
                protocol="fixed_single_kernel", classifier=clf_name, route=route,
                **{f"group_ba_k{k}": float(gba_e[ki]) for ki, k in enumerate(KS)},
                **{f"group_nll_k{k}": float(gnll_e[ki]) for ki, k in enumerate(KS)},
                correct_trays=ntr_e,
            ))

            # August 6 conditions (average 4 outer-fold probabilities on pooled spectra)
            for cond in CONDITIONS:
                corr_a = np.empty((6, 500, 10), dtype=bool)
                nll_a = np.empty((6, 500, 10), dtype=np.float64)
                # k=1 identical to late
                for t, ent in enumerate(entities):
                    _, pm_e = entity_sampling[ent]
                    j = aug_tray_idxs[cond][t]
                    tr = aug_truth[t]
                    p1 = ref_ext_p[j[pm_e[:, 0]]]
                    corr_a[0, :, t] = p1.argmax(axis=1) == tr
                    nll_a[0, :, t] = -np.log(np.clip(p1[:, tr], EPS, 1.0))
                for ki, k in [(1, 3), (2, 5), (3, 10), (4, 20), (5, "All")]:
                    Xp = aug_pooled[rkey][(cond, k)]
                    pp_sum = np.zeros((len(Xp), 12), dtype=np.float64)
                    for f in range(4):
                        if clf_name == "cnn1d":
                            sc, mdl = fold_models[f]
                            pp_sum += predict_cnn1d_proba(sc, mdl, Xp)
                        else:
                            pp_sum += predict_batched_parallel(fold_models[f], Xp, is_svm=(clf_name == "rbf_svm"))
                    pp = pp_sum / 4.0
                    S_k = 1 if k == "All" else 500
                    pp_r = pp.reshape(10, S_k, 12)
                    for t in range(10):
                        tr = aug_truth[t]
                        q = pp_r[t]
                        corr_a[ki, :, t] = q.argmax(axis=1) == tr
                        nll_a[ki, :, t] = -np.log(np.clip(q[:, tr], EPS, 1.0))
                gba_a = corr_a.mean(axis=1) @ w_aug
                gnll_a = nll_a.mean(axis=1) @ w_aug
                aug_rows.append(dict(
                    protocol="fixed_single_kernel", classifier=clf_name, route=route, condition=cond,
                    **{f"group_ba_k{k}": float(gba_a[ki]) for ki, k in enumerate(KS)},
                    **{f"group_nll_k{k}": float(gnll_a[ki]) for ki, k in enumerate(KS)},
                    correct_trays=int(corr_a[5, 0, :].sum()),
                ))

        # Paired bootstrap contrasts for Protocol 1
        for ra, rb in [
            ("early1_raw_pool_snv", "late_prob_pool"),
            ("early2_snv_pool", "late_prob_pool"),
            ("early2_snv_pool", "early1_raw_pool_snv"),
        ]:
            for ki, k in [(1, 3), (2, 5), (3, 10), (4, 20), (5, "All")]:
                db = route_boots[ra][ki] - route_boots[rb][ki]
                contrast_rows.append(dict(
                    protocol="fixed_single_kernel", classifier=clf_name,
                    route_a=ra, route_b=rb, endpoint=f"k{k}",
                    delta_ba=float(route_gbas[ra][ki] - route_gbas[rb][ki]),
                    ci_low=float(np.quantile(db, 0.025)),
                    ci_high=float(np.quantile(db, 0.975)),
                ))
        logging.info("  Protocol 1 %-12s DONE (%.1fs)", clf_name, time.monotonic() - t0)

    # ---- PROTOCOL 2: Scale-matched multi-kernel spectral training (Logistic & RBF-SVM) ----
    # Strictly inside outer training trays: M=50 without-replacement draws per training tray for k in {3,5,10,20};
    # For All: evaluate BOTH exact full-tray training (1 spectrum/tray) AND k=20 scale-matched transfer to All!
    grids_p2 = {
        "logistic": [{"C": c} for c in (0.1, 1.0, 10.0)],
        "rbf_svm": [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01)],
    }
    M_TRAIN_DRAWS = 50

    def build_matched_train_set(outer_idx: np.ndarray, inner_vals_outer: np.ndarray, k_train, rkey: str, fold: int):
        # Strictly group by outer training trays
        u_trays = sorted(np.unique(rec_july[outer_idx]).tolist())
        X_list, y_list, rec_list, inf_list = [], [], [], []
        for ti, tid in enumerate(u_trays):
            loc_rel = np.flatnonzero(rec_july[outer_idx] == tid)
            loc_abs = outer_idx[loc_rel]
            lbl = y_july_str[loc_abs[0]]
            inf = int(inner_vals_outer[loc_rel[0]])
            xr = X_raw_july[loc_abs]
            xs = X_snv_july[loc_abs]
            if k_train == "All_exact":
                sp = snv(xr.mean(axis=0, keepdims=True)) if rkey == "early1" else xs.mean(axis=0, keepdims=True)
                X_list.append(sp)
                y_list.append(lbl)
                rec_list.append(tid)
                inf_list.append(inf)
            else:
                kk = min(int(k_train), len(loc_abs))
                rng = np.random.default_rng(SEED + fold * 10000 + int(k_train) * 100 + ti)
                idx_draws = np.stack([rng.permutation(len(loc_abs))[:kk] for _ in range(M_TRAIN_DRAWS)])
                if rkey == "early1":
                    sp = snv(xr[idx_draws].mean(axis=1))
                else:
                    sp = xs[idx_draws].mean(axis=1)
                X_list.append(sp)
                y_list.extend([lbl] * M_TRAIN_DRAWS)
                rec_list.extend([tid] * M_TRAIN_DRAWS)
                inf_list.extend([inf] * M_TRAIN_DRAWS)
        return np.concatenate(X_list, axis=0), np.asarray(y_list, dtype=object), np.asarray(rec_list, dtype=object), np.asarray(inf_list, dtype=int)

    for clf_name in ("logistic", "rbf_svm"):
        grid = grids_p2[clf_name]
        for rkey, route_base in [("early1", "early1_raw_pool_snv"), ("early2", "early2_snv_pool")]:
            t0 = time.monotonic()
            # Track primary & August arrays for matched(k) with All_exact AND matched_k20_to_All
            corr_exact = np.empty((6, 500, 67), dtype=bool)
            nll_exact = np.empty((6, 500, 67), dtype=np.float64)
            # Fill k=1 from fixed single-kernel late (since k=1 is single-kernel training!)
            with np.load(preds_dir / f"S_{clf_name}_mean.npz") as a:
                ref_oof_p = a["outer_oof_prob"]
                ref_ext_p = a["ext_fold_prob"].mean(axis=0)
            for t, idx in enumerate(tray_idx_list):
                tr = truth_arr[t]
                p1 = ref_oof_p[idx][tray_local_perms[t][:, 0]]
                corr_exact[0, :, t] = p1.argmax(axis=1) == tr
                nll_exact[0, :, t] = -np.log(np.clip(p1[:, tr], EPS, 1.0))

            aug_corr_exact = {c: np.empty((6, 500, 10), dtype=bool) for c in CONDITIONS}
            aug_nll_exact = {c: np.empty((6, 500, 10), dtype=np.float64) for c in CONDITIONS}
            for cond in CONDITIONS:
                for t, ent in enumerate(entities):
                    _, pm_e = entity_sampling[ent]
                    j = aug_tray_idxs[cond][t]
                    tr = aug_truth[t]
                    p1 = ref_ext_p[j[pm_e[:, 0]]]
                    aug_corr_exact[cond][0, :, t] = p1.argmax(axis=1) == tr
                    aug_nll_exact[cond][0, :, t] = -np.log(np.clip(p1[:, tr], EPS, 1.0))

            # Also track k=20 model evaluated on All
            corr_k20_all = np.empty((500, 67), dtype=bool)
            nll_k20_all = np.empty((500, 67), dtype=np.float64)
            aug_corr_k20_all = {c: np.empty((500, 10), dtype=bool) for c in CONDITIONS}
            aug_nll_k20_all = {c: np.empty((500, 10), dtype=np.float64) for c in CONDITIONS}

            for ki, k_eval, k_train in [(1, 3, 3), (2, 5, 5), (3, 10, 10), (4, 20, 20), (5, "All", "All_exact")]:
                fold_models_k = []
                for f in range(4):
                    outer_idx = np.flatnonzero(folds != f)
                    inner_vals = primary_rows[f"inner_fold_for_outer{f}"].to_numpy()[outer_idx]
                    X_tr_m, y_tr_m, rec_tr_m, inf_tr_m = build_matched_train_set(outer_idx, inner_vals, k_train, rkey, f)

                    def _eval_cand_fold(params, g):
                        tr_m = inf_tr_m != g
                        va_m = inf_tr_m == g
                        est = make_classical_estimator(clf_name, params, X_tr_m.shape[1], int(tr_m.sum()))
                        est.fit(X_tr_m[tr_m], y_tr_m[tr_m])
                        return va_m, est.predict(X_tr_m[va_m])

                    if len(grid) > 1:
                        tasks = [(pi, g) for pi in range(len(grid)) for g in range(3)]
                        outs = Parallel(n_jobs=14, backend="loky")(
                            delayed(_eval_cand_fold)(grid[pi], g) for pi, g in tasks
                        )
                        scores = []
                        for pi in range(len(grid)):
                            pred_in = np.empty(len(y_tr_m), dtype=object)
                            for g in range(3):
                                va_m, pr = outs[pi * 3 + g]
                                pred_in[va_m] = pr
                            scores.append(float(balanced_accuracy_score(y_tr_m, pred_in)))
                        best_i = int(np.argmax(scores))
                        best_ba = scores[best_i]
                        chosen = grid[best_i]
                    else:
                        chosen = grid[0]

                    mdl = fit_classical_model(
                        clf_name, chosen, X_tr_m, y_tr_m, rec_tr_m, cal_splits=3, seed_offset=f * 100 + 7
                    )
                    fold_models_k.append(mdl)

                    # Predict Primary fold f at k_eval
                    t_list = fold_test_trays[f]
                    Xp = prim_pooled[rkey][(f, k_eval)]
                    pp = predict_batched_parallel(mdl, Xp, is_svm=(clf_name == "rbf_svm"))
                    S_k = 1 if k_eval == "All" else 500
                    pp_r = pp.reshape(len(t_list), S_k, 12)
                    for lt, t in enumerate(t_list):
                        tr = truth_arr[t]
                        q = pp_r[lt]
                        corr_exact[ki, :, t] = q.argmax(axis=1) == tr
                        nll_exact[ki, :, t] = -np.log(np.clip(q[:, tr], EPS, 1.0))

                    # If k_train == 20, also predict Primary All using this k=20 trained model!
                    if k_train == 20:
                        Xp_all = prim_pooled[rkey][(f, "All")]
                        pp_all = predict_aligned_proba(mdl, Xp_all)
                        for lt, t in enumerate(t_list):
                            tr = truth_arr[t]
                            corr_k20_all[:, t] = int(pp_all[lt].argmax()) == tr
                            nll_k20_all[:, t] = -np.log(max(float(pp_all[lt, tr]), EPS))

                # Predict August 6 conditions at k_eval
                for cond in CONDITIONS:
                    Xp = aug_pooled[rkey][(cond, k_eval)]
                    pp_sum = np.zeros((len(Xp), 12), dtype=np.float64)
                    for f in range(4):
                        pp_sum += predict_batched_parallel(fold_models_k[f], Xp, is_svm=(clf_name == "rbf_svm"))
                    pp = pp_sum / 4.0
                    S_k = 1 if k_eval == "All" else 500
                    pp_r = pp.reshape(10, S_k, 12)
                    for t in range(10):
                        tr = aug_truth[t]
                        q = pp_r[t]
                        aug_corr_exact[cond][ki, :, t] = q.argmax(axis=1) == tr
                        aug_nll_exact[cond][ki, :, t] = -np.log(np.clip(q[:, tr], EPS, 1.0))

                    if k_train == 20:
                        Xp_all = aug_pooled[rkey][(cond, "All")]
                        pp_all = sum(predict_aligned_proba(fold_models_k[f], Xp_all) for f in range(4)) / 4.0
                        for t in range(10):
                            tr = aug_truth[t]
                            aug_corr_k20_all[cond][:, t] = int(pp_all[t].argmax()) == tr
                            aug_nll_k20_all[cond][:, t] = -np.log(max(float(pp_all[t, tr]), EPS))

            # Record Protocol 2 (exact All) and Protocol 2 (k20->All)
            for proto_name, c_all_override, n_all_override in [
                ("scale_matched_train_exactAll", None, None),
                ("scale_matched_train_k20toAll", corr_k20_all, nll_k20_all),
            ]:
                cm = corr_exact.copy()
                nm = nll_exact.copy()
                if c_all_override is not None:
                    cm[5] = c_all_override
                    nm[5] = n_all_override
                gba_m, gnll_m, ntr_m, bg_m = summarize_primary_from_correct(cm, nm)
                prim_rows.append(dict(
                    protocol=proto_name, classifier=clf_name, route=route_base,
                    **{f"group_ba_k{k}": float(gba_m[ki]) for ki, k in enumerate(KS)},
                    **{f"group_nll_k{k}": float(gnll_m[ki]) for ki, k in enumerate(KS)},
                    correct_trays=ntr_m,
                ))
                # Contrast vs fixed single-kernel late
                for ki, k in [(1, 3), (2, 5), (3, 10), (4, 20), (5, "All")]:
                    if proto_name == "scale_matched_train_k20toAll" and k != "All":
                        continue
                    db = bg_m[ki] - late_boots_by_clf[clf_name][ki]
                    contrast_rows.append(dict(
                        protocol=proto_name, classifier=clf_name,
                        route_a=route_base, route_b="late_prob_pool", endpoint=f"k{k}",
                        delta_ba=float(gba_m[ki] - late_gba_by_clf[clf_name][ki]),
                        ci_low=float(np.quantile(db, 0.025)),
                        ci_high=float(np.quantile(db, 0.975)),
                    ))

                for cond in CONDITIONS:
                    ca = aug_corr_exact[cond].copy()
                    na = aug_nll_exact[cond].copy()
                    if c_all_override is not None:
                        ca[5] = aug_corr_k20_all[cond]
                        na[5] = aug_nll_k20_all[cond]
                    gba_a = ca.mean(axis=1) @ w_aug
                    gnll_a = na.mean(axis=1) @ w_aug
                    aug_rows.append(dict(
                        protocol=proto_name, classifier=clf_name, route=route_base, condition=cond,
                        **{f"group_ba_k{k}": float(gba_a[ki]) for ki, k in enumerate(KS)},
                        **{f"group_nll_k{k}": float(gnll_a[ki]) for ki, k in enumerate(KS)},
                        correct_trays=int(ca[5, 0, :].sum()),
                    ))
            logging.info("  Protocol 2 %-10s %-20s DONE (%.1fs)", clf_name, route_base, time.monotonic() - t0)

    df_p = pd.DataFrame(prim_rows)
    df_a = pd.DataFrame(aug_rows)
    df_c = pd.DataFrame(contrast_rows)
    df_p.to_csv(cache_p1, index=False)
    df_a.to_csv(cache_a1, index=False)
    df_c.to_csv(cache_c1, index=False)
    return df_p, df_a, df_c


# ============================================================================
# BLOCK 5: P1.5 Unified RGB Comparator & Multimodal Table (Zero Retraining)
# ============================================================================
def run_block5_unified_rgb_summary() -> pd.DataFrame:
    logging.info("Block 5: Unified RGB comparator suite (M_raw, D, MD_raw, SWIR, MSD_raw)...")
    pm = pd.read_csv(ROOT / "stage2_diagnostics" / "primary_metrics.csv").set_index("configuration")
    ec = pd.read_csv(ROOT / "stage2_diagnostics" / "external_6cond_metrics.csv")

    selected = [
        ("RGB_M_jpeg", "M (JPEG-M)"),
        ("RGB_M_raw", "M (RAW-M)"),
        ("RGB_D_phys", "D (JPEG EfficientNet)"),
        ("RGB_MD_jpeg", "MD (JPEG-M + D)"),
        ("RGB_MD_raw", "MD (RAW-M + D)"),
        ("S_logistic_mean_std", "SWIR: Logistic (mean+SD)"),
        ("S_extra_trees_mean", "SWIR: ExtraTrees (mean)"),
        ("S_rbf_svm_mean_std", "SWIR: RBF-SVM (mean+SD)"),
        ("S_cnn1d_mean", "SWIR: 1D-CNN (mean)"),
        ("MSD_raw__S_logistic_median", "MSD_raw + Logistic (median)"),
        ("MSD_raw__S_logistic_mean_std", "MSD_raw + Logistic (mean+SD)"),
        ("MSD_raw__S_extra_trees_mean_std", "MSD_raw + ExtraTrees (mean+SD)"),
        ("MSD_raw__S_rbf_svm_mean_std", "MSD_raw + RBF-SVM (mean+SD)"),
        ("MSD_raw__S_cnn1d_mean", "MSD_raw + 1D-CNN (mean)"),
    ]
    rows = []
    for cfg, lbl in selected:
        pr = pm.loc[cfg]
        sub = ec[ec["configuration"].eq(cfg)].set_index("condition")
        rec = dict(
            configuration=cfg,
            label=lbl,
            prim_k1=float(pr["group_ba_k1"]),
            prim_k3=float(pr["group_ba_k3"]),
            prim_k5=float(pr["group_ba_k5"]),
            prim_k10=float(pr["group_ba_k10"]),
            prim_k20=float(pr["group_ba_k20"]),
            prim_kAll=float(pr["group_ba_kAll"]),
            prim_trays=int(pr["correct_trays"]),
            aug6_mean_k1=float(sub["group_ba_k1"].mean()),
            aug6_mean_k3=float(sub["group_ba_k3"].mean()),
            aug6_mean_k10=float(sub["group_ba_k10"].mean()),
            aug6_mean_kAll=float(sub["group_ba_kAll"].mean()),
        )
        for cond in CONDITIONS:
            rec[f"aug_{cond}_k3"] = float(sub.loc[cond, "group_ba_k3"])
            rec[f"aug_{cond}_kAll"] = float(sub.loc[cond, "group_ba_kAll"])
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "p1_unified_rgb_multimodal_summary.csv", index=False)
    return df


def main() -> None:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT_DIR / "stage6.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    y_july_idx = np.array([LABEL_TO_IDX[v] for v in primary_rows["variety_name"]], dtype=int)

    saved_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    rowpositions = pd.Series(np.arange(len(primary_rows)), index=primary_rows["seed_id"]).loc[saved_rows["seed_id"]].to_numpy()
    trays_df = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    perms = np.load(LONG / "batch2_schedules.npz")["permutations"]
    boot_npz = np.load(LONG / "batch2_bootstrap.npz")
    boot_draws, boot_weights = boot_npz["draws"], boot_npz["weights"]

    # Entity-paired August schedules (identical seed 20261114)
    base_ext = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    entities = sorted(base_ext["entity_tray_id"].unique())
    entity_sampling: dict[str, tuple[list[str], np.ndarray]] = {}
    for t, ent in enumerate(entities):
        sids = base_ext.loc[base_ext["entity_tray_id"].eq(ent), "entity_seed_id"].tolist()
        rng = np.random.default_rng(20261114 + t)
        entity_sampling[ent] = (sids, np.stack([rng.permutation(len(sids)) for _ in range(500)]))

    rk_df, cfg_df, svm_audit = run_block1_p0_unified_rankings_and_svm_audit(
        primary_rows, y_july_idx, trays_df, saved_rows, rowpositions, perms, boot_draws, boot_weights
    )
    wdf, rdf = run_block2_wavelength_affine_residuals(ext_rows)
    snv_p, snv_a, snv_c = run_block3_snv_ablation(
        primary_rows, ext_rows, y_july_idx, trays_df, saved_rows, rowpositions,
        perms, boot_draws, boot_weights, entity_sampling, entities
    )
    ep_p, ep_a, ep_c = run_block4_early_vs_late_pooling(
        primary_rows, ext_rows, y_july_idx, trays_df, saved_rows, rowpositions,
        perms, boot_draws, boot_weights, entity_sampling, entities
    )
    rgb_df = run_block5_unified_rgb_summary()

    # Print compact verification tables
    print("\n=== BLOCK 1: UNIFIED TRAY-EQUAL RANK CORRELATIONS (95% BOOTSTRAP CI) ===")
    print(rk_df[rk_df["endpoint"].isin(["k1", "k3", "k10", "k20", "kAll"])][
        ["ensemble", "endpoint", "n_tied_values_point", "spearman_rho", "rho_ci_low", "rho_ci_high", "kendall_tau"]
    ].round(4).to_string(index=False))

    print("\n=== BLOCK 2: WAVELENGTH-DEPENDENT AFFINE RESIDUAL SUMMARY (FULL & PEAK) ===")
    print(rdf[rdf["region"].eq("Full 950-1650 nm")][
        ["condition", "mean_tray_rms_resid", "max_tray_rms_resid", "cohort_peak_nm", "affine_reduction_pct", "mean_sign_consistency", "mean_kernel_profile_corr"]
    ].round(4).to_string(index=False))

    print("\n=== BLOCK 3: SNV vs NON-SNV ABLATION — PRIMARY (SWIR ONLY) ===")
    print(snv_p[snv_p["mode"].eq("SWIR_only")][
        ["classifier", "preprocessing", "kernel_ba", "group_ba_k3", "group_ba_kAll", "correct_trays", "mean_mu_star"]
    ].round(4).to_string(index=False))

    print("\n=== BLOCK 3: SNV vs NON-SNV ABLATION — AUGUST 6 CONDITIONS (SWIR ONLY, Full-Tray BA) ===")
    piv_a = snv_a[snv_a["mode"].eq("SWIR_only")].pivot(index=["classifier", "preprocessing"], columns="condition", values="group_ba_kAll")[CONDITIONS]
    piv_a["Mean6"] = piv_a.mean(axis=1)
    print(piv_a.round(3).to_string())

    print("\n=== BLOCK 3: SNV vs NON-SNV ABLATION — AUGUST 6 CONDITIONS (SWIR ONLY, Kernel BA) ===")
    piv_ak = snv_a[snv_a["mode"].eq("SWIR_only")].pivot(index=["classifier", "preprocessing"], columns="condition", values="kernel_ba")[CONDITIONS]
    piv_ak["Mean6"] = piv_ak.mean(axis=1)
    print(piv_ak.round(4).to_string())

    print("\n=== BLOCK 4: SPECTRAL-LEVEL (EARLY) vs PROBABILITY-LEVEL (LATE) POOLING — PRIMARY ===")
    print(ep_p[["protocol", "classifier", "route", "group_ba_k3", "group_ba_k10", "group_ba_k20", "group_ba_kAll", "correct_trays"]].round(4).to_string(index=False))

    print("\n=== BLOCK 4: EARLY vs LATE — AUGUST 6-COND MEAN (Full-Tray BA) ===")
    ea_m = ep_a.groupby(["protocol", "classifier", "route"], as_index=False)[["group_ba_k3", "group_ba_k10", "group_ba_kAll"]].mean()
    print(ea_m.round(4).to_string(index=False))

    logging.info("Stage 6 COMPLETE in %.1fs.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
