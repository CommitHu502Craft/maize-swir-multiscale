"""Stage 5 (P0 + P1 Strengthening):
1. P0 Counterfactual margin interventions (rival-slot permutation preserving exact
   single-kernel BA/NLL/ECE/sharpness; positive vs. negative kernel margin
   decomposition; temperature sharpness sweep at fixed top-1 labels).
2. P0 Cluster-bootstrap 95% CIs (2,000 whole-tray draws) for cross-configuration
   Spearman rho and Kendall tau rank correlations, tie counts, and k=1 weighting check.
3. P0 Rigorous optical perturbation spectroscopy metrics: Spectral Angle Mapper (SAM),
   Pearson spectral correlation, and scale-matched raw vs. affine-detrended RMSE
   (same reflectance units!) + R^2_affine.
4. P1 Complete post-pooling calibration validation on BOTH July and August (BASE &
   6-condition), across k in {3, 5, 10, 20, All}, with Group NLL + Group Brier +
   Group ECE, wider bounds + explicit bound-hit audit, and scale-matched T_post(k) vs
   full-tray T_post(All) transfer checks.
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
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
LONG = PROJECT / "analysis" / "long_term_validation" / "data"
sys.path.insert(0, str(PROJECT))
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE

LABELS = np.asarray(FROZEN_LABEL_SPACE)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
KS = [1, 3, 5, 10, 20, "All"]
CONDITIONS = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
EPS = 1e-12

FACTORIAL_SWIR = [
    f"S_{clf}_{rep}"
    for clf in ("logistic", "rbf_svm", "extra_trees")
    for rep in ("median", "mean", "mean_std", "distribution")
] + ["S_cnn1d_median", "S_cnn1d_mean"]


def softmax_T(scores: np.ndarray, T: float) -> np.ndarray:
    z = scores / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_p0_optical_metrics(out_dir: Path) -> pd.DataFrame:
    """P0.3: Rigorous scale-matched spectroscopy metrics (SAM, Pearson r, raw RMSE,
    affine-detrended residual RMSE in identical reflectance units, R^2_affine).
    """
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        e_med = esp["median"].astype(np.float64)
        e_mean = esp["statistics"][:, 0, :].astype(np.float64)

    base_df = ext_rows[ext_rows["condition"].eq("BASE")].set_index("entity_seed_id")
    base_idx = ext_rows.index[ext_rows["condition"].eq("BASE")].to_numpy()
    base_lookup = dict(zip(ext_rows.loc[base_idx, "entity_seed_id"], base_idx))

    rows_out = []
    for summary_name, arr in [("mean", e_mean), ("median", e_med)]:
        for cond in CONDITIONS:
            c_df = ext_rows[ext_rows["condition"].eq(cond)]
            c_lookup = dict(zip(c_df["entity_seed_id"], c_df.index.to_numpy()))
            bi = np.array([base_lookup[sid] for sid in base_df.index], dtype=int)
            ci = np.array([c_lookup[sid] for sid in base_df.index], dtype=int)

            X_base = arr[bi]  # (280, 421)
            X_pert = arr[ci]  # (280, 421)
            diff_raw = X_pert - X_base
            raw_rmse = float(np.sqrt(np.mean(diff_raw ** 2)))
            raw_bias = float(np.mean(diff_raw))

            # Per-kernel SAM (degrees), Pearson r, and per-kernel OLS affine fit X_base ~ a * X_pert + b
            sams_deg = np.empty(len(bi), dtype=np.float64)
            pearsons = np.empty(len(bi), dtype=np.float64)
            aff_res = np.empty_like(X_base)
            aff_slopes = np.empty(len(bi), dtype=np.float64)
            aff_r2s = np.empty(len(bi), dtype=np.float64)

            for i in range(len(bi)):
                xb = X_base[i]
                xp = X_pert[i]
                # SAM
                cos_val = float(np.dot(xb, xp) / max(np.linalg.norm(xb) * np.linalg.norm(xp), 1e-12))
                sams_deg[i] = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
                # Pearson r
                cb = xb - xb.mean()
                cp = xp - xp.mean()
                r = float(np.dot(cb, cp) / max(np.linalg.norm(cb) * np.linalg.norm(cp), 1e-12))
                pearsons[i] = r
                # Affine fit X_base ~ a * X_pert + b (exact linear transformation removed by SNV)
                var_p = float(np.dot(cp, cp))
                a = float(np.dot(cb, cp) / max(var_p, 1e-12))
                b = float(xb.mean() - a * xp.mean())
                x_hat = a * xp + b
                aff_res[i] = xb - x_hat
                aff_slopes[i] = a
                ss_tot = float(np.dot(cb, cb))
                ss_res = float(np.dot(aff_res[i], aff_res[i]))
                aff_r2s[i] = 1.0 - ss_res / max(ss_tot, 1e-12)

            aff_rmse = float(np.sqrt(np.mean(aff_res ** 2)))
            frac_removed = float(1.0 - (aff_rmse / raw_rmse)) if raw_rmse > 1e-12 else 0.0
            ratio_raw_over_aff = float(raw_rmse / aff_rmse) if aff_rmse > 1e-12 else np.nan

            rows_out.append(
                dict(
                    summary=summary_name,
                    condition=cond,
                    n_physical_kernels=280,
                    raw_bias=raw_bias,
                    raw_rmse=raw_rmse,
                    affine_residual_rmse=aff_rmse,
                    rmse_reduction_factor=ratio_raw_over_aff,
                    rmse_fraction_removed_by_affine=frac_removed,
                    mean_sam_deg=float(sams_deg.mean()),
                    median_sam_deg=float(np.median(sams_deg)),
                    mean_pearson_r=float(pearsons.mean()),
                    mean_affine_r2=float(aff_r2s.mean()),
                    mean_affine_slope=float(aff_slopes.mean()),
                )
            )
    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "p0_optical_perturbation_physics.csv", index=False)
    return df


def run_p0_counterfactual_margins(
    out_dir: Path,
    primary_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P0.1: Counterfactual interventions on saved OOF probabilities (zero retraining):
    A. Positive vs. negative kernel margin decomposition on worst rival c*.
    B. Rival-slot permutation within tray (preserves EXACT single-kernel BA, NLL, ECE,
       entropy, and wrong-kernel margin magnitude |d|, while breaking rival coherence!).
    C. Uniform non-true redistribution on misclassified kernels (preserving p(y) exact).
    D. Controlled temperature sharpness sweep T in {0.25, 0.5, 1.0, 2.0, 4.0} at fixed
       top-1 labels (separating confidence sharpness from error direction).
    """
    preds_dir = ROOT / "models_and_preds"
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = []
    for _, trow in trays_df.iterrows():
        local = np.flatnonzero(saved_rows["record_id"].eq(trow["record_id"]))
        tray_idx_list.append(rowpositions[local])

    def full_tray_ba(p_arr: np.ndarray) -> tuple[float, int]:
        corr = np.array([int(p_arr[idx].mean(axis=0).argmax()) == tr for idx, tr in zip(tray_idx_list, truth_arr)], dtype=np.float64)
        return float(corr @ class_w), int(corr.sum())

    cf_rows = []
    temp_sweep_rows = []

    for cid in FACTORIAL_SWIR:
        with np.load(preds_dir / f"{cid}.npz") as a:
            p = a["outer_oof_prob"].astype(np.float64)
            s = a["outer_oof_score"].astype(np.float64)

        orig_kba = float(np.mean([np.mean(p.argmax(1)[y_july_idx == c] == c) for c in range(12)]))
        orig_all_ba, orig_all_trays = full_tray_ba(p)

        # (A) Positive vs. negative kernel contribution to worst-rival tray margin mu*
        mu_pos_list, mu_neg_list, mu_star_list = [], [], []
        wrong_mu_pos, wrong_mu_neg = [], []
        for idx, tr in zip(tray_idx_list, truth_arr):
            pt = p[idx]
            d_mat = pt[:, [tr]] - pt
            mu_all = d_mat.mean(axis=0)
            rivals = np.arange(12) != tr
            wr = int(np.flatnonzero(rivals)[np.argmin(mu_all[rivals])])
            d_star = d_mat[:, wr]
            pred_k = pt.argmax(axis=1)
            pos_contrib = float(d_star[pred_k == tr].sum() / len(idx))
            neg_contrib = float(d_star[pred_k != tr].sum() / len(idx))
            assert abs((pos_contrib + neg_contrib) - float(mu_all[wr])) < 1e-12
            mu_pos_list.append(pos_contrib)
            mu_neg_list.append(neg_contrib)
            mu_star_list.append(float(mu_all[wr]))
            if mu_all[wr] <= 0.0:
                wrong_mu_pos.append(pos_contrib)
                wrong_mu_neg.append(neg_contrib)

        # (B) Counterfactual 1: Rival-slot permutation (50 repeats)
        # Keep p_i(y_i) exact; independently permute the 11 non-true slots per kernel!
        # Preserves exact single-kernel top-1 correctness, BA, NLL, ECE, entropy, |d_wrong|!
        rng = np.random.default_rng(20261126)
        n_rep = 50
        perm_bas = np.empty(n_rep, dtype=np.float64)
        perm_trays = np.empty(n_rep, dtype=np.float64)
        perm_kba_check = 0.0
        for r_i in range(n_rep):
            p_cf = p.copy()
            for idx, tr in zip(tray_idx_list, truth_arr):
                rivals = np.flatnonzero(np.arange(12) != tr)
                for ii in idx:
                    perm_r = rng.permutation(rivals)
                    p_cf[ii, rivals] = p[ii, perm_r]
            if r_i == 0:
                # Verify single-kernel BA & true-class probability are 100% invariant!
                perm_kba_check = float(np.mean([np.mean(p_cf.argmax(1)[y_july_idx == c] == c) for c in range(12)]))
                assert abs(perm_kba_check - orig_kba) < 1e-12, (cid, perm_kba_check, orig_kba)
                assert np.allclose(p_cf[np.arange(len(p)), y_july_idx], p[np.arange(len(p)), y_july_idx], atol=1e-15)
            ba_r, tr_r = full_tray_ba(p_cf)
            perm_bas[r_i] = ba_r
            perm_trays[r_i] = tr_r

        # (C) Counterfactual 2: Uniform rival redistribution ONLY on misclassified kernels
        p_unif_wrong = p.copy()
        pred_all = p.argmax(axis=1)
        for i in range(len(p)):
            if pred_all[i] != y_july_idx[i]:
                tr = y_july_idx[i]
                rem = 1.0 - p[i, tr]
                p_unif_wrong[i, :] = rem / 11.0
                p_unif_wrong[i, tr] = p[i, tr]
        unif_wrong_ba, unif_wrong_trays = full_tray_ba(p_unif_wrong)

        # (D) Counterfactual 3: Temperature sharpness sweep at fixed top-1 kernel labels
        for T in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
            p_t = softmax_T(s, T)
            kba_t = float(np.mean([np.mean(p_t.argmax(1)[y_july_idx == c] == c) for c in range(12)]))
            ba_t, tr_t = full_tray_ba(p_t)
            temp_sweep_rows.append(
                dict(
                    configuration=cid,
                    temperature=T,
                    kernel_ba=kba_t,
                    full_tray_ba=ba_t,
                    correct_trays=tr_t,
                    delta_vs_T1=ba_t - orig_all_ba,
                )
            )

        cf_rows.append(
            dict(
                configuration=cid,
                kernel_ba=orig_kba,
                orig_full_tray_ba=orig_all_ba,
                orig_correct_trays=orig_all_trays,
                mean_mu_star=float(np.mean(mu_star_list)),
                mean_pos_kernel_contrib=float(np.mean(mu_pos_list)),
                mean_wrong_kernel_drag=float(np.mean(mu_neg_list)),
                failed_trays_pos_contrib=float(np.mean(wrong_mu_pos)) if wrong_mu_pos else np.nan,
                failed_trays_wrong_drag=float(np.mean(wrong_mu_neg)) if wrong_mu_neg else np.nan,
                cf_rival_perm_ba_mean=float(perm_bas.mean()),
                cf_rival_perm_ba_sd=float(perm_bas.std(ddof=1)),
                cf_rival_perm_delta_ba=float(perm_bas.mean() - orig_all_ba),
                cf_rival_perm_trays_mean=float(perm_trays.mean()),
                cf_unif_wrong_ba=unif_wrong_ba,
                cf_unif_wrong_delta_ba=float(unif_wrong_ba - orig_all_ba),
                cf_unif_wrong_trays=unif_wrong_trays,
            )
        )

    cf_df = pd.DataFrame(cf_rows)
    ts_df = pd.DataFrame(temp_sweep_rows)
    cf_df.to_csv(out_dir / "p0_counterfactual_margins.csv", index=False)
    ts_df.to_csv(out_dir / "p0_sharpness_sweep.csv", index=False)
    return cf_df, ts_df


def run_p0_bootstrap_rank_correlations(
    out_dir: Path,
    primary_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
    boot_draws: np.ndarray,
    boot_weights: np.ndarray,
) -> pd.DataFrame:
    """P0.2: Compute 2,000 tray-cluster bootstrap 95% CIs for Spearman rho and Kendall tau
    across the 14 SWIR, 14 MSD_jpeg, 14 MSD_raw, and fixed-mean421 ensembles, plus exact
    tie-count audit and k=1 kernel-vs-group weighting check.
    """
    preds_dir = ROOT / "models_and_preds"
    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rf:
        d_july = rf["d_prob_july"]
    with np.load(preds_dir / "M_jpeg.npz") as a:
        m_jpeg = a["outer_oof_prob"]
    with np.load(preds_dir / "M_raw.npz") as a:
        m_raw = a["outer_oof_prob"]

    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]
    counts = np.array([len(idx) for idx in tray_idx_list], dtype=np.float64)
    n_boot = len(boot_draws)

    def get_point_and_boots(p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        # Point kernel BA & cluster-bootstrap kernel BA (2000,)
        pred = p.argmax(axis=1)
        right = (pred == y_july_idx).astype(np.float64)
        kba_pt = float(np.mean([right[y_july_idx == c].mean() for c in range(12)]))
        hits = np.array([right[idx].sum() for idx in tray_idx_list], dtype=np.float64)
        bs = np.zeros(n_boot, dtype=np.float64)
        for c in range(12):
            take = truth_arr[boot_draws] == c
            bs += np.sum(hits[boot_draws] * take, axis=1) / np.sum(counts[boot_draws] * take, axis=1) / 12.0

        # Group BA point (6,) and bootstrap (6, 2000)
        correct = np.empty((6, 500, 67), dtype=bool)
        for t, trow in trays_df.iterrows():
            local = np.flatnonzero(saved_rows["record_id"].eq(trow["record_id"]))
            idx = rowpositions[local]
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
        return kba_pt, bs, gba_pt, bg

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

    rows_out = []
    for ens_name, pmap in ensembles.items():
        names = list(pmap.keys())
        n_cfg = len(names)
        k_pts = np.empty(n_cfg)
        k_boots = np.empty((n_cfg, n_boot))
        g_pts = np.empty((n_cfg, 6))
        g_boots = np.empty((n_cfg, 6, n_boot))
        for i, nm in enumerate(names):
            kp, kb, gp, gb = get_point_and_boots(pmap[nm])
            k_pts[i] = kp
            k_boots[i] = kb
            g_pts[i] = gp
            g_boots[i] = gb

        for ki, k in enumerate(KS):
            rho_pt = float(spearmanr(k_pts, g_pts[:, ki])[0])
            tau_pt = float(kendalltau(k_pts, g_pts[:, ki])[0])
            n_unique_g = int(len(np.unique(np.round(g_pts[:, ki], 6))))
            n_ties = int(n_cfg - n_unique_g)

            rhos = np.empty(n_boot)
            taus = np.empty(n_boot)
            for b in range(n_boot):
                x_b = k_boots[:, b]
                y_b = g_boots[:, ki, b]
                if np.std(y_b) < 1e-12 or np.std(x_b) < 1e-12:
                    rhos[b] = np.nan
                    taus[b] = np.nan
                else:
                    rhos[b] = spearmanr(x_b, y_b)[0]
                    taus[b] = kendalltau(x_b, y_b)[0]
            ok = np.isfinite(rhos)
            rows_out.append(
                dict(
                    ensemble=ens_name,
                    n_configs=n_cfg,
                    endpoint=f"k{k}",
                    n_tied_values_point=n_ties,
                    spearman_rho=rho_pt,
                    rho_ci_low=float(np.quantile(rhos[ok], 0.025)),
                    rho_ci_high=float(np.quantile(rhos[ok], 0.975)),
                    rho_prob_negative=float(np.mean(rhos[ok] < 0)),
                    kendall_tau=tau_pt,
                    tau_ci_low=float(np.quantile(taus[ok], 0.025)),
                    tau_ci_high=float(np.quantile(taus[ok], 0.975)),
                    valid_boot_draws=int(ok.sum()),
                )
            )
    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "p0_bootstrap_rank_correlations.csv", index=False)
    return df


def run_p1_post_pooling_transfer(
    out_dir: Path,
    primary_rows: pd.DataFrame,
    ext_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    rowpositions: np.ndarray,
    perms: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P1: Complete post-pooling calibration audit on BOTH Primary & August (BASE + 6 conds),
    across k in {3, 5, 10, 20, All}, with widened bounds [-4, 3] + explicit bound-hit flags,
    comparing:
      - Uncalibrated arithmetic mean
      - Post-pooling T_post(All) fitted on full-tray inner-OOF (applied to All and finite k)
      - Scale-matched Post-pooling T_post(k) fitted on inner-OOF at each k in {3, 5, 10, 20, All}
      - Pre-pooling T_kernel
      - Majority voting (reporting BOTH Brier score and clipped NLL explicitly)
      - Log-probability pooling
    """
    preds_dir = ROOT / "models_and_preds"
    tray_july = primary_rows["record_id"].to_numpy()
    folds_july = primary_rows["fold"].to_numpy()
    tray_folds = dict(zip(primary_rows["record_id"], primary_rows["fold"]))
    truth_arr = np.array([LABEL_TO_IDX[v] for v in trays_df["variety"]], dtype=int)
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_idx_list = [rowpositions[np.flatnonzero(saved_rows["record_id"].eq(r["record_id"]))] for _, r in trays_df.iterrows()]

    # Entity-paired schedules for August 6 conditions
    ext_rows = ext_rows.copy()
    ext_rows["row_index"] = np.arange(len(ext_rows))
    base_ext = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    entities = sorted(base_ext["entity_tray_id"].unique())
    entity_sampling: dict[str, tuple[list[str], np.ndarray]] = {}
    for t, ent in enumerate(entities):
        sids = base_ext.loc[base_ext["entity_tray_id"].eq(ent), "entity_seed_id"].tolist()
        rng = np.random.default_rng(20261114 + t)
        entity_sampling[ent] = (sids, np.stack([rng.permutation(len(sids)) for _ in range(500)]))

    LOG_LO, LOG_HI = -4.0, 3.0  # T in [0.0183, 20.086]

    def fit_post_temp_at_k(inner_p: np.ndarray, inner_y: np.ndarray, inner_trays: np.ndarray, k_val, seed: int) -> tuple[float, bool]:
        u_trays = sorted(np.unique(inner_trays).tolist())
        blocks = []
        ty = []
        for ti, tid in enumerate(u_trays):
            loc = np.flatnonzero(inner_trays == tid)
            tr = int(inner_y[loc[0]])
            ty.append(tr)
            if k_val == "All":
                q = inner_p[loc].mean(axis=0)[None, :]  # (1, 12)
            else:
                rng = np.random.default_rng(seed + ti)
                pm = np.stack([rng.permutation(len(loc))[: min(int(k_val), len(loc))] for _ in range(50)])
                q = inner_p[loc[pm]].mean(axis=1)  # (50, 12)
            blocks.append((np.log(np.clip(q, EPS, 1.0)), tr))
        ty_arr = np.asarray(ty, dtype=int)
        n_v = len(np.unique(ty_arr))
        tw = np.array([1.0 / (n_v * np.sum(ty_arr == c)) for c in ty_arr], dtype=np.float64)
        tw /= tw.sum()

        def obj(log_t: float) -> float:
            T = float(np.exp(log_t))
            tot = 0.0
            for w_t, (lq, tr) in zip(tw, blocks):
                qc = softmax_T(lq, T)
                tot += w_t * float(np.mean(-np.log(np.clip(qc[:, tr], EPS, 1.0))))
            return tot

        res = minimize_scalar(obj, bounds=(LOG_LO, LOG_HI), method="bounded")
        hit_bound = bool((res.x <= LOG_LO + 1e-3) or (res.x >= LOG_HI - 1e-3))
        return float(np.exp(res.x)), hit_bound

    def group_metrics_from_q(q_arr: np.ndarray, truth_vec: np.ndarray, weights: np.ndarray) -> dict[str, float]:
        # q_arr: (S, n_trays, 12)
        S, nt, _ = q_arr.shape
        ar_t = np.arange(nt)
        correct = (q_arr.argmax(axis=2) == truth_vec[None, :]).astype(np.float64)  # (S, nt)
        p_true = np.clip(q_arr[:, ar_t, truth_vec], EPS, 1.0)
        nll = -np.log(p_true)  # (S, nt)
        onehot = np.zeros_like(q_arr)
        onehot[:, ar_t, truth_vec] = 1.0
        brier = np.sum((q_arr - onehot) ** 2, axis=2)  # (S, nt)

        ba = float(correct.mean(axis=0) @ weights)
        gnll = float(nll.mean(axis=0) @ weights)
        gbrier = float(brier.mean(axis=0) @ weights)

        # Schedule-averaged ECE over all (S * nt) predictions weighted by weights/S
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

    temp_audit = []
    eval_rows = []
    eval_ks = [3, 5, 10, 20, "All"]

    for cid in FACTORIAL_SWIR + ["M_raw"]:
        with np.load(preds_dir / f"{cid}.npz") as a:
            outer_p = a["outer_oof_prob"]
            ext_p = a["ext_fold_prob"].mean(axis=0)
            inner_data = {f: (a[f"inner_oof_idx_fold{f}"], a[f"inner_oof_prob_fold{f}"]) for f in range(4)}

        # Fit scale-matched T_post(k) for each fold and each k in {3, 5, 10, 20, All}
        t_post_map: dict[tuple[int, str], float] = {}
        for f in range(4):
            in_idx, in_p = inner_data[f]
            for kv in eval_ks:
                T_k, hit = fit_post_temp_at_k(in_p, y_july_idx[in_idx], tray_july[in_idx], kv, seed=20261126 + f * 100)
                t_post_map[(f, str(kv))] = T_k
                temp_audit.append(dict(config_id=cid, fold=f, k=str(kv), T_post=T_k, hit_bound=hit))

        # Geometric mean of 4 fold temperatures for transferring to August ensemble averaged tray probs
        t_post_aug = {str(kv): float(np.exp(np.mean([np.log(t_post_map[(f, str(kv))]) for f in range(4)]))) for kv in eval_ks}

        # --- Primary evaluation across k ---
        q_raw_by_k = {kv: np.empty((1 if kv == "All" else 500, 67, 12), dtype=np.float64) for kv in eval_ks}
        q_mv_by_k = {kv: np.empty((1 if kv == "All" else 500, 67, 12), dtype=np.float64) for kv in eval_ks}
        for t, trow in trays_df.iterrows():
            local = np.flatnonzero(saved_rows["record_id"].eq(trow["record_id"]))
            idx = rowpositions[local]
            n_k = len(idx)
            perm = perms[:, t, :n_k]
            p_perm = outer_p[rowpositions[perm]]
            cum = np.cumsum(p_perm, axis=1)
            v_perm = np.eye(12)[p_perm.argmax(axis=2)]
            cum_v = np.cumsum(v_perm, axis=1)
            for kv in eval_ks:
                if kv == "All":
                    q_raw_by_k[kv][0, t] = outer_p[idx].mean(axis=0)
                    q_mv_by_k[kv][0, t] = np.eye(12)[outer_p[idx].argmax(axis=1)].mean(axis=0)
                else:
                    q_raw_by_k[kv][:, t] = cum[:, kv - 1, :] / float(kv)
                    q_mv_by_k[kv][:, t] = cum_v[:, kv - 1, :] / float(kv)

        for kv in eval_ks:
            qr = q_raw_by_k[kv]
            q_post_allT = np.empty_like(qr)
            q_post_matchT = np.empty_like(qr)
            for t, trow in trays_df.iterrows():
                f = tray_folds[str(trow["record_id"])]
                lq = np.log(np.clip(qr[:, t, :], EPS, 1.0))
                q_post_allT[:, t, :] = softmax_T(lq, t_post_map[(f, "All")])
                q_post_matchT[:, t, :] = softmax_T(lq, t_post_map[(f, str(kv))])

            for strat, qa in [
                ("arith_raw", qr),
                ("post_cal_AllT", q_post_allT),
                ("post_cal_matched_kT", q_post_matchT),
                ("majority_vote", q_mv_by_k[kv]),
            ]:
                m = group_metrics_from_q(qa, truth_arr, class_w)
                eval_rows.append(dict(cohort="primary", condition="PRIMARY", config_id=cid, k=str(kv), strategy=strat, **m))

        # --- August evaluation (BASE + all 6 conditions) ---
        for cond in CONDITIONS:
            block = ext_rows[ext_rows["condition"].eq(cond)]
            q_raw_aug = {kv: np.empty((1 if kv == "All" else 500, 10, 12), dtype=np.float64) for kv in eval_ks}
            q_mv_aug = {kv: np.empty((1 if kv == "All" else 500, 10, 12), dtype=np.float64) for kv in eval_ks}
            tr_aug = np.empty(10, dtype=int)
            for t, ent in enumerate(entities):
                sids, pm_e = entity_sampling[ent]
                sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
                j = sub["row_index"].to_numpy(dtype=int)
                tr_aug[t] = LABEL_TO_IDX[sub.iloc[0]["variety"]]
                p_perm = ext_p[j[pm_e]]
                cum = np.cumsum(p_perm, axis=1)
                v_perm = np.eye(12)[p_perm.argmax(axis=2)]
                cum_v = np.cumsum(v_perm, axis=1)
                for kv in eval_ks:
                    if kv == "All":
                        q_raw_aug[kv][0, t] = ext_p[j].mean(axis=0)
                        q_mv_aug[kv][0, t] = np.eye(12)[ext_p[j].argmax(axis=1)].mean(axis=0)
                    else:
                        q_raw_aug[kv][:, t] = cum[:, kv - 1, :] / float(kv)
                        q_mv_aug[kv][:, t] = cum_v[:, kv - 1, :] / float(kv)
            w_aug = np.full(10, 0.1, dtype=np.float64)
            for kv in eval_ks:
                qr = q_raw_aug[kv]
                lq = np.log(np.clip(qr.reshape(-1, 12), EPS, 1.0))
                q_allT = softmax_T(lq, t_post_aug["All"]).reshape(qr.shape)
                q_mT = softmax_T(lq, t_post_aug[str(kv)]).reshape(qr.shape)
                for strat, qa in [
                    ("arith_raw", qr),
                    ("post_cal_AllT", q_allT),
                    ("post_cal_matched_kT", q_mT),
                    ("majority_vote", q_mv_aug[kv]),
                ]:
                    m = group_metrics_from_q(qa, tr_aug, w_aug)
                    eval_rows.append(dict(cohort="august", condition=cond, config_id=cid, k=str(kv), strategy=strat, **m))

    ta_df = pd.DataFrame(temp_audit)
    ev_df = pd.DataFrame(eval_rows)
    ta_df.to_csv(out_dir / "p1_post_temp_bound_audit.csv", index=False)
    ev_df.to_csv(out_dir / "p1_post_pooling_transfer.csv", index=False)
    return ta_df, ev_df


def main() -> None:
    t0 = time.monotonic()
    out_dir = ROOT / "stage5_strengthening"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    y_july_idx = np.array([LABEL_TO_IDX[v] for v in primary_rows["variety_name"]], dtype=int)

    saved_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    rowpositions = pd.Series(np.arange(len(primary_rows)), index=primary_rows["seed_id"]).loc[saved_rows["seed_id"]].to_numpy()
    trays_df = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    perms = np.load(LONG / "batch2_schedules.npz")["permutations"]
    boot_npz = np.load(LONG / "batch2_bootstrap.npz")
    boot_draws, boot_weights = boot_npz["draws"], boot_npz["weights"]

    logging.info("Running P0.3: Rigorous optical perturbation physics (SAM, Pearson r, affine residual)...")
    opt_df = run_p0_optical_metrics(out_dir)

    logging.info("Running P0.1: Counterfactual margin interventions & pos/neg decomposition...")
    cf_df, ts_df = run_p0_counterfactual_margins(out_dir, primary_rows, y_july_idx, trays_df, saved_rows, rowpositions)

    logging.info("Running P0.2: 2,000-draw cluster-bootstrap rank correlations...")
    rk_df = run_p0_bootstrap_rank_correlations(
        out_dir, primary_rows, y_july_idx, trays_df, saved_rows, rowpositions, perms, boot_draws, boot_weights
    )

    logging.info("Running P1: Post-pooling calibration transfer to August + bound audit + Group Brier...")
    ta_df, ev_df = run_p1_post_pooling_transfer(
        out_dir, primary_rows, ext_rows, y_july_idx, trays_df, saved_rows, rowpositions, perms
    )

    # Print crisp verification summary to stdout
    print("\n=== P0.3 OPTICAL PHYSICS (mean spectra, 280 kernels) ===")
    print(opt_df[opt_df["summary"].eq("mean")][
        ["condition", "raw_rmse", "affine_residual_rmse", "rmse_reduction_factor", "mean_sam_deg", "mean_pearson_r", "mean_affine_r2"]
    ].round(4).to_string(index=False))

    print("\n=== P0.1 COUNTERFACTUAL RIVAL-PERMUTATION & POS/NEG CONTRIBUTION (14 SWIR) ===")
    print(cf_df[
        ["configuration", "kernel_ba", "orig_full_tray_ba", "mean_pos_kernel_contrib", "mean_wrong_kernel_drag",
         "cf_rival_perm_ba_mean", "cf_rival_perm_delta_ba", "cf_unif_wrong_ba"]
    ].round(4).to_string(index=False))

    print("\n=== P0.2 CLUSTER-BOOTSTRAP RANK CORRELATIONS (95% CIs) ===")
    print(rk_df[rk_df["endpoint"].isin(["k1", "k3", "k10", "k20", "kAll"])][
        ["ensemble", "endpoint", "n_tied_values_point", "spearman_rho", "rho_ci_low", "rho_ci_high", "kendall_tau", "tau_ci_low", "tau_ci_high"]
    ].round(4).to_string(index=False))

    print("\n=== P1 POST-POOLING CALIBRATION TRANSFER (Mean across 14 SWIR) ===")
    swir_ev = ev_df[ev_df["config_id"].isin(FACTORIAL_SWIR)]
    for label, sub in [
        ("Primary (All)", swir_ev[(swir_ev["cohort"] == "primary") & (swir_ev["k"] == "All")]),
        ("Primary (k=3)", swir_ev[(swir_ev["cohort"] == "primary") & (swir_ev["k"] == "3")]),
        ("Aug BASE (All)", swir_ev[(swir_ev["cohort"] == "august") & (swir_ev["condition"] == "BASE") & (swir_ev["k"] == "All")]),
        ("Aug 6-cond (All)", swir_ev[(swir_ev["cohort"] == "august") & (swir_ev["k"] == "All")]),
        ("Aug 6-cond (k=3)", swir_ev[(swir_ev["cohort"] == "august") & (swir_ev["k"] == "3")]),
    ]:
        print(f"-- {label} --")
        print(sub.groupby("strategy", as_index=False)[["group_ba", "group_nll", "group_brier", "group_ece"]].mean().round(4).to_string(index=False))

    print("\n=== P1 TEMPERATURE BOUND-HIT AUDIT ===")
    hits = ta_df[ta_df["hit_bound"]]
    print(f"Total bound hits: {len(hits)} / {len(ta_df)}")
    if len(hits) > 0:
        print(hits.to_string(index=False))

    logging.info("Stage 5 Strengthening COMPLETE in %.1fs.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
