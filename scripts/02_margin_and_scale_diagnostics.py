"""Stage 2: Multi-scale evaluation, cross-configuration rank decoupling,
and mathematical margin / finite-population variance / rival-concentration /
multimodal complementarity diagnostics for Path B+.

Evaluates all Stage 1 models + frozen references on identical 500 nested schedules
(primary 67 trays, August BASE 10 trays, and August 6-condition entity-paired).
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

BRIDGE_SWIR = [f"S_logistic_fixedC1_{rep}" for rep in ("median", "mean", "mean_std", "distribution")]
ALL_SWIR = FACTORIAL_SWIR + BRIDGE_SWIR


def weighted_prob_diagnostics(
    p: np.ndarray,
    y_idx: np.ndarray,
    tray_ids: np.ndarray,
    weighting: str = "variety_then_tray_equal",
) -> dict[str, float]:
    n = len(y_idx)
    if weighting == "kernel_equal":
        w = np.full(n, 1.0 / n, dtype=np.float64)
    elif weighting == "variety_then_tray_equal":
        # Equal weight per observed variety; within variety equal per tray; within tray equal per kernel
        df = pd.DataFrame({"y": y_idx, "t": tray_ids})
        n_vars = df["y"].nunique()
        trays_per_var = df.groupby("y")["t"].nunique().to_dict()
        seeds_per_tray = df.groupby("t").size().to_dict()
        w = np.array(
            [1.0 / (n_vars * trays_per_var[yy] * seeds_per_tray[tt]) for yy, tt in zip(y_idx, tray_ids)],
            dtype=np.float64,
        )
        w /= w.sum()
    else:
        raise ValueError(weighting)

    pred = p.argmax(axis=1)
    correct = (pred == y_idx).astype(np.float64)
    conf = p.max(axis=1)
    p_true = np.clip(p[np.arange(n), y_idx], EPS, 1.0)

    nll = float(np.sum(w * (-np.log(p_true))))
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y_idx] = 1.0
    brier = float(np.sum(w * np.sum((p - onehot) ** 2, axis=1)))
    norm_ent = float(np.sum(w * (-np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1) / np.log(12.0))))
    acc = float(np.sum(w * correct))
    mean_conf = float(np.sum(w * conf))

    # 10-bin equal-width top-label ECE
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf <= hi if b == 9 else conf < hi)
        wb = w[mask].sum()
        if wb > 0:
            acc_b = np.sum(w[mask] * correct[mask]) / wb
            conf_b = np.sum(w[mask] * conf[mask]) / wb
            ece += wb * abs(acc_b - conf_b)

    # Variety-equal kernel BA (canonical primary kernel discrimination metric)
    recalls = [float(np.mean(pred[y_idx == c] == c)) for c in np.unique(y_idx)]
    ba = float(np.mean(recalls))
    return dict(
        kernel_ba=ba,
        kernel_acc=acc,
        kernel_nll=nll,
        kernel_brier=brier,
        kernel_ece10=float(ece),
        kernel_norm_entropy=norm_ent,
        kernel_mean_conf=mean_conf,
    )


def eval_primary_schedules(
    p: np.ndarray,
    rowpositions: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    perms: np.ndarray,
    boot_draws: np.ndarray,
    boot_weights: np.ndarray,
    y_july_idx: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    """Evaluate 500 shared nested schedules on primary 67 trays.
    Returns:
      summary dict (group BA, group NLL, correct trays),
      boot_group (6, 2000),
      boot_seed (2000,),
      tray_margin_records (67 dicts with exact finite-pop variance check & rival concentration).
    """
    n_trays = len(trays_df)
    correct = np.empty((6, 500, n_trays), dtype=bool)
    group_nll = np.empty((6, 500, n_trays), dtype=np.float64)
    truth_arr = np.empty(n_trays, dtype=int)
    tray_indices: list[np.ndarray] = []
    tray_records: list[dict] = []

    for t, trow in trays_df.iterrows():
        local = np.flatnonzero(saved_rows["record_id"].eq(trow["record_id"]))
        idx = rowpositions[local]
        n_k = len(idx)
        truth = LABEL_TO_IDX[trow["variety"]]
        truth_arr[t] = truth
        tray_indices.append(idx)

        perm = perms[:, t, :n_k]
        p_perm = p[rowpositions[perm]]  # (500, n_k, 12)
        cum = np.cumsum(p_perm, axis=1)
        p_all = p[idx].mean(axis=0)

        for ki, k in enumerate(KS):
            if k == "All":
                correct[ki, :, t] = int(p_all.argmax()) == truth
                group_nll[ki, :, t] = -np.log(max(float(p_all[truth]), EPS))
            else:
                q = cum[:, k - 1, :] / float(k)
                correct[ki, :, t] = q.argmax(axis=1) == truth
                group_nll[ki, :, t] = -np.log(np.clip(q[:, truth], EPS, 1.0))

        # --- Mathematical margin & rival concentration diagnostics for tray t ---
        pt = p[idx]  # (n_k, 12)
        # Pairwise margins d_{i,c} = p_i(y) - p_i(c) for all 12 classes (truth entry = 0)
        d_mat = pt[:, [truth]] - pt  # (n_k, 12)
        mu_all = d_mat.mean(axis=0)  # (12,)
        var_all = d_mat.var(axis=0, ddof=1)  # (12,)
        rivals = np.arange(12) != truth
        rival_indices = np.flatnonzero(rivals)
        worst_rival = int(rival_indices[np.argmin(mu_all[rivals])])
        mu_star = float(mu_all[worst_rival])
        var_star = float(var_all[worst_rival])
        sd_star = float(np.sqrt(max(var_star, 0.0)))
        snr_star = float(mu_star / sd_star) if sd_star > 1e-12 else float(np.sign(mu_star) * 1e6)

        # Finite-population variance check on worst rival c* across k=1,3,5,10,20
        d_perm_star = p_perm[:, :, truth] - p_perm[:, :, worst_rival]  # (500, n_k)
        cum_d_star = np.cumsum(d_perm_star, axis=1)
        var_checks = {}
        for k in (1, 3, 5, 10, 20):
            if k <= n_k:
                emp_var = float(np.var(cum_d_star[:, k - 1] / float(k), ddof=1))
                theo_var = float((var_star / float(k)) * (1.0 - float(k) / float(n_k)))
                var_checks[f"var_emp_k{k}"] = emp_var
                var_checks[f"var_theo_k{k}"] = theo_var
            else:
                var_checks[f"var_emp_k{k}"] = np.nan
                var_checks[f"var_theo_k{k}"] = np.nan

        # Error rival concentration among misclassified kernels on tray t
        pred_k = pt.argmax(axis=1)
        wrong_mask = pred_k != truth
        n_wrong = int(wrong_mask.sum())
        if n_wrong > 0:
            wrong_preds = pred_k[wrong_mask]
            counts_w = np.bincount(wrong_preds, minlength=12).astype(np.float64)
            counts_w[truth] = 0.0
            shares = counts_w / n_wrong
            top_wrong_cls = int(np.argmax(counts_w))
            top1_share = float(shares[top_wrong_cls])
            rival_hhi = float(np.sum(shares ** 2))
            # Mean negative margin on misclassified kernels (p(y) - p(pred_k))
            mean_neg_margin_wrong = float(np.mean(pt[wrong_mask, truth] - pt[wrong_mask, wrong_preds]))
        else:
            top_wrong_cls = -1
            top1_share = 0.0
            rival_hhi = 0.0
            mean_neg_margin_wrong = 0.0

        rec = dict(
            record_id=str(trow["record_id"]),
            variety=str(trow["variety"]),
            n_kernels=int(n_k),
            kernel_correct_frac=float((pred_k == truth).mean()),
            full_tray_correct=bool(mu_star > 0.0),
            worst_rival=str(LABELS[worst_rival]),
            worst_rival_idx=worst_rival,
            mu_star=mu_star,
            sd_star=sd_star,
            var_star=var_star,
            snr_star=snr_star,
            n_wrong_kernels=n_wrong,
            wrong_kernel_frac=float(n_wrong / n_k),
            top_wrong_rival=str(LABELS[top_wrong_cls]) if top_wrong_cls >= 0 else "NONE",
            top1_rival_share=top1_share,
            rival_hhi=rival_hhi,
            mean_neg_margin_wrong=mean_neg_margin_wrong,
            full_tray_nll=float(group_nll[5, 0, t]),
        )
        for ki, k in enumerate(KS):
            rec[f"correct_frac_k{k}"] = float(correct[ki, :, t].mean())
        rec.update(var_checks)
        tray_records.append(rec)

    # Variety-balanced group BA & group NLL
    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    group_mean = correct.mean(axis=1)  # (6, 67)
    nll_mean = group_nll.mean(axis=1)  # (6, 67)
    group_ba = group_mean @ class_w  # (6,)
    group_nll_bal = nll_mean @ class_w  # (6,)
    boot_group = group_mean @ boot_weights.T  # (6, 2000)

    # Kernel cluster bootstrap
    pred_all = p.argmax(axis=1)
    right_all = (pred_all == y_july_idx).astype(np.float64)
    counts = np.array([len(idx) for idx in tray_indices], dtype=np.float64)
    hits = np.array([right_all[idx].sum() for idx in tray_indices], dtype=np.float64)
    boot_seed = np.zeros(len(boot_draws), dtype=np.float64)
    for c in range(12):
        take = truth_arr[boot_draws] == c
        boot_seed += np.sum(hits[boot_draws] * take, axis=1) / np.sum(counts[boot_draws] * take, axis=1) / 12.0

    # Failure regime decomposition (variance-driven vs systematic negative-margin errors)
    sys_wrong_mask = np.array([not r["full_tray_correct"] for r in tray_records], dtype=bool)
    summary = dict(correct_trays=int(np.sum(~sys_wrong_mask)), total_trays=int(n_trays))
    for ki, k in enumerate(KS):
        summary[f"group_ba_k{k}"] = float(group_ba[ki])
        summary[f"group_nll_k{k}"] = float(group_nll_bal[ki])
        err_rate_tray = 1.0 - group_mean[ki]  # (67,)
        total_err = float(err_rate_tray @ class_w)
        var_err = float((err_rate_tray * (~sys_wrong_mask)) @ class_w)
        sys_err = float((err_rate_tray * sys_wrong_mask) @ class_w)
        summary[f"err_total_k{k}"] = total_err
        summary[f"err_variance_driven_k{k}"] = var_err
        summary[f"err_systematic_k{k}"] = sys_err

    return summary, boot_group, boot_seed, tray_records


def eval_external_conditions(
    p_ext: np.ndarray,
    ext_rows: pd.DataFrame,
    entity_sampling: dict[str, tuple[list[str], np.ndarray]],
) -> tuple[list[dict], list[dict]]:
    """Evaluate all 6 August conditions using identical entity-paired 500 schedules."""
    cond_rows = []
    tray_rows = []
    entities = sorted(entity_sampling.keys())
    y_all_idx = np.array([LABEL_TO_IDX[v] for v in ext_rows["variety"]], dtype=int)

    for cond in CONDITIONS:
        block = ext_rows[ext_rows["condition"].eq(cond)]
        c_idx = block.index.to_numpy()
        p_cond = p_ext[c_idx]
        y_cond = y_all_idx[c_idx]
        t_cond = block["entity_tray_id"].to_numpy()
        diag = weighted_prob_diagnostics(p_cond, y_cond, t_cond, "variety_then_tray_equal")

        correct = np.empty((6, 500, 10), dtype=bool)
        g_nll = np.empty((6, 500, 10), dtype=np.float64)

        for t, ent in enumerate(entities):
            sids, perm = entity_sampling[ent]
            sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
            j = sub["row_index"].to_numpy(dtype=int)
            truth = LABEL_TO_IDX[sub.iloc[0]["variety"]]
            pt = p_ext[j]
            p_perm = p_ext[j[perm]]
            cum = np.cumsum(p_perm, axis=1)
            p_all = pt.mean(axis=0)
            for ki, k in enumerate(KS):
                if k == "All":
                    correct[ki, :, t] = int(p_all.argmax()) == truth
                    g_nll[ki, :, t] = -np.log(max(float(p_all[truth]), EPS))
                else:
                    q = cum[:, k - 1, :] / float(k)
                    correct[ki, :, t] = q.argmax(axis=1) == truth
                    g_nll[ki, :, t] = -np.log(np.clip(q[:, truth], EPS, 1.0))

            d_mat = pt[:, [truth]] - pt
            mu_all = d_mat.mean(axis=0)
            var_all = d_mat.var(axis=0, ddof=1)
            rivals = np.arange(12) != truth
            r_idx = np.flatnonzero(rivals)
            wr = int(r_idx[np.argmin(mu_all[rivals])])
            mu_star = float(mu_all[wr])
            sd_star = float(np.sqrt(max(float(var_all[wr]), 0.0)))
            tray_rows.append(
                dict(
                    condition=cond,
                    entity_tray_id=ent,
                    variety=str(sub.iloc[0]["variety"]),
                    n_kernels=int(len(j)),
                    full_tray_correct=bool(mu_star > 0.0),
                    worst_rival=str(LABELS[wr]),
                    mu_star=mu_star,
                    sd_star=sd_star,
                    correct_frac_k3=float(correct[1, :, t].mean()),
                    correct_frac_k20=float(correct[4, :, t].mean()),
                    full_tray_nll=float(g_nll[5, 0, t]),
                )
            )

        rec = dict(condition=cond, correct_trays=int(correct[5, 0, :].sum()), total_trays=10)
        rec.update(diag)
        for ki, k in enumerate(KS):
            rec[f"group_ba_k{k}"] = float(correct[ki].mean())
            rec[f"group_nll_k{k}"] = float(g_nll[ki].mean())
        cond_rows.append(rec)

    return cond_rows, tray_rows


def compute_multimodal_margin_complementarity(
    p_md: np.ndarray,
    p_s: np.ndarray,
    rows_df: pd.DataFrame,
    tray_col: str,
    var_col: str,
    scheme_label: str,
) -> list[dict]:
    """Exact additive margin decomposition for MSD = (2/3)MD + (1/3)S across all trays."""
    p_msd = (2.0 / 3.0) * p_md + (1.0 / 3.0) * p_s
    out = []
    for tray_id, grp in rows_df.groupby(tray_col, sort=True):
        idx = grp.index.to_numpy()
        truth_str = str(grp[var_col].iloc[0])
        ti = LABEL_TO_IDX[truth_str]
        rivals = np.arange(12) != ti
        r_idx = np.flatnonzero(rivals)

        mu_md = p_md[idx, ti].mean() - p_md[idx].mean(axis=0)
        mu_s = p_s[idx, ti].mean() - p_s[idx].mean(axis=0)
        mu_msd = p_msd[idx, ti].mean() - p_msd[idx].mean(axis=0)
        delta_all = (1.0 / 3.0) * (mu_s - mu_md)
        # Verify exact algebraic identity
        assert np.max(np.abs(mu_msd - (mu_md + delta_all))) < 1e-10

        wr_md = int(r_idx[np.argmin(mu_md[rivals])])
        wr_s = int(r_idx[np.argmin(mu_s[rivals])])
        wr_msd = int(r_idx[np.argmin(mu_msd[rivals])])

        md_ok = bool(mu_md[wr_md] > 0.0)
        s_ok = bool(mu_s[wr_s] > 0.0)
        msd_ok = bool(mu_msd[wr_msd] > 0.0)
        if md_ok and msd_ok:
            outcome = "both_correct"
        elif (not md_ok) and msd_ok:
            outcome = "rescued"
        elif md_ok and (not msd_ok):
            outcome = "harmed"
        else:
            outcome = "both_wrong"

        out.append(
            dict(
                scheme=scheme_label,
                tray_id=str(tray_id),
                variety=truth_str,
                n_kernels=int(len(idx)),
                md_correct=md_ok,
                s_correct=s_ok,
                msd_correct=msd_ok,
                outcome=outcome,
                rival_md=str(LABELS[wr_md]),
                rival_s=str(LABELS[wr_s]),
                rival_msd=str(LABELS[wr_msd]),
                rival_switched=bool(wr_md != wr_msd),
                mu_star_md=float(mu_md[wr_md]),
                mu_star_s=float(mu_s[wr_s]),
                mu_star_msd=float(mu_msd[wr_msd]),
                net_mu_star_shift=float(mu_msd[wr_msd] - mu_md[wr_md]),
                delta_d_on_md_rival=float(delta_all[wr_md]),
                delta_d_on_msd_rival=float(delta_all[wr_msd]),
                s_margin_on_md_rival=float(mu_s[wr_md]),
            )
        )
    return out


def main() -> None:
    t0 = time.monotonic()
    out_dir = ROOT / "stage2_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "stage2.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    ext_rows["row_index"] = np.arange(len(ext_rows))
    y_july_idx = np.array([LABEL_TO_IDX[v] for v in primary_rows["variety_name"]], dtype=int)
    tray_july = primary_rows["record_id"].to_numpy()

    saved_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    rowpositions = pd.Series(np.arange(len(primary_rows)), index=primary_rows["seed_id"]).loc[saved_rows["seed_id"]].to_numpy()
    trays_df = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    perms = np.load(LONG / "batch2_schedules.npz")["permutations"]
    boot_npz = np.load(LONG / "batch2_bootstrap.npz")
    boot_draws = boot_npz["draws"]
    boot_weights = boot_npz["weights"]

    # Entity-paired 500 schedules across 6 August conditions
    base_ext = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    entities = sorted(base_ext["entity_tray_id"].unique())
    entity_sampling: dict[str, tuple[list[str], np.ndarray]] = {}
    for t, ent in enumerate(entities):
        sids = base_ext.loc[base_ext["entity_tray_id"].eq(ent), "entity_seed_id"].tolist()
        rng = np.random.default_rng(20261114 + t)
        entity_sampling[ent] = (sids, np.stack([rng.permutation(len(sids)) for _ in range(500)]))

    # Load RGB & frozen references + Stage 1 SWIR/M predictions
    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rf:
        d_july = rf["d_prob_july"]
        d_aug = rf["d_prob_aug"]
        frozen_m_july = rf["frozen_m_july"]
        frozen_s_july = rf["frozen_s_july"]
        frozen_m_aug = rf["frozen_m_aug"]
        frozen_s_aug = rf["frozen_s_aug"]

    preds_dir = ROOT / "models_and_preds"
    def load_cfg(cid: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(preds_dir / f"{cid}.npz") as a:
            return a["outer_oof_prob"], a["ext_fold_prob"].mean(axis=0)

    m_jpeg_july, m_jpeg_aug = load_cfg("M_jpeg")
    m_raw_july, m_raw_aug = load_cfg("M_raw")

    # Assemble all primary & external probability dictionaries
    eval_configs: dict[str, tuple[np.ndarray, np.ndarray, str, str, str]] = {
        # name: (p_july, p_aug, family, classifier, representation)
        "RGB_M_jpeg": (m_jpeg_july, m_jpeg_aug, "RGB", "logistic_C1", "JPEG"),
        "RGB_M_raw": (m_raw_july, m_raw_aug, "RGB", "logistic_C1", "RAW"),
        "RGB_D_phys": (d_july, d_aug, "RGB", "efficientnet_b0", "JPEG"),
        "RGB_MD_jpeg": ((m_jpeg_july + d_july) / 2.0, (m_jpeg_aug + d_aug) / 2.0, "RGB", "MD", "JPEG"),
        "RGB_MD_raw": ((m_raw_july + d_july) / 2.0, (m_raw_aug + d_aug) / 2.0, "RGB", "MD", "RAW_M+JPEG_D"),
        "REF_frozen_S": (frozen_s_july, frozen_s_aug, "SWIR_ref", "frozen_logistic", "median"),
        "REF_frozen_MSD": ((frozen_m_july + d_july + frozen_s_july) / 3.0, (frozen_m_aug + d_aug + frozen_s_aug) / 3.0, "MSD_ref", "frozen_logistic", "median"),
    }

    swir_cache: dict[str, tuple[np.ndarray, np.ndarray, str, str]] = {}
    for cid in ALL_SWIR:
        pj, pa = load_cfg(cid)
        parts = cid.split("_")
        # Parse clf and rep cleanly
        if cid.startswith("S_logistic_fixedC1_"):
            clf, rep = "logistic_fixedC1", cid[len("S_logistic_fixedC1_"):]
        elif cid.startswith("S_logistic_"):
            clf, rep = "logistic", cid[len("S_logistic_"):]
        elif cid.startswith("S_rbf_svm_"):
            clf, rep = "rbf_svm", cid[len("S_rbf_svm_"):]
        elif cid.startswith("S_extra_trees_"):
            clf, rep = "extra_trees", cid[len("S_extra_trees_"):]
        elif cid.startswith("S_cnn1d_"):
            clf, rep = "cnn1d", cid[len("S_cnn1d_"):]
        else:
            raise ValueError(cid)
        swir_cache[cid] = (pj, pa, clf, rep)
        fam_prefix = "SWIR_factorial" if cid in FACTORIAL_SWIR else "SWIR_bridge"
        eval_configs[cid] = (pj, pa, fam_prefix, clf, rep)
        eval_configs[f"MS_jpeg__{cid}"] = ((m_jpeg_july + pj) / 2.0, (m_jpeg_aug + pa) / 2.0, "MS_jpeg", clf, rep)
        eval_configs[f"MS_raw__{cid}"] = ((m_raw_july + pj) / 2.0, (m_raw_aug + pa) / 2.0, "MS_raw", clf, rep)
        eval_configs[f"SD__{cid}"] = ((d_july + pj) / 2.0, (d_aug + pa) / 2.0, "SD", clf, rep)
        eval_configs[f"MSD_jpeg__{cid}"] = ((m_jpeg_july + d_july + pj) / 3.0, (m_jpeg_aug + d_aug + pa) / 3.0, "MSD_jpeg", clf, rep)
        eval_configs[f"MSD_raw__{cid}"] = ((m_raw_july + d_july + pj) / 3.0, (m_raw_aug + d_aug + pa) / 3.0, "MSD_raw", clf, rep)

    primary_metrics = []
    primary_tray_margins = []
    ext_cond_metrics = []
    ext_tray_margins = []
    boot_group_map: dict[str, np.ndarray] = {}
    boot_seed_map: dict[str, np.ndarray] = {}

    for name, (pj, pa, family, clf, rep) in eval_configs.items():
        diag_vt = weighted_prob_diagnostics(pj, y_july_idx, tray_july, "variety_then_tray_equal")
        diag_ke = weighted_prob_diagnostics(pj, y_july_idx, tray_july, "kernel_equal")
        psumm, bg, bs, trecs = eval_primary_schedules(
            pj, rowpositions, trays_df, saved_rows, perms, boot_draws, boot_weights, y_july_idx
        )
        boot_group_map[name] = bg
        boot_seed_map[name] = bs

        row = dict(configuration=name, family=family, classifier=clf, representation=rep)
        row.update(diag_vt)
        row["kernel_acc_equal"] = diag_ke["kernel_acc"]
        row["kernel_ece10_equal"] = diag_ke["kernel_ece10"]
        row.update(psumm)

        # Summarize tray-level margin & finite-population variance agreement
        tdf = pd.DataFrame(trecs)
        row["mean_mu_star"] = float(tdf["mu_star"].mean())
        row["median_mu_star"] = float(tdf["mu_star"].median())
        row["mean_sd_star"] = float(tdf["sd_star"].mean())
        row["mean_snr_star"] = float(np.median(tdf["snr_star"]))
        row["mean_top1_rival_share"] = float(tdf.loc[tdf["n_wrong_kernels"] > 0, "top1_rival_share"].mean())
        row["mean_rival_hhi"] = float(tdf.loc[tdf["n_wrong_kernels"] > 0, "rival_hhi"].mean())
        row["mean_neg_margin_wrong"] = float(tdf.loc[tdf["n_wrong_kernels"] > 0, "mean_neg_margin_wrong"].mean())
        # Variance correlation across all trays x k=1,3,5,10,20
        ve = np.concatenate([tdf[f"var_emp_k{k}"].to_numpy() for k in (1, 3, 5, 10, 20)])
        vt = np.concatenate([tdf[f"var_theo_k{k}"].to_numpy() for k in (1, 3, 5, 10, 20)])
        ok = np.isfinite(ve) & np.isfinite(vt) & (vt > 0)
        row["finite_pop_var_r2"] = float(np.corrcoef(ve[ok], vt[ok])[0, 1] ** 2)
        row["finite_pop_var_ratio_median"] = float(np.median(ve[ok] / vt[ok]))
        primary_metrics.append(row)

        for r in trecs:
            r["configuration"] = name
            r["family"] = family
            r["classifier"] = clf
            r["representation"] = rep
        primary_tray_margins.extend(trecs)

        # External 6-condition evaluation
        ec_rows, et_rows = eval_external_conditions(pa, ext_rows, entity_sampling)
        for r in ec_rows:
            r.update(configuration=name, family=family, classifier=clf, representation=rep)
        for r in et_rows:
            r.update(configuration=name, family=family, classifier=clf, representation=rep)
        ext_cond_metrics.extend(ec_rows)
        ext_tray_margins.extend(et_rows)

    pm_df = pd.DataFrame(primary_metrics)
    pm_df.to_csv(out_dir / "primary_metrics.csv", index=False)
    pd.DataFrame(primary_tray_margins).to_csv(out_dir / "primary_tray_margins.csv", index=False)
    ec_df = pd.DataFrame(ext_cond_metrics)
    ec_df.to_csv(out_dir / "external_6cond_metrics.csv", index=False)
    pd.DataFrame(ext_tray_margins).to_csv(out_dir / "external_tray_margins.csv", index=False)

    # --- Rank correlations between kernel_ba and group_ba_k{3,5,10,20,All} ---
    rank_rows = []
    subsets = [
        ("SWIR_factorial_all14", pm_df[pm_df["family"].eq("SWIR_factorial")]),
        ("MSD_jpeg_factorial14", pm_df[pm_df["family"].eq("MSD_jpeg") & pm_df["configuration"].str.replace("MSD_jpeg__", "", regex=False).isin(FACTORIAL_SWIR)]),
        ("MSD_raw_factorial14", pm_df[pm_df["family"].eq("MSD_raw") & pm_df["configuration"].str.replace("MSD_raw__", "", regex=False).isin(FACTORIAL_SWIR)]),
    ]
    for rep in ("median", "mean", "mean_std", "distribution"):
        subsets.append((f"SWIR_fixed_rep_{rep}", pm_df[pm_df["family"].eq("SWIR_factorial") & pm_df["representation"].eq(rep)]))
    for clf in ("logistic", "rbf_svm", "extra_trees"):
        subsets.append((f"SWIR_fixed_clf_{clf}", pm_df[pm_df["family"].eq("SWIR_factorial") & pm_df["classifier"].eq(clf)]))

    for label, sub in subsets:
        if len(sub) < 3:
            continue
        x = sub["kernel_ba"].to_numpy()
        for k in (1, 3, 5, 10, 20, "All"):
            yk = sub[f"group_ba_k{k}"].to_numpy()
            rho, p_rho = spearmanr(x, yk)
            tau, p_tau = kendalltau(x, yk)
            rank_rows.append(
                dict(
                    subset=label,
                    n_configs=int(len(sub)),
                    endpoint=f"k{k}",
                    spearman_rho=float(rho) if np.isfinite(rho) else np.nan,
                    spearman_p=float(p_rho) if np.isfinite(p_rho) else np.nan,
                    kendall_tau=float(tau) if np.isfinite(tau) else np.nan,
                    kendall_p=float(p_tau) if np.isfinite(p_tau) else np.nan,
                )
            )
    pd.DataFrame(rank_rows).to_csv(out_dir / "rank_correlations.csv", index=False)

    # --- Paired cluster-bootstrap contrasts (key comparisons) ---
    contrast_pairs = []
    # Within-representation classifier contrasts vs logistic
    for rep in ("median", "mean", "mean_std", "distribution"):
        for clf in ("rbf_svm", "extra_trees", "cnn1d"):
            a = f"S_{clf}_{rep}"
            b = f"S_logistic_{rep}"
            if a in boot_group_map and b in boot_group_map:
                contrast_pairs.append((a, b, f"clf_vs_logistic_within_{rep}"))
    # Within-classifier representation contrasts vs median
    for clf in ("logistic", "logistic_fixedC1", "rbf_svm", "extra_trees", "cnn1d"):
        for rep in ("mean", "mean_std", "distribution"):
            a = f"S_{clf}_{rep}"
            b = f"S_{clf}_median"
            if a in boot_group_map and b in boot_group_map:
                contrast_pairs.append((a, b, f"rep_vs_median_within_{clf}"))
    # Multimodal SWIR gain MSD - MD under JPEG-M and RAW-M
    for cid in FACTORIAL_SWIR + BRIDGE_SWIR:
        contrast_pairs.append((f"MSD_jpeg__{cid}", "RGB_MD_jpeg", "MSD_minus_MD_jpeg"))
        contrast_pairs.append((f"MSD_raw__{cid}", "RGB_MD_raw", "MSD_minus_MD_raw"))
        contrast_pairs.append((f"MS_jpeg__{cid}", "RGB_M_jpeg", "MS_minus_M_jpeg"))
        contrast_pairs.append((f"MS_raw__{cid}", "RGB_M_raw", "MS_minus_M_raw"))
        contrast_pairs.append((f"SD__{cid}", "RGB_D_phys", "SD_minus_D"))
    # RAW-M vs JPEG-M
    contrast_pairs.append(("RGB_M_raw", "RGB_M_jpeg", "RAW_vs_JPEG_M"))
    contrast_pairs.append(("RGB_MD_raw", "RGB_MD_jpeg", "RAW_vs_JPEG_MD"))

    pm_idx = pm_df.set_index("configuration")
    c_rows = []
    for a, b, kind in contrast_pairs:
        d_seed = boot_seed_map[a] - boot_seed_map[b]
        c_rows.append(
            dict(
                config_a=a,
                config_b=b,
                contrast_kind=kind,
                endpoint="kernel",
                delta_ba=float(pm_idx.loc[a, "kernel_ba"] - pm_idx.loc[b, "kernel_ba"]),
                ci_low=float(np.quantile(d_seed, 0.025)),
                ci_high=float(np.quantile(d_seed, 0.975)),
            )
        )
        for ki, k in enumerate(KS):
            d_g = boot_group_map[a][ki] - boot_group_map[b][ki]
            c_rows.append(
                dict(
                    config_a=a,
                    config_b=b,
                    contrast_kind=kind,
                    endpoint=f"k{k}",
                    delta_ba=float(pm_idx.loc[a, f"group_ba_k{k}"] - pm_idx.loc[b, f"group_ba_k{k}"]),
                    ci_low=float(np.quantile(d_g, 0.025)),
                    ci_high=float(np.quantile(d_g, 0.975)),
                )
            )
    pd.DataFrame(c_rows).to_csv(out_dir / "primary_paired_contrasts.csv", index=False)

    # --- Exact additive multimodal margin complementarity (Primary + August BASE) ---
    comp_primary = []
    comp_aug_base = []
    aug_base_rows = ext_rows[ext_rows["condition"].eq("BASE")].copy().reset_index(drop=True)
    aug_base_idx = ext_rows.index[ext_rows["condition"].eq("BASE")].to_numpy()
    md_jpeg_j = (m_jpeg_july + d_july) / 2.0
    md_raw_j = (m_raw_july + d_july) / 2.0
    md_jpeg_a = ((m_jpeg_aug + d_aug) / 2.0)[aug_base_idx]
    md_raw_a = ((m_raw_aug + d_aug) / 2.0)[aug_base_idx]

    for cid in FACTORIAL_SWIR + BRIDGE_SWIR:
        pj, pa, _, _ = swir_cache[cid]
        comp_primary.extend(
            compute_multimodal_margin_complementarity(md_jpeg_j, pj, primary_rows, "record_id", "variety_name", f"MSD_jpeg__{cid}")
        )
        comp_primary.extend(
            compute_multimodal_margin_complementarity(md_raw_j, pj, primary_rows, "record_id", "variety_name", f"MSD_raw__{cid}")
        )
        comp_aug_base.extend(
            compute_multimodal_margin_complementarity(md_jpeg_a, pa[aug_base_idx], aug_base_rows, "entity_tray_id", "variety", f"MSD_jpeg__{cid}")
        )
        comp_aug_base.extend(
            compute_multimodal_margin_complementarity(md_raw_a, pa[aug_base_idx], aug_base_rows, "entity_tray_id", "variety", f"MSD_raw__{cid}")
        )

    pd.DataFrame(comp_primary).to_csv(out_dir / "multimodal_margin_complementarity_primary.csv", index=False)
    pd.DataFrame(comp_aug_base).to_csv(out_dir / "multimodal_margin_complementarity_aug_base.csv", index=False)

    # Summary table of complementarity counts per scheme
    def summarize_comp(df: pd.DataFrame) -> pd.DataFrame:
        rows_s = []
        for sch, g in df.groupby("scheme", sort=True):
            rows_s.append(
                dict(
                    scheme=sch,
                    n_trays=int(len(g)),
                    rescued=int((g["outcome"] == "rescued").sum()),
                    harmed=int((g["outcome"] == "harmed").sum()),
                    both_correct=int((g["outcome"] == "both_correct").sum()),
                    both_wrong=int((g["outcome"] == "both_wrong").sum()),
                    rival_switched_trays=int(g["rival_switched"].sum()),
                    mean_net_mu_star_shift=float(g["net_mu_star_shift"].mean()),
                    mean_delta_d_on_md_rival=float(g["delta_d_on_md_rival"].mean()),
                    frac_trays_positive_delta=float((g["delta_d_on_md_rival"] > 0).mean()),
                )
            )
        return pd.DataFrame(rows_s)

    summarize_comp(pd.DataFrame(comp_primary)).to_csv(out_dir / "complementarity_summary_primary.csv", index=False)
    summarize_comp(pd.DataFrame(comp_aug_base)).to_csv(out_dir / "complementarity_summary_aug_base.csv", index=False)

    logging.info("Stage 2 COMPLETE in %.1fs. Evaluated %d configurations.", time.monotonic() - t0, len(eval_configs))


if __name__ == "__main__":
    main()
