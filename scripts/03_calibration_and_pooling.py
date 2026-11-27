"""Stage 3: Strictly leak-free nested tray-grouped probability calibration
(standard Kernel-NLL Temperature Scaling vs conditional Group-Aware Temperature Scaling)
and multi-kernel pooling strategy comparisons (Arithmetic, Majority Voting,
Log-Probability Pooling, Pre-pooling Calibrated, and Post-pooling Reference).

All temperature parameters are fitted ONLY on independent inner-OOF predictions
inside each outer training fold, never touching outer test trays or August data.
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

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
LONG = PROJECT / "analysis" / "long_term_validation" / "data"
sys.path.insert(0, str(PROJECT))
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE

LABELS = np.asarray(FROZEN_LABEL_SPACE)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
KS = [1, 3, 5, 10, 20, "All"]
GROUP_FIT_KS = [3, 5, 10, "All"]
CONDITIONS = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
EPS = 1e-12
INNER_POOL_SCHEDULES = 50

FACTORIAL_SWIR = [
    f"S_{clf}_{rep}"
    for clf in ("logistic", "rbf_svm", "extra_trees")
    for rep in ("median", "mean", "mean_std", "distribution")
] + ["S_cnn1d_median", "S_cnn1d_mean"]
BRIDGE_SWIR = [f"S_logistic_fixedC1_{rep}" for rep in ("median", "mean", "mean_std", "distribution")]
ALL_SWIR = FACTORIAL_SWIR + BRIDGE_SWIR


def softmax_T(scores: np.ndarray, T: float) -> np.ndarray:
    z = scores / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_vt_weights(y_idx: np.ndarray, tray_ids: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"y": y_idx, "t": tray_ids})
    n_vars = df["y"].nunique()
    tpv = df.groupby("y")["t"].nunique().to_dict()
    spt = df.groupby("t").size().to_dict()
    w = np.array([1.0 / (n_vars * tpv[yy] * spt[tt]) for yy, tt in zip(y_idx, tray_ids)], dtype=np.float64)
    return w / w.sum()


def fit_temperature_kernel(scores: np.ndarray, y_idx: np.ndarray, tray_ids: np.ndarray) -> float:
    w = build_vt_weights(y_idx, tray_ids)
    n = len(y_idx)
    ar = np.arange(n)

    def obj(log_t: float) -> float:
        T = float(np.exp(log_t))
        p = softmax_T(scores, T)
        return float(np.sum(w * (-np.log(np.clip(p[ar, y_idx], EPS, 1.0)))))

    res = minimize_scalar(obj, bounds=(-2.5, 2.5), method="bounded")
    return float(np.exp(res.x))


def fit_temperature_group(
    scores: np.ndarray,
    y_idx: np.ndarray,
    tray_ids: np.ndarray,
    seed: int,
) -> float:
    """Group-aware temperature scaling: minimize equal-weighted multi-k group NLL
    across k in {3, 5, 10, All} strictly on inner-OOF trays.
    """
    unique_trays = sorted(np.unique(tray_ids).tolist())
    # Prepare deterministic permutations per inner tray
    tray_blocks = []
    tray_truths = []
    for t_i, tid in enumerate(unique_trays):
        loc = np.flatnonzero(tray_ids == tid)
        truth = int(y_idx[loc[0]])
        assert np.all(y_idx[loc] == truth)
        rng = np.random.default_rng(seed + t_i)
        perms = np.stack([rng.permutation(len(loc)) for _ in range(INNER_POOL_SCHEDULES)])
        tray_blocks.append((loc, perms, truth))
        tray_truths.append(truth)

    truth_arr = np.asarray(tray_truths, dtype=int)
    n_vars = len(np.unique(truth_arr))
    tray_w = np.array([1.0 / (n_vars * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    tray_w /= tray_w.sum()

    def obj(log_t: float) -> float:
        T = float(np.exp(log_t))
        p = softmax_T(scores, T)
        total_loss = 0.0
        for tw, (loc, perms, truth) in zip(tray_w, tray_blocks):
            pt = p[loc]
            # All
            q_all = float(pt[:, truth].mean())
            loss_t = -np.log(max(q_all, EPS))
            # Finite k in {3, 5, 10}
            p_true_perm = pt[perms, truth]  # (S, n_k)
            cum_true = np.cumsum(p_true_perm, axis=1)
            for k in (3, 5, 10):
                kk = min(k, len(loc))
                qk = cum_true[:, kk - 1] / float(kk)
                loss_t += float(np.mean(-np.log(np.clip(qk, EPS, 1.0))))
            total_loss += tw * (loss_t / 4.0)
        return float(total_loss)

    res = minimize_scalar(obj, bounds=(-2.5, 2.5), method="bounded")
    return float(np.exp(res.x))


def fit_temperature_post_tray(probs: np.ndarray, y_idx: np.ndarray, tray_ids: np.ndarray) -> float:
    """Post-pooling calibration reference fitted on inner-OOF full-tray averaged probabilities."""
    unique_trays = sorted(np.unique(tray_ids).tolist())
    tray_scores = []
    tray_y = []
    for tid in unique_trays:
        loc = np.flatnonzero(tray_ids == tid)
        q = probs[loc].mean(axis=0)
        tray_scores.append(np.log(np.clip(q, EPS, 1.0)))
        tray_y.append(int(y_idx[loc[0]]))
    ts = np.asarray(tray_scores, dtype=np.float64)
    ty = np.asarray(tray_y, dtype=int)
    n_vars = len(np.unique(ty))
    tw = np.array([1.0 / (n_vars * np.sum(ty == c)) for c in ty], dtype=np.float64)
    tw /= tw.sum()
    ar = np.arange(len(ty))

    def obj(log_t: float) -> float:
        T = float(np.exp(log_t))
        q_cal = softmax_T(ts, T)
        return float(np.sum(tw * (-np.log(np.clip(q_cal[ar, ty], EPS, 1.0)))))

    res = minimize_scalar(obj, bounds=(-2.5, 2.5), method="bounded")
    return float(np.exp(res.x))


def kernel_quality_metrics(p: np.ndarray, y_idx: np.ndarray, tray_ids: np.ndarray) -> dict[str, float]:
    w = build_vt_weights(y_idx, tray_ids)
    n = len(y_idx)
    ar = np.arange(n)
    pred = p.argmax(axis=1)
    correct = (pred == y_idx).astype(np.float64)
    conf = p.max(axis=1)
    nll = float(np.sum(w * (-np.log(np.clip(p[ar, y_idx], EPS, 1.0)))))
    onehot = np.zeros_like(p)
    onehot[ar, y_idx] = 1.0
    brier = float(np.sum(w * np.sum((p - onehot) ** 2, axis=1)))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf <= hi if b == 9 else conf < hi)
        wb = w[mask].sum()
        if wb > 0:
            ece += wb * abs(np.sum(w[mask] * correct[mask]) / wb - np.sum(w[mask] * conf[mask]) / wb)
    ba = float(np.mean([np.mean(pred[y_idx == c] == c) for c in np.unique(y_idx)]))
    return dict(kernel_ba=ba, kernel_nll=nll, kernel_brier=brier, kernel_ece10=float(ece))


def eval_pooling_primary(
    p: np.ndarray,
    strategy: str,
    rowpositions: np.ndarray,
    trays_df: pd.DataFrame,
    saved_rows: pd.DataFrame,
    perms: np.ndarray,
    boot_weights: np.ndarray,
    post_temps: dict[int, float] | None = None,
    tray_folds: dict[str, int] | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate a pooling strategy across 500 shared nested schedules on 67 primary trays."""
    n_trays = len(trays_df)
    correct = np.empty((6, 500, n_trays), dtype=bool)
    group_nll = np.empty((6, 500, n_trays), dtype=np.float64)
    truth_arr = np.empty(n_trays, dtype=int)
    full_tray_q = np.empty((n_trays, 12), dtype=np.float64)

    for t, trow in trays_df.iterrows():
        rec_id = str(trow["record_id"])
        local = np.flatnonzero(saved_rows["record_id"].eq(rec_id))
        idx = rowpositions[local]
        n_k = len(idx)
        truth = LABEL_TO_IDX[trow["variety"]]
        truth_arr[t] = truth

        perm = perms[:, t, :n_k]
        p_perm = p[rowpositions[perm]]  # (500, n_k, 12)
        p_tray = p[idx]

        if strategy in ("arith_raw", "cal_kernel_pre", "cal_group_pre", "cal_post_ref"):
            cum = np.cumsum(p_perm, axis=1)
            q_all = p_tray.mean(axis=0)
            if strategy == "cal_post_ref" and post_temps is not None and tray_folds is not None:
                T_post = post_temps[tray_folds[rec_id]]
                q_all_cal = softmax_T(np.log(np.clip(q_all[None, :], EPS, 1.0)), T_post)[0]
            else:
                q_all_cal = q_all
            full_tray_q[t] = q_all_cal
            for ki, k in enumerate(KS):
                if k == "All":
                    correct[ki, :, t] = int(q_all_cal.argmax()) == truth
                    group_nll[ki, :, t] = -np.log(max(float(q_all_cal[truth]), EPS))
                else:
                    q = cum[:, k - 1, :] / float(k)
                    if strategy == "cal_post_ref" and post_temps is not None and tray_folds is not None:
                        T_post = post_temps[tray_folds[rec_id]]
                        q = softmax_T(np.log(np.clip(q, EPS, 1.0)), T_post)
                    correct[ki, :, t] = q.argmax(axis=1) == truth
                    group_nll[ki, :, t] = -np.log(np.clip(q[:, truth], EPS, 1.0))

        elif strategy == "majority_vote":
            # Hard-label vote count; deterministic tie-break via highest arithmetic mean prob
            votes_tray = np.eye(12, dtype=np.float64)[p_tray.argmax(axis=1)]  # (n_k, 12)
            votes_perm = np.eye(12, dtype=np.float64)[p_perm.argmax(axis=2)]  # (500, n_k, 12)
            cum_v = np.cumsum(votes_perm, axis=1)
            cum_p = np.cumsum(p_perm, axis=1)
            v_all = votes_tray.sum(axis=0)
            p_all = p_tray.mean(axis=0)
            # Composite score: integer vote count + tiny prob tie-breaker (< 1e-3)
            score_all = v_all + 1e-4 * p_all
            full_tray_q[t] = v_all / float(n_k)
            for ki, k in enumerate(KS):
                if k == "All":
                    correct[ki, :, t] = int(score_all.argmax()) == truth
                    group_nll[ki, :, t] = -np.log(max(float(v_all[truth] / n_k), 1e-3))
                else:
                    vk = cum_v[:, k - 1, :]
                    pk = cum_p[:, k - 1, :] / float(k)
                    sk = vk + 1e-4 * pk
                    correct[ki, :, t] = sk.argmax(axis=1) == truth
                    group_nll[ki, :, t] = -np.log(np.clip(vk[:, truth] / float(k), 1e-3, 1.0))

        elif strategy == "log_prob_pool":
            lp_perm = np.log(np.clip(p_perm, EPS, 1.0))
            lp_tray = np.log(np.clip(p_tray, EPS, 1.0))
            cum_lp = np.cumsum(lp_perm, axis=1)
            mean_lp_all = lp_tray.mean(axis=0)
            q_all = softmax_T(mean_lp_all[None, :], 1.0)[0]
            full_tray_q[t] = q_all
            for ki, k in enumerate(KS):
                if k == "All":
                    correct[ki, :, t] = int(q_all.argmax()) == truth
                    group_nll[ki, :, t] = -np.log(max(float(q_all[truth]), EPS))
                else:
                    mk = cum_lp[:, k - 1, :] / float(k)
                    qk = softmax_T(mk, 1.0)
                    correct[ki, :, t] = qk.argmax(axis=1) == truth
                    group_nll[ki, :, t] = -np.log(np.clip(qk[:, truth], EPS, 1.0))
        else:
            raise ValueError(strategy)

    class_w = np.array([1.0 / (12.0 * np.sum(truth_arr == tr)) for tr in truth_arr], dtype=np.float64)
    group_mean = correct.mean(axis=1)
    nll_mean = group_nll.mean(axis=1)
    group_ba = group_mean @ class_w
    group_nll_bal = nll_mean @ class_w
    boot_group = group_mean @ boot_weights.T

    # Full-tray ECE (variety-balanced)
    ft_conf = full_tray_q.max(axis=1)
    ft_corr = (full_tray_q.argmax(axis=1) == truth_arr).astype(np.float64)
    ft_ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for b in range(10):
        lo, hi = bins[b], bins[b + 1]
        m = (ft_conf >= lo) & (ft_conf <= hi if b == 9 else ft_conf < hi)
        wb = class_w[m].sum()
        if wb > 0:
            ft_ece += wb * abs(np.sum(class_w[m] * ft_corr[m]) / wb - np.sum(class_w[m] * ft_conf[m]) / wb)

    res = dict(correct_trays=int(correct[5, 0, :].sum()), full_tray_ece=float(ft_ece))
    for ki, k in enumerate(KS):
        res[f"group_ba_k{k}"] = float(group_ba[ki])
        res[f"group_nll_k{k}"] = float(group_nll_bal[ki])
    return res, boot_group


def eval_pooling_external(
    p_ext: np.ndarray,
    strategy: str,
    ext_rows: pd.DataFrame,
    entity_sampling: dict[str, tuple[list[str], np.ndarray]],
) -> list[dict]:
    entities = sorted(entity_sampling.keys())
    y_all_idx = np.array([LABEL_TO_IDX[v] for v in ext_rows["variety"]], dtype=int)
    rows_out = []
    for cond in CONDITIONS:
        block = ext_rows[ext_rows["condition"].eq(cond)]
        c_idx = block.index.to_numpy()
        kq = kernel_quality_metrics(p_ext[c_idx], y_all_idx[c_idx], block["entity_tray_id"].to_numpy())

        correct = np.empty((6, 500, 10), dtype=bool)
        g_nll = np.empty((6, 500, 10), dtype=np.float64)
        for t, ent in enumerate(entities):
            sids, perm = entity_sampling[ent]
            sub = block[block["entity_tray_id"].eq(ent)].set_index("entity_seed_id").loc[sids]
            j = sub["row_index"].to_numpy(dtype=int)
            truth = LABEL_TO_IDX[sub.iloc[0]["variety"]]
            pt = p_ext[j]
            p_perm = p_ext[j[perm]]
            n_k = len(j)

            if strategy in ("arith_raw", "cal_kernel_pre", "cal_group_pre"):
                cum = np.cumsum(p_perm, axis=1)
                q_all = pt.mean(axis=0)
                for ki, k in enumerate(KS):
                    if k == "All":
                        correct[ki, :, t] = int(q_all.argmax()) == truth
                        g_nll[ki, :, t] = -np.log(max(float(q_all[truth]), EPS))
                    else:
                        qk = cum[:, k - 1, :] / float(k)
                        correct[ki, :, t] = qk.argmax(axis=1) == truth
                        g_nll[ki, :, t] = -np.log(np.clip(qk[:, truth], EPS, 1.0))
            elif strategy == "majority_vote":
                votes_t = np.eye(12)[pt.argmax(axis=1)]
                votes_p = np.eye(12)[p_perm.argmax(axis=2)]
                cum_v = np.cumsum(votes_p, axis=1)
                cum_p = np.cumsum(p_perm, axis=1)
                v_all = votes_t.sum(axis=0)
                s_all = v_all + 1e-4 * pt.mean(axis=0)
                for ki, k in enumerate(KS):
                    if k == "All":
                        correct[ki, :, t] = int(s_all.argmax()) == truth
                        g_nll[ki, :, t] = -np.log(max(float(v_all[truth] / n_k), 1e-3))
                    else:
                        vk = cum_v[:, k - 1, :]
                        sk = vk + 1e-4 * (cum_p[:, k - 1, :] / float(k))
                        correct[ki, :, t] = sk.argmax(axis=1) == truth
                        g_nll[ki, :, t] = -np.log(np.clip(vk[:, truth] / float(k), 1e-3, 1.0))
            elif strategy == "log_prob_pool":
                lp_t = np.log(np.clip(pt, EPS, 1.0))
                lp_p = np.log(np.clip(p_perm, EPS, 1.0))
                cum_lp = np.cumsum(lp_p, axis=1)
                q_all = softmax_T(lp_t.mean(axis=0)[None, :], 1.0)[0]
                for ki, k in enumerate(KS):
                    if k == "All":
                        correct[ki, :, t] = int(q_all.argmax()) == truth
                        g_nll[ki, :, t] = -np.log(max(float(q_all[truth]), EPS))
                    else:
                        qk = softmax_T(cum_lp[:, k - 1, :] / float(k), 1.0)
                        correct[ki, :, t] = qk.argmax(axis=1) == truth
                        g_nll[ki, :, t] = -np.log(np.clip(qk[:, truth], EPS, 1.0))

        rec = dict(condition=cond, correct_trays=int(correct[5, 0, :].sum()))
        rec.update(kq)
        for ki, k in enumerate(KS):
            rec[f"group_ba_k{k}"] = float(correct[ki].mean())
            rec[f"group_nll_k{k}"] = float(g_nll[ki].mean())
        rows_out.append(rec)
    return rows_out


def calibrate_single_branch(
    cid: str,
    primary_rows: pd.DataFrame,
    y_july_idx: np.ndarray,
    tray_july: np.ndarray,
    folds_july: np.ndarray,
) -> dict:
    """Fit per-fold T_kernel, T_group, and T_post strictly on inner-OOF scores of outer_train."""
    with np.load(ROOT / "models_and_preds" / f"{cid}.npz") as a:
        outer_p = a["outer_oof_prob"]
        outer_s = a["outer_oof_score"]
        ext_fold_p = a["ext_fold_prob"]
        ext_fold_s = a["ext_fold_score"]
        inner_data = {
            f: (a[f"inner_oof_idx_fold{f}"], a[f"inner_oof_prob_fold{f}"], a[f"inner_oof_score_fold{f}"])
            for f in range(4)
        }

    p_cal_kernel = np.zeros_like(outer_p)
    p_cal_group = np.zeros_like(outer_p)
    ext_cal_kernel_folds = np.zeros_like(ext_fold_p)
    ext_cal_group_folds = np.zeros_like(ext_fold_p)
    temps = []
    post_temps = {}

    for f in range(4):
        inner_idx, inner_p, inner_s = inner_data[f]
        test_idx = np.flatnonzero(folds_july == f)
        assert not (set(tray_july[inner_idx]) & set(tray_july[test_idx]))

        tk = fit_temperature_kernel(inner_s, y_july_idx[inner_idx], tray_july[inner_idx])
        tg = fit_temperature_group(inner_s, y_july_idx[inner_idx], tray_july[inner_idx], seed=20261126 + f * 1000)
        tp = fit_temperature_post_tray(inner_p, y_july_idx[inner_idx], tray_july[inner_idx])
        post_temps[f] = tp

        p_cal_kernel[test_idx] = softmax_T(outer_s[test_idx], tk)
        p_cal_group[test_idx] = softmax_T(outer_s[test_idx], tg)
        ext_cal_kernel_folds[f] = softmax_T(ext_fold_s[f], tk)
        ext_cal_group_folds[f] = softmax_T(ext_fold_s[f], tg)
        temps.append(dict(config_id=cid, fold=f, T_kernel=tk, T_group=tg, T_post=tp))

    return dict(
        raw_july=outer_p,
        raw_aug=ext_fold_p.mean(axis=0),
        cal_kernel_july=p_cal_kernel,
        cal_kernel_aug=ext_cal_kernel_folds.mean(axis=0),
        cal_group_july=p_cal_group,
        cal_group_aug=ext_cal_group_folds.mean(axis=0),
        temps=temps,
        post_temps=post_temps,
    )


def main() -> None:
    t0 = time.monotonic()
    out_dir = ROOT / "stage3_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "stage3.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})
    ext_rows["row_index"] = np.arange(len(ext_rows))
    y_july_idx = np.array([LABEL_TO_IDX[v] for v in primary_rows["variety_name"]], dtype=int)
    tray_july = primary_rows["record_id"].to_numpy()
    folds_july = primary_rows["fold"].to_numpy()
    tray_folds = dict(zip(primary_rows["record_id"], primary_rows["fold"]))

    saved_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    rowpositions = pd.Series(np.arange(len(primary_rows)), index=primary_rows["seed_id"]).loc[saved_rows["seed_id"]].to_numpy()
    trays_df = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    perms = np.load(LONG / "batch2_schedules.npz")["permutations"]
    boot_weights = np.load(LONG / "batch2_bootstrap.npz")["weights"]

    base_ext = ext_rows[ext_rows["condition"].eq("BASE")].sort_values(["entity_tray_id", "entity_seed_id"])
    entities = sorted(base_ext["entity_tray_id"].unique())
    entity_sampling: dict[str, tuple[list[str], np.ndarray]] = {}
    for t, ent in enumerate(entities):
        sids = base_ext.loc[base_ext["entity_tray_id"].eq(ent), "entity_seed_id"].tolist()
        rng = np.random.default_rng(20261114 + t)
        entity_sampling[ent] = (sids, np.stack([rng.permutation(len(sids)) for _ in range(500)]))

    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rf:
        d_july = rf["d_prob_july"]
        d_aug = rf["d_prob_aug"]

    # 1. Fit nested per-fold temperatures for M_jpeg, M_raw, and all 18 SWIR configs
    branch_ids = ["M_jpeg", "M_raw"] + ALL_SWIR
    calibrated: dict[str, dict] = {}
    temp_rows = []
    for cid in branch_ids:
        cal = calibrate_single_branch(cid, primary_rows, y_july_idx, tray_july, folds_july)
        calibrated[cid] = cal
        temp_rows.extend(cal["temps"])
        tk_m = np.mean([r["T_kernel"] for r in cal["temps"]])
        tg_m = np.mean([r["T_group"] for r in cal["temps"]])
        logging.info("TEMP %-28s mean T_kernel=%.3f  mean T_group=%.3f", cid, tk_m, tg_m)
    pd.DataFrame(temp_rows).to_csv(out_dir / "fitted_temperatures.csv", index=False)

    # 2. Evaluate all 6 strategies on single-modality SWIR (14 factorial + 4 bridge), M_jpeg, M_raw,
    #    and multimodal MS_jpeg / MS_raw / MSD_jpeg / MSD_raw (for all 14 factorial SWIR + 4 bridge)
    primary_rows_out = []
    primary_contrasts_out = []
    ext_rows_out = []

    def evaluate_entity(
        entity_name: str,
        entity_family: str,
        p_raw_j: np.ndarray,
        p_raw_a: np.ndarray,
        p_ck_j: np.ndarray,
        p_ck_a: np.ndarray,
        p_cg_j: np.ndarray,
        p_cg_a: np.ndarray,
        post_temps: dict[int, float] | None,
    ) -> None:
        strategies = [
            ("arith_raw", p_raw_j, p_raw_a),
            ("majority_vote", p_raw_j, p_raw_a),
            ("log_prob_pool", p_raw_j, p_raw_a),
            ("cal_kernel_pre", p_ck_j, p_ck_a),
            ("cal_group_pre", p_cg_j, p_cg_a),
        ]
        if post_temps is not None:
            strategies.append(("cal_post_ref", p_raw_j, p_raw_a))

        boots = {}
        raw_res = None
        for strat, pj, pa in strategies:
            kq = kernel_quality_metrics(pj, y_july_idx, tray_july)
            pres, bg = eval_pooling_primary(
                pj, strat, rowpositions, trays_df, saved_rows, perms, boot_weights,
                post_temps=post_temps if strat == "cal_post_ref" else None,
                tray_folds=tray_folds if strat == "cal_post_ref" else None,
            )
            boots[strat] = bg
            if strat == "arith_raw":
                raw_res = pres
            if strat == "cal_post_ref":
                # Verify mathematical invariance of positive temperature scaling on Group BA
                for k in KS:
                    assert abs(pres[f"group_ba_k{k}"] - raw_res[f"group_ba_k{k}"]) < 1e-12, (entity_name, k)
            row = dict(entity=entity_name, family=entity_family, strategy=strat)
            row.update(kq)
            row.update(pres)
            primary_rows_out.append(row)

            if strat != "arith_raw":
                for ki, k in enumerate(KS):
                    delta_b = bg[ki] - boots["arith_raw"][ki]
                    primary_contrasts_out.append(
                        dict(
                            entity=entity_name,
                            family=entity_family,
                            strategy=strat,
                            endpoint=f"k{k}",
                            delta_ba=float(pres[f"group_ba_k{k}"] - raw_res[f"group_ba_k{k}"]),
                            ci_low=float(np.quantile(delta_b, 0.025)),
                            ci_high=float(np.quantile(delta_b, 0.975)),
                            delta_nll=float(pres[f"group_nll_k{k}"] - raw_res[f"group_nll_k{k}"]),
                        )
                    )

            if strat != "cal_post_ref":
                ec = eval_pooling_external(pa, strat, ext_rows, entity_sampling)
                for r in ec:
                    r.update(entity=entity_name, family=entity_family, strategy=strat)
                ext_rows_out.extend(ec)

    # Single-branch evaluations
    for cid in branch_ids:
        c = calibrated[cid]
        fam = "RGB_M" if cid.startswith("M_") else ("SWIR_factorial" if cid in FACTORIAL_SWIR else "SWIR_bridge")
        evaluate_entity(
            cid, fam,
            c["raw_july"], c["raw_aug"],
            c["cal_kernel_july"], c["cal_kernel_aug"],
            c["cal_group_july"], c["cal_group_aug"],
            c["post_temps"],
        )

    # Multimodal MS (both M & S calibrated) and MSD (M & S calibrated, D fixed)
    mj = calibrated["M_jpeg"]
    mr = calibrated["M_raw"]
    for cid in ALL_SWIR:
        s = calibrated[cid]
        # MS_jpeg and MS_raw
        evaluate_entity(
            f"MS_jpeg__{cid}", "MS_jpeg",
            (mj["raw_july"] + s["raw_july"]) / 2.0, (mj["raw_aug"] + s["raw_aug"]) / 2.0,
            (mj["cal_kernel_july"] + s["cal_kernel_july"]) / 2.0, (mj["cal_kernel_aug"] + s["cal_kernel_aug"]) / 2.0,
            (mj["cal_group_july"] + s["cal_group_july"]) / 2.0, (mj["cal_group_aug"] + s["cal_group_aug"]) / 2.0,
            None,
        )
        evaluate_entity(
            f"MS_raw__{cid}", "MS_raw",
            (mr["raw_july"] + s["raw_july"]) / 2.0, (mr["raw_aug"] + s["raw_aug"]) / 2.0,
            (mr["cal_kernel_july"] + s["cal_kernel_july"]) / 2.0, (mr["cal_kernel_aug"] + s["cal_kernel_aug"]) / 2.0,
            (mr["cal_group_july"] + s["cal_group_july"]) / 2.0, (mr["cal_group_aug"] + s["cal_group_aug"]) / 2.0,
            None,
        )
        # MSD_jpeg and MSD_raw (D fixed, M and S calibrated)
        evaluate_entity(
            f"MSD_jpeg__{cid}", "MSD_jpeg",
            (mj["raw_july"] + d_july + s["raw_july"]) / 3.0, (mj["raw_aug"] + d_aug + s["raw_aug"]) / 3.0,
            (mj["cal_kernel_july"] + d_july + s["cal_kernel_july"]) / 3.0, (mj["cal_kernel_aug"] + d_aug + s["cal_kernel_aug"]) / 3.0,
            (mj["cal_group_july"] + d_july + s["cal_group_july"]) / 3.0, (mj["cal_group_aug"] + d_aug + s["cal_group_aug"]) / 3.0,
            None,
        )
        evaluate_entity(
            f"MSD_raw__{cid}", "MSD_raw",
            (mr["raw_july"] + d_july + s["raw_july"]) / 3.0, (mr["raw_aug"] + d_aug + s["raw_aug"]) / 3.0,
            (mr["cal_kernel_july"] + d_july + s["cal_kernel_july"]) / 3.0, (mr["cal_kernel_aug"] + d_aug + s["cal_kernel_aug"]) / 3.0,
            (mr["cal_group_july"] + d_july + s["cal_group_july"]) / 3.0, (mr["cal_group_aug"] + d_aug + s["cal_group_aug"]) / 3.0,
            None,
        )

    pd.DataFrame(primary_rows_out).to_csv(out_dir / "primary_pooling_comparison.csv", index=False)
    pd.DataFrame(primary_contrasts_out).to_csv(out_dir / "primary_pooling_contrasts.csv", index=False)
    pd.DataFrame(ext_rows_out).to_csv(out_dir / "external_pooling_comparison.csv", index=False)

    logging.info("Stage 3 COMPLETE in %.1fs. Saved pooling/calibration comparisons.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
