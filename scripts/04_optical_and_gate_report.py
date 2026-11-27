"""Stage 4: Optical spectral characterization (12-cultivar reflectance spectra,
within-kernel spatial spectral heterogeneity, 6-condition spectral perturbations,
and reuse of verified random-pixel sampling results) + comprehensive Go/No-Go
Gate Report (RESULTS_GATE_REPORT.md) summarizing all Stage 0-4 verified outputs.
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

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PILOT = PROJECT / "analysis" / "roi_pixel_pilot_20261113"
sys.path.insert(0, str(PROJECT))
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE

LABELS = np.asarray(FROZEN_LABEL_SPACE)
CONDITIONS = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]


def compute_optical_summaries(out_dir: Path) -> dict:
    primary_rows = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    ext_rows = pd.read_csv(ROOT / "cache" / "external_rows.csv", dtype={"record_id": str})

    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        waves = psp["waves"]
        p_med = psp["median"]
        p_stat = psp["statistics"]  # [mean, std, q10, q50, q90]
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        e_med = esp["median"]
        e_stat = esp["statistics"]

    # 1. Tray-balanced cultivar mean reflectance and between-tray SD (primary cohort)
    p_mean = p_stat[:, 0, :]  # (3627, 421)
    p_sd = p_stat[:, 1, :]    # (3627, 421) within-kernel spatial SD per band

    cultivar_mean = {}
    cultivar_tray_sd = {}
    cultivar_within_sd = {}
    hetero_rows = []

    for var_id, grp in primary_rows.groupby("variety_id", sort=True):
        var_name = str(grp["variety_name"].iloc[0])
        tray_means = []
        tray_meds = []
        tray_sds = []
        for rec, tgrp in grp.groupby("record_id", sort=True):
            idx = tgrp.index.to_numpy()
            tray_means.append(p_mean[idx].mean(axis=0))
            tray_meds.append(p_med[idx].mean(axis=0))
            tray_sds.append(p_sd[idx].mean(axis=0))
        tm = np.asarray(tray_means)
        ts = np.asarray(tray_sds)
        cultivar_mean[var_id] = tm.mean(axis=0)
        cultivar_tray_sd[var_id] = tm.std(axis=0, ddof=1)
        cultivar_within_sd[var_id] = ts.mean(axis=0)
        idx_all = grp.index.to_numpy()
        # Median vs mean absolute difference across bands (raw reflectance)
        med_mean_abs = float(np.mean(np.abs(p_mean[idx_all] - p_med[idx_all])))
        hetero_rows.append(
            dict(
                variety_id=var_id,
                variety_name=var_name,
                n_trays=int(grp["record_id"].nunique()),
                n_kernels=int(len(idx_all)),
                mean_reflectance=float(tm.mean()),
                between_tray_sd_mean=float(tm.std(axis=0, ddof=1).mean()),
                within_kernel_spatial_sd_mean=float(p_sd[idx_all].mean()),
                within_kernel_iqr90_10_mean=float((p_stat[idx_all, 4, :] - p_stat[idx_all, 2, :]).mean()),
                mean_minus_median_abs_diff=med_mean_abs,
                mean_inner_pixels=float(grp["pixels_inner"].mean()),
            )
        )

    pd.DataFrame(hetero_rows).to_csv(out_dir / "cultivar_optical_heterogeneity.csv", index=False)
    np.savez_compressed(
        out_dir / "cultivar_spectra_summary.npz",
        waves=waves,
        variety_ids=np.array(sorted(cultivar_mean.keys())),
        mean_spectra=np.stack([cultivar_mean[v] for v in sorted(cultivar_mean.keys())]),
        tray_sd_spectra=np.stack([cultivar_tray_sd[v] for v in sorted(cultivar_tray_sd.keys())]),
        within_kernel_sd_spectra=np.stack([cultivar_within_sd[v] for v in sorted(cultivar_within_sd.keys())]),
    )

    # 2. Spectral shift across the 6 August imaging conditions (280 paired physical kernels)
    e_mean = e_stat[:, 0, :]
    e_sd = e_stat[:, 1, :]
    base_df = ext_rows[ext_rows["condition"].eq("BASE")].set_index("entity_seed_id")
    base_idx = ext_rows.index[ext_rows["condition"].eq("BASE")].to_numpy()
    base_lookup = dict(zip(ext_rows.loc[base_idx, "entity_seed_id"], base_idx))

    pert_rows = []
    for cond in CONDITIONS:
        c_df = ext_rows[ext_rows["condition"].eq(cond)]
        c_lookup = dict(zip(c_df["entity_seed_id"], c_df.index.to_numpy()))
        b_i = np.array([base_lookup[sid] for sid in base_df.index], dtype=int)
        c_i = np.array([c_lookup[sid] for sid in base_df.index], dtype=int)

        diff_raw = e_mean[c_i] - e_mean[b_i]
        # SNV normalized difference
        def _snv(a):
            return (a - a.mean(axis=1, keepdims=True)) / np.maximum(a.std(axis=1, keepdims=True), 1e-6)
        diff_snv_mean = _snv(e_mean[c_i]) - _snv(e_mean[b_i])
        diff_snv_med = _snv(e_med[c_i]) - _snv(e_med[b_i])

        pert_rows.append(
            dict(
                condition=cond,
                n_physical_kernels=280,
                mean_reflectance=float(e_mean[c_i].mean()),
                within_kernel_spatial_sd=float(e_sd[c_i].mean()),
                raw_mean_bias_vs_BASE=float(diff_raw.mean()),
                raw_rmse_vs_BASE=float(np.sqrt(np.mean(diff_raw ** 2))),
                snv_mean_rmse_vs_BASE=float(np.sqrt(np.mean(diff_snv_mean ** 2))),
                snv_median_rmse_vs_BASE=float(np.sqrt(np.mean(diff_snv_med ** 2))),
            )
        )
    pd.DataFrame(pert_rows).to_csv(out_dir / "external_spectral_perturbation_shifts.csv", index=False)

    # 3. Copy/verify verified random-pixel sampling results from roi_pixel_pilot_20261113/metrics.csv
    pilot_m = pd.read_csv(PILOT / "metrics.csv")
    pixel_rows = []
    for key_prefix, label in [
        ("point1_mean", "1 pixel (mean)"),
        ("point4_mean", "4 pixels (mean)"),
        ("point16_mean", "16 pixels (mean)"),
        ("point32_mean", "32 pixels (mean)"),
        ("patch3x3_32_mean", "32 x 3x3 patches (mean)"),
        ("all_inner_mean", "All inner pixels (mean)"),
        ("point1_median", "1 pixel (median)"),
        ("point4_median", "4 pixels (median)"),
        ("point16_median", "16 pixels (median)"),
        ("point32_median", "32 pixels (median)"),
        ("median_refit", "All inner pixels (median)"),
        ("distribution_snv", "All inner pixels (distribution)"),
    ]:
        sub = pilot_m[pilot_m["configuration"].str.startswith(key_prefix)]
        if len(sub) > 0:
            pixel_rows.append(
                dict(
                    config_group=key_prefix,
                    label=label,
                    n_repeats=int(len(sub)),
                    mean_kernel_ba=float(sub["balanced_accuracy"].mean()),
                    min_kernel_ba=float(sub["balanced_accuracy"].min()),
                    max_kernel_ba=float(sub["balanced_accuracy"].max()),
                )
            )
    pd.DataFrame(pixel_rows).to_csv(out_dir / "intrakernel_pixel_sampling_summary.csv", index=False)
    return dict(
        n_cultivars=len(hetero_rows),
        overall_within_kernel_sd=float(p_sd.mean()),
        overall_med_mean_diff=float(np.mean(np.abs(p_mean - p_med))),
    )


def build_gate_report(out_dir: Path) -> None:
    s2 = ROOT / "stage2_diagnostics"
    s3 = ROOT / "stage3_calibration"

    pm = pd.read_csv(s2 / "primary_metrics.csv")
    rk = pd.read_csv(s2 / "rank_correlations.csv")
    ec = pd.read_csv(s2 / "external_6cond_metrics.csv")
    comp_p = pd.read_csv(s2 / "complementarity_summary_primary.csv")
    comp_a = pd.read_csv(s2 / "complementarity_summary_aug_base.csv")
    p3 = pd.read_csv(s3 / "primary_pooling_comparison.csv")
    c3 = pd.read_csv(s3 / "primary_pooling_contrasts.csv")
    e3 = pd.read_csv(s3 / "external_pooling_comparison.csv")
    temps = pd.read_csv(s3 / "fitted_temperatures.csv")
    opt_p = pd.read_csv(out_dir / "external_spectral_perturbation_shifts.csv")

    lines = [
        "# Path B+ Results & Go/No-Go Gate Report",
        "",
        f"Generated UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## 1. Cross-Factorial Single-Modality SWIR Results (Primary Cohort: 3,627 kernels, 67 trays, 12 cultivars)",
        "",
    ]

    swir_fac = pm[pm["family"].eq("SWIR_factorial")].sort_values(["representation", "classifier"])
    cols1 = [
        "configuration", "classifier", "representation",
        "kernel_ba", "kernel_acc", "kernel_nll", "kernel_ece10",
        "group_ba_k3", "group_ba_k10", "group_ba_k20", "group_ba_kAll", "correct_trays",
        "mean_mu_star", "finite_pop_var_r2", "mean_top1_rival_share",
    ]
    lines.append(swir_fac[cols1].round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 2. Rank Correlations Between Kernel BA and Group BA(k)")
    lines.append("")
    lines.append(rk.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 3. RGB Baselines & Multimodal Fusion (Primary Cohort & August BASE)")
    lines.append("")
    rgb_rows = pm[pm["family"].eq("RGB")]
    lines.append("### 3.1 RGB Baselines (Primary Cohort)")
    lines.append(rgb_rows[["configuration", "kernel_ba", "kernel_nll", "kernel_ece10", "group_ba_k3", "group_ba_k20", "group_ba_kAll", "correct_trays", "mean_mu_star"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append("### 3.2 Multimodal MSD_jpeg & MSD_raw Across Representative SWIR Configs (Primary Cohort)")
    msd_sub = pm[pm["family"].isin(["MSD_jpeg", "MSD_raw"]) & pm["classifier"].isin(["logistic", "logistic_fixedC1", "rbf_svm", "extra_trees", "cnn1d"])]
    lines.append(msd_sub[["configuration", "family", "classifier", "representation", "kernel_ba", "group_ba_k3", "group_ba_k20", "group_ba_kAll", "correct_trays"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append("### 3.3 August BASE (10 trays) Across RGB & Factorial SWIR / MSD")
    ec_base = ec[ec["condition"].eq("BASE") & ec["family"].isin(["RGB", "SWIR_factorial", "MSD_jpeg", "MSD_raw"])]
    lines.append(ec_base[["configuration", "family", "kernel_ba", "kernel_nll", "kernel_ece10", "group_ba_k3", "group_ba_k20", "group_ba_kAll", "correct_trays"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Calibration & Multi-Kernel Pooling Strategies (Stage 3)")
    lines.append("")
    lines.append("### 4.1 Mean Nested Fitted Temperatures (4 Outer Folds)")
    t_sum = temps.groupby("config_id", as_index=False)[["T_kernel", "T_group", "T_post"]].agg(["mean", "std"])
    t_sum.columns = ["_".join(c).strip("_") for c in t_sum.columns]
    lines.append(t_sum.round(3).to_markdown(index=False))
    lines.append("")

    lines.append("### 4.2 Single-Modality SWIR: Mean Across 14 Factorial Configs by Pooling Strategy (Primary)")
    p3_swir = p3[p3["family"].eq("SWIR_factorial")]
    agg3 = p3_swir.groupby("strategy", as_index=False)[
        ["kernel_ba", "kernel_nll", "kernel_ece10", "group_ba_k3", "group_ba_k10", "group_ba_k20", "group_ba_kAll", "group_nll_k3", "group_nll_kAll", "full_tray_ece", "correct_trays"]
    ].mean()
    lines.append(agg3.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("### 4.3 Single-Modality SWIR: Mean Across 14 Factorial Configs by Pooling Strategy (August BASE & All 6 Conds)")
    e3_swir = e3[e3["family"].eq("SWIR_factorial")]
    agg3_base = e3_swir[e3_swir["condition"].eq("BASE")].groupby("strategy", as_index=False)[
        ["kernel_ba", "kernel_nll", "kernel_ece10", "group_ba_k3", "group_ba_k20", "group_ba_kAll", "group_nll_kAll", "correct_trays"]
    ].mean()
    lines.append("**August BASE:**")
    lines.append(agg3_base.round(4).to_markdown(index=False))
    lines.append("")
    agg3_6c = e3_swir.groupby("strategy", as_index=False)[
        ["kernel_ba", "kernel_nll", "kernel_ece10", "group_ba_k3", "group_ba_k20", "group_ba_kAll", "group_nll_kAll", "correct_trays"]
    ].mean()
    lines.append("**August 6-Condition Mean:**")
    lines.append(agg3_6c.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("### 4.4 Multimodal MSD_raw & MSD_jpeg: Mean Across 14 Factorial SWIR Configs by Strategy")
    for fam in ("MSD_jpeg", "MSD_raw"):
        sub_p = p3[p3["family"].eq(fam) & p3["entity"].str.split("__").str[1].isin(p3_swir["entity"].unique())]
        sub_e = e3[e3["family"].eq(fam) & e3["condition"].eq("BASE") & e3["entity"].str.split("__").str[1].isin(p3_swir["entity"].unique())]
        lines.append(f"**{fam} — Primary:**")
        lines.append(sub_p.groupby("strategy", as_index=False)[["group_ba_k3", "group_ba_k20", "group_ba_kAll", "group_nll_kAll", "correct_trays"]].mean().round(4).to_markdown(index=False))
        lines.append(f"**{fam} — August BASE:**")
        lines.append(sub_e.groupby("strategy", as_index=False)[["group_ba_k3", "group_ba_k20", "group_ba_kAll", "group_nll_kAll", "correct_trays"]].mean().round(4).to_markdown(index=False))
        lines.append("")

    lines.append("### 4.5 Detailed Single-Config Pooling Contrasts vs arith_raw (Primary k=3 and k=All, Factorial SWIR)")
    c3_sub = c3[c3["family"].eq("SWIR_factorial") & c3["endpoint"].isin(["k3", "kAll"]) & c3["strategy"].isin(["majority_vote", "log_prob_pool", "cal_kernel_pre", "cal_group_pre"])]
    piv = c3_sub.pivot_table(index=["entity", "endpoint"], columns="strategy", values="delta_ba")
    lines.append(piv.round(4).to_markdown())
    lines.append("")

    lines.append("## 5. Optical Perturbation Shifts Across 6 August Imaging Conditions")
    lines.append("")
    lines.append(opt_p.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 6. Multimodal Margin Complementarity Summary (Primary & August BASE)")
    lines.append("")
    lines.append("### Primary (67 trays)")
    lines.append(comp_p[comp_p["scheme"].str.contains("S_logistic_|S_rbf_svm_|S_extra_trees_|S_cnn1d_") & ~comp_p["scheme"].str.contains("fixedC1")].round(4).to_markdown(index=False))
    lines.append("")
    lines.append("### August BASE (10 trays)")
    lines.append(comp_a[comp_a["scheme"].str.contains("S_logistic_|S_rbf_svm_|S_extra_trees_|S_cnn1d_") & ~comp_a["scheme"].str.contains("fixedC1")].round(4).to_markdown(index=False))
    lines.append("")

    (ROOT / "RESULTS_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.monotonic()
    out_dir = ROOT / "stage4_optical"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "stage4.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    opt_meta = compute_optical_summaries(out_dir)
    build_gate_report(out_dir)
    logging.info("Stage 4 COMPLETE in %.1fs. Optical meta: %s", time.monotonic() - t0, opt_meta)


if __name__ == "__main__":
    main()
