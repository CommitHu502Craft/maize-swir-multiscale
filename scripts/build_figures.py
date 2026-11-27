"""Generate six publication-grade main figures (vector PDF + 600-dpi PNG)
for the Infrared Physics & Technology (IPT) submission strictly from verified
Path B+ outputs (Stages 0-6, including P0 unified tray-equal rankings and P1
wavelength-dependent affine residuals, SNV vs. non-SNV ablation, spectral-level
vs. probability-level pooling, and unified RGB comparator analyses).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import base64
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PBP = PROJECT / "analysis" / "path_b_plus"
S6 = PBP / "stage6_ipt_revision"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.3,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CLF_COLORS = {
    "logistic": "#0072B2",
    "rbf_svm": "#D55E00",
    "extra_trees": "#009E73",
    "cnn1d": "#CC79A7",
}
CLF_LABELS = {
    "logistic": "Logistic",
    "rbf_svm": "RBF-SVM",
    "extra_trees": "ExtraTrees",
    "cnn1d": "1D-CNN",
}
REP_MARKERS = {
    "median": "o",
    "mean": "s",
    "mean_std": "^",
    "distribution": "D",
}
REP_LABELS = {
    "median": "Median",
    "mean": "Mean",
    "mean_std": "Mean+SD",
    "distribution": "Distribution",
}


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, text: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


def _extract_setup_photos() -> tuple[np.ndarray, np.ndarray]:
    """Read-only extraction of recorded setup and example tray photos from archived SVG."""
    svg_path = PROJECT / "analysis" / "food_control_submission" / "figures" / "main" / "Figure_1_experimental_design.svg"
    root = ET.parse(svg_path).getroot()
    imgs = [el for el in root.iter() if el.tag.endswith("image")]

    def decode_el(el) -> np.ndarray:
        href = el.attrib.get("{http://www.w3.org/1999/xlink}href") or el.attrib.get("href", "")
        b64 = href.split(",", 1)[1]
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

    return decode_el(imgs[0]), decode_el(imgs[1])


def _draw_box(ax, x, y, w, h, title, body="", fc="#F7FAFC", ec="#2B6CB0", title_col="#1A365D", lw=1.1, tsize=7.6, bsize=6.7):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    if body:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=title_col, zorder=4)
        ax.text(x + w / 2, y + h * 0.29, body, ha="center", va="center", fontsize=bsize, color="#333333", linespacing=1.22, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=title_col, zorder=4)


def _draw_arrow(ax, x0, y0, x1, y1, color="#4A5568", lw=1.2):
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4.5,head_width=2.8",
                          color=color, lw=lw, zorder=5, mutation_scale=1.0)
    ax.add_patch(arr)


def build_figure_1() -> None:
    """Figure 1: Paired RGB-SWIR imaging apparatus, perturbation protocol, and
    dual-scale SWIR optical information-aggregation framework (vector SVG ->
    PDF & 600-DPI PNG via build_figure1_svg.py).
    """
    from build_figure1_svg import build_figure_1 as _build_svg_fig1

    _build_svg_fig1()
    fig1_png = FIG_DIR / "Figure_1_experimental_framework.png"
    with Image.open(fig1_png) as im:
        im.load()
        im.save(fig1_png, dpi=(600, 600))


def build_figure_2() -> None:
    """Figure 2: Optical SWIR characterization, Scale 1 pixel aggregation,
    scale-matched perturbation spectroscopy, and 421-band wavelength-dependent
    paired affine residuals.
    """
    spec = np.load(PBP / "stage4_optical/cultivar_spectra_summary.npz")
    waves = spec["waves"]
    vids = spec["variety_ids"]
    means = spec["mean_spectra"]
    within_sd = spec["within_kernel_sd_spectra"]

    pix = pd.read_csv(PBP / "stage4_optical/intrakernel_pixel_sampling_summary.csv")
    opt = pd.read_csv(PBP / "stage5_strengthening/p0_optical_perturbation_physics.csv")
    wres = pd.read_csv(S6 / "p1_wavelength_affine_residuals.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    plt.subplots_adjust(wspace=0.33, hspace=0.36)

    # Panel A: 12-cultivar mean reflectance spectra + within-kernel SD envelope
    ax = axes[0, 0]
    cmap = plt.cm.tab20(np.linspace(0, 0.95, 12))
    grand_mean = means.mean(axis=0)
    grand_within = within_sd.mean(axis=0)
    ax.fill_between(waves, grand_mean - grand_within, grand_mean + grand_within, color="#BDBDBD", alpha=0.34)
    for i in range(len(vids)):
        ax.plot(waves, means[i], color=cmap[i], lw=1.0, alpha=0.92)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative reflectance")
    ax.set_title("SWIR cultivar reflectance & intra-kernel spread")
    ax.set_xlim(950, 1650)
    ax.set_ylim(0.05, 0.68)
    ax.grid(alpha=0.25, ls="--")
    h = [
        Line2D([0], [0], color="#444444", lw=1.1, label="12 cultivar tray-balanced means"),
        Rectangle((0, 0), 1, 1, fc="#BDBDBD", alpha=0.45, label="Within-kernel spatial ±1 SD (~9× tray SD)"),
    ]
    ax.legend(handles=h, loc="lower left", frameon=True, framealpha=0.93)
    panel_label(ax, "A")

    # Panel B: Scale 1 intra-kernel pixel aggregation
    ax = axes[0, 1]
    pidx = pix.set_index("config_group")
    x_pos = np.arange(5)
    x_lbls = ["1 px", "4 px", "16 px", "32 px", "All inner\n(~450 px)"]
    mean_keys = ["point1_mean", "point4_mean", "point16_mean", "point32_mean", "all_inner_mean"]
    med_keys = ["point1_median", "point4_median", "point16_median", "point32_median", "median_refit"]
    y_mean = [pidx.loc[k, "mean_kernel_ba"] * 100 for k in mean_keys]
    y_med = [pidx.loc[k, "mean_kernel_ba"] * 100 for k in med_keys]
    ax.plot(x_pos, y_mean, "s-", color="#0072B2", lw=1.8, ms=5.5, label="Pixel mean (SNV)")
    ax.plot(x_pos, y_med, "o--", color="#D55E00", lw=1.8, ms=5.5, label="Pixel median (SNV)")
    dist_ba = pidx.loc["distribution_snv", "mean_kernel_ba"] * 100
    ax.scatter([4], [dist_ba], marker="D", s=46, color="#009E73", zorder=5, label="Mean + dispersion (1,263 feat.)")
    ax.annotate(f"{dist_ba:.1f}%", (4, dist_ba), xytext=(4.0, dist_ba + 2.6), ha="center", va="bottom",
                fontsize=7.6, color="#009E73", fontweight="bold")
    ax.annotate(f"{y_mean[-1]:.1f}%", (4, y_mean[-1]), xytext=(4.14, y_mean[-1] - 0.8), ha="left", va="center",
                fontsize=7.6, color="#0072B2", fontweight="bold")
    ax.annotate(f"{y_med[-1]:.1f}%", (4, y_med[-1]), xytext=(4.14, y_med[-1]), ha="left", va="center",
                fontsize=7.6, color="#D55E00", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbls)
    ax.set_xlim(-0.25, 4.72)
    ax.set_xlabel("Sampled inner-mask pixels per kernel")
    ax.set_ylabel("Single-kernel OOF BA (%)")
    ax.set_title("Scale 1: Intra-kernel pixel aggregation (Logistic)")
    ax.set_ylim(18, 69)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    panel_label(ax, "B")

    # Panel C: Scale-matched raw vs affine-residual RMSE + SAM
    ax = axes[1, 0]
    osub = opt[(opt["summary"] == "mean") & (opt["condition"] != "BASE")].set_index("condition")
    conds = ["REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
    osub = osub.loc[conds]
    x = np.arange(len(conds))
    w = 0.32
    ax.bar(x - w / 2, osub["raw_rmse"], width=w, color="#8C8C8C", label="Raw reflectance RMSE")
    ax.bar(x + w / 2, osub["affine_residual_rmse"], width=w, color="#0072B2", label="Affine-residual RMSE (same units)")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, osub["mean_sam_deg"], "D-", color="#D55E00", lw=1.7, ms=5.6, label="Mean Spectral Angle (SAM, °)")
    ax2.set_ylabel("Mean Spectral Angle Mapper, SAM (°)", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0, 7.6)
    ax2.annotate("FLIP: 5.34°\n(0.0080 resid.)", xy=(1.96, 5.34), xytext=(0.05, 3.70),
                 arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.85), fontsize=7.0, color="#B84A00", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E0C0A8", alpha=0.95))
    ax.annotate("Affine alignment cuts\nRMSE 98.5–98.7%", xy=(3.18, 0.011), xytext=(3.48, 0.296), ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9),
                fontsize=6.8, color="#005B8F", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#C0D8E8", alpha=0.96))
    ax.set_xticks(x)
    ax.set_xticklabels(["REP", "REPOS", "FLIP", "LEFT\nOFF", "RIGHT\nOFF"])
    ax.set_xlim(-0.55, 4.55)
    ax.set_xlabel("Paired perturbation vs. BASE (280 physical kernels)")
    ax.set_ylabel("Reflectance RMSE (identical units)")
    ax.set_title("Scale-matched affine residual & spectral angle")
    ax.set_ylim(0, 0.345)
    ax.grid(axis="y", alpha=0.25, ls="--")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.94, fontsize=6.2)
    panel_label(ax, "C")

    # Panel D (NEW): Wavelength-dependent paired affine residual RMS across 421 SWIR bands
    ax = axes[1, 1]
    c_styles = [
        ("FLIP", "FLIP (mean 0.0073; peak 1,464 nm)", "#D55E00", "-", 1.8, True),
        ("LEFT_OFF", "LEFT_OFF (mean 0.0034; −98.5% vs raw)", "#0072B2", "-", 1.5, True),
        ("REPOSITION", "REPOSITION (mean 0.0033)", "#009E73", "--", 1.3, False),
        ("RIGHT_OFF", "RIGHT_OFF (mean 0.0023; −98.7% vs raw)", "#56B4E9", "-.", 1.4, False),
        ("REP", "REP noise floor (mean 0.0017)", "#666666", ":", 1.3, False),
    ]
    for cond, lbl, col, ls, lw, shade in c_styles:
        sub = wres[wres["condition"].eq(cond)].sort_values("wavelength_nm")
        wv = sub["wavelength_nm"].to_numpy()
        rm = sub["tray_mean_rms_resid"].to_numpy()
        ax.plot(wv, rm, ls=ls, color=col, lw=lw, label=lbl)
        if shade:
            ax.fill_between(wv, sub["tray_q25_rms_resid"], sub["tray_q75_rms_resid"], color=col, alpha=0.14, edgecolor="none")
    ax.axvspan(1100, 1250, color="#E2E8F0", alpha=0.45, zorder=0)
    ax.annotate("Min. affine\nresidual\n(99.2–99.4%)", xy=(1160, 0.0021), xytext=(1160, 0.0114),
                ha="center", va="bottom", fontsize=6.4, color="#4A5568",
                arrowprops=dict(arrowstyle="->", color="#718096", lw=0.75),
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E0", alpha=0.92))
    ax.annotate("FLIP non-affine peak\n1,464 nm (0.0127)", xy=(1463.6, 0.0129), xytext=(1490, 0.0172),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.85), fontsize=6.8, color="#B84A00", fontweight="bold")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Tray-mean RMS affine residual (reflectance)")
    ax.set_title("Wavelength-dependent paired affine residuals (10 trays)")
    ax.set_xlim(950, 1650)
    ax.set_ylim(0, 0.0222)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=6.1)
    panel_label(ax, "D")

    save_fig(fig, "Figure_2_optical_perturbation_spectra")


def build_figure_3() -> None:
    """Figure 3: Unified tray-equal scale decoupling across 14 SWIR configurations
    AND spectral-level (early) vs. probability-level (late) pooling comparison
    (Protocol 1 fixed single-kernel models + Protocol 2 scale-matched training).
    """
    pm = pd.read_csv(PBP / "stage2_diagnostics/primary_metrics.csv")
    u_cfg = pd.read_csv(S6 / "p0_swir14_unified_summary.csv").set_index("configuration")
    rk = pd.read_csv(S6 / "p0_unified_tray_equal_rankings.csv")
    ep_p = pd.read_csv(S6 / "p1_early_vs_late_primary.csv")
    ep_a = pd.read_csv(S6 / "p1_early_vs_late_august.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    plt.subplots_adjust(wspace=0.34, hspace=0.38)

    # Panel A: Tray-equal single-kernel BA vs Full-tray BA (14 SWIR)
    ax = axes[0, 0]
    sf = pm[pm["family"].eq("SWIR_factorial")].copy()
    sf["tray_equal_k1_ba"] = sf["configuration"].map(u_cfg["tray_equal_k1_ba"])
    for _, r in sf.iterrows():
        ax.scatter(
            r["tray_equal_k1_ba"] * 100, r["group_ba_kAll"] * 100,
            color=CLF_COLORS[r["classifier"]], marker=REP_MARKERS[r["representation"]],
            s=58, edgecolor="white", linewidth=0.6, zorder=4,
        )
    sf_idx = sf.set_index("configuration")
    callouts = [
        ("S_cnn1d_mean", "1D-CNN (mean)", (-4, 8), "center", False),
        ("S_extra_trees_mean", "ET (mean)", (8, 5), "left", False),
        ("S_extra_trees_median", "ET (med)", (0, 8), "center", False),
        ("S_logistic_mean_std", "LR (m+SD)", (0, -14), "center", False),
        ("S_logistic_distribution", "LR (dist)", (7, 2), "left", False),
        ("S_rbf_svm_median", "SVM (med)", (8, -2), "left", False),
        ("S_rbf_svm_mean", "SVM (mean)", (-14, -9), "right", True),
    ]
    for cfg, txt, offset, ha_val, use_arr in callouts:
        r = sf_idx.loc[cfg]
        ax.annotate(
            txt,
            (r["tray_equal_k1_ba"] * 100, r["group_ba_kAll"] * 100),
            xytext=offset,
            textcoords="offset points",
            ha=ha_val,
            fontsize=6.8,
            color="#222222",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.65, shrinkB=4) if use_arr else None,
        )
    ax.text(
        0.025, 0.37,
        "Unified tray-equal ref:\nSpearman ρ = −0.026 (3 ties)\n95% CI [−0.315, +0.613]",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=6.6,
        linespacing=1.28,
        bbox=dict(boxstyle="round,pad=0.28", fc="#F7F8FA", ec="#CBD5E0", lw=0.8),
    )
    ax.set_xlabel("Tray-equal single-kernel OOF BA (%)")
    ax.set_ylabel("Full-tray (All) group BA (%)")
    ax.set_title("Single-kernel vs. full-tray SWIR performance")
    ax.set_xlim(28.5, 65.5)
    ax.set_ylim(74.0, 99.6)
    ax.grid(alpha=0.25, ls="--")
    handles_clf = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLF_COLORS[k], ms=5.8, label=CLF_LABELS[k]) for k in CLF_COLORS]
    handles_rep = [Line2D([0], [0], marker=REP_MARKERS[k], color="w", markerfacecolor="#555555", ms=5.6, label=REP_LABELS[k]) for k in REP_MARKERS]
    leg1 = ax.legend(handles=handles_clf, loc="lower right", bbox_to_anchor=(0.99, 0.02),
                     frameon=True, framealpha=0.95, title="Classifier", title_fontsize=6.8, fontsize=6.4)
    ax.add_artist(leg1)
    ax.legend(handles=handles_rep, loc="lower right", bbox_to_anchor=(0.73, 0.02),
              frameon=True, framealpha=0.95, title="SWIR summary", title_fontsize=6.8, fontsize=6.4)
    panel_label(ax, "A")

    # Panel B: Unified tray-equal Spearman rho(k) with 2,000-draw cluster-bootstrap 95% CIs
    ax = axes[0, 1]
    ks_lbl = ["k1", "k3", "k5", "k10", "k20", "kAll"]
    kx = np.arange(len(ks_lbl))
    series = [
        ("SWIR_factorial_14", "SWIR only (14 configs; 3 All-ties)", "#0072B2", "o-", True),
        ("SWIR_fixed_mean421_4", "Fixed mean421 SWIR (4 classifiers)", "#009E73", "D:", False),
        ("MSD_jpeg_factorial_14", "MSD_jpeg (14 configs; 11 All-ties)", "#D55E00", "s--", True),
        ("MSD_raw_factorial_14", "MSD_raw (14 configs; 11 All-ties)", "#CC79A7", "^-.", False),
    ]
    for sub_name, lbl, col, sty, shade in series:
        sub = rk[rk["ensemble"].eq(sub_name)].set_index("endpoint").loc[ks_lbl]
        ax.plot(kx, sub["spearman_rho"], sty, color=col, lw=1.75, ms=5.0, label=lbl)
        if shade:
            ax.fill_between(kx, sub["rho_ci_low"], sub["rho_ci_high"], color=col, alpha=0.13, edgecolor="none")
    ax.axhline(0, color="#666666", lw=0.8, ls=":")
    ax.set_xticks(kx)
    ax.set_xticklabels(["1", "3", "5", "10", "20", "All"])
    ax.set_xlabel("Decision kernel count, k")
    ax.set_ylabel("Spearman ρ vs. tray-equal k=1 (95% CI)")
    ax.set_title("Scale-dependent ranking decoupling (unified weighting)")
    ax.set_ylim(-0.95, 1.12)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=6.6)
    panel_label(ax, "B")

    # Panel C (NEW): Protocol 1 — Fixed single-kernel classifiers under Late vs Early1 vs Early2 (Full tray, Primary)
    ax = axes[1, 0]
    p1 = ep_p[ep_p["protocol"].eq("fixed_single_kernel")].set_index(["classifier", "route"])
    clfs = ["logistic", "rbf_svm", "extra_trees", "cnn1d"]
    x = np.arange(len(clfs))
    w = 0.25
    routes_c = [
        ("late_prob_pool", "Late probability pool", "#0072B2"),
        ("early1_raw_pool_snv", "Early1: raw pool → SNV", "#009E73"),
        ("early2_snv_pool", "Early2: SNV → pool", "#D55E00"),
    ]
    for j, (rt, lbl, col) in enumerate(routes_c):
        vals = [p1.loc[(c, rt), "group_ba_kAll"] * 100 for c in clfs]
        bars = ax.bar(x + (j - 1) * w, vals, width=w, color=col, alpha=0.90, label=lbl)
        for bi, v in enumerate(vals):
            ax.text(x[bi] + (j - 1) * w, v + 1.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([CLF_LABELS[c] for c in clfs])
    ax.set_xlim(-0.52, 3.92)
    ax.set_xlabel("Fixed single-kernel classifier (mean421 SWIR)")
    ax.set_ylabel("Primary full-tray Group BA (%)")
    ax.set_title("Protocol 1: Aggregation position with fixed classifier")
    ax.set_ylim(0, 121)
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper center", ncol=3, frameon=True, framealpha=0.95, fontsize=6.2, columnspacing=0.9)
    ax.annotate("Training–inference\nscale mismatch\n(−44 to −49 pp)", xy=(3.12, 57.5), xytext=(3.42, 81.5),
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#B84A00", lw=0.9), fontsize=6.3, color="#B84A00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E0C0A8", alpha=0.96))
    panel_label(ax, "C")

    # Panel D (NEW): Protocol 2 — Scale-matched spectral training (Logistic & RBF-SVM) vs Late & ExactAll
    ax = axes[1, 1]
    ks_p2 = ["group_ba_k1", "group_ba_k3", "group_ba_k5", "group_ba_k10", "group_ba_k20", "group_ba_kAll"]
    x_pts = np.array([0, 1, 2, 3, 4, 5.35])
    p_idx = ep_p.set_index(["protocol", "classifier", "route"])
    a_mean = ep_a.groupby(["protocol", "classifier", "route"])["group_ba_kAll"].mean()

    curves_d = [
        (("fixed_single_kernel", "logistic", "late_prob_pool"), "LR: Late prob. pool (Aug 51.7%)", "#0072B2", "o--", 1.5),
        (("scale_matched_train_k20toAll", "logistic", "early1_raw_pool_snv"), "LR: Scale-matched Early1 (Aug 81.7%)", "#009E73", "s-", 1.9),
        (("scale_matched_train_exactAll", "logistic", "early1_raw_pool_snv"), "LR: Exact-All train (~50 trays; Aug 56.7%)", "#56B4E9", "^:", 1.3),
        (("fixed_single_kernel", "rbf_svm", "late_prob_pool"), "SVM: Late prob. pool (Aug 76.7%)", "#D55E00", "o--", 1.5),
        (("scale_matched_train_k20toAll", "rbf_svm", "early1_raw_pool_snv"), "SVM: Scale-matched Early1 (Aug 71.7%)", "#CC79A7", "D-", 1.8),
        (("scale_matched_train_exactAll", "rbf_svm", "early1_raw_pool_snv"), "SVM: Exact-All train (~50 trays; Aug 46.7%)", "#999999", "v:", 1.3),
    ]
    for key, lbl, col, sty, lw in curves_d:
        y = [p_idx.loc[key, c] * 100 for c in ks_p2]
        if "exactAll" in key[0]:
            # Only plot its All marker distinctly (k=1..20 overlap k20toAll)
            ax.scatter([x_pts[5]], [y[5]], color=col, marker=sty[0], s=48, zorder=6, label=lbl)
        else:
            ax.plot(x_pts[:5], y[:5], sty, color=col, lw=lw, ms=4.8, label=lbl)
            ax.scatter([x_pts[5]], [y[5]], color=col, marker=sty[0], s=46, zorder=5)
            ax.plot([x_pts[4], x_pts[5]], [y[4], y[5]], ":", color=col, lw=1.0, alpha=0.65)
    ax.axvline(4.68, color="#999999", ls="--", lw=0.8)
    ax.set_xticks(x_pts)
    ax.set_xticklabels(["1", "3", "5", "10", "20", "All"])
    ax.set_xlabel("Decision kernel count, k (Primary 67 trays; Aug 6-cond in legend)")
    ax.set_ylabel("Group balanced accuracy (%)")
    ax.set_title("Protocol 2: Scale-matched early spectral training")
    ax.set_ylim(54, 104.0)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower right", frameon=True, framealpha=0.93, fontsize=6.3)
    ax.annotate("LR k20→All: 94.9%\n(+6.1 pp vs late;\n+30.0 pp Aug)", xy=(5.32, 95.1), xytext=(2.85, 96.2),
                arrowprops=dict(arrowstyle="->", color="#009E73", lw=0.85), fontsize=6.5, color="#1C4532", fontweight="bold")
    panel_label(ax, "D")

    save_fig(fig, "Figure_3_scale_decoupling_and_pooling_stage")


def build_figure_4() -> None:
    """Figure 4: Label-conditioned counterfactual rival-decorrelation diagnostic,
    failure-regime transition, correct-vs-misclassified margin pathways, and
    exact additive multimodal margin shift.
    """
    pm = pd.read_csv(PBP / "stage2_diagnostics/primary_metrics.csv")
    cf = pd.read_csv(PBP / "stage5_strengthening/p0_counterfactual_margins.csv")
    comp = pd.read_csv(PBP / "stage2_diagnostics/multimodal_margin_complementarity_primary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    plt.subplots_adjust(wspace=0.34, hspace=0.36)

    # Panel A: Label-conditioned rival-slot permutation diagnostic
    ax = axes[0, 0]
    show_cf = [
        ("S_rbf_svm_median", "SVM (med)"),
        ("S_rbf_svm_mean", "SVM (mean)"),
        ("S_logistic_mean", "LR (mean)"),
        ("S_logistic_mean_std", "LR (m+SD)"),
        ("S_extra_trees_median", "ET (med)"),
        ("S_extra_trees_mean", "ET (mean)"),
        ("S_cnn1d_mean", "1D-CNN (mean)"),
    ]
    cf_idx = cf.set_index("configuration")
    y_pos = np.arange(len(show_cf))[::-1]
    for yi, (cfg, lbl) in zip(y_pos, show_cf):
        r = cf_idx.loc[cfg]
        o_ba = r["orig_full_tray_ba"] * 100
        c_ba = r["cf_rival_perm_ba_mean"] * 100
        ax.plot([o_ba, c_ba], [yi, yi], "-", color="#888888", lw=1.6, zorder=2)
        ax.scatter([o_ba], [yi], color="#0072B2", s=48, zorder=4, label="Original full-tray BA" if yi == y_pos[0] else None)
        ax.scatter([c_ba], [yi], color="#D55E00", marker="D", s=46, zorder=5, label="Rival-decorrelated diagnostic" if yi == y_pos[0] else None)
        ax.text(c_ba + 0.8, yi, f"+{c_ba - o_ba:.1f} pp", va="center", fontsize=7.0, color="#B84A00", fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[1] for s in show_cf])
    ax.set_xlabel("Full-tray balanced accuracy (%)")
    ax.set_title("Label-conditioned diagnostic: rival-slot permutation")
    ax.set_xlim(74, 108.5)
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.8)
    panel_label(ax, "A")

    # Panel B: Failure regime decomposition across k
    ax = axes[0, 1]
    sf = pm[pm["family"].eq("SWIR_factorial")]
    ks_s = ["1", "3", "5", "10", "20", "All"]
    var_e = [sf[f"err_variance_driven_k{k}"].mean() * 100 for k in ks_s]
    sys_e = [sf[f"err_systematic_k{k}"].mean() * 100 for k in ks_s]
    x = np.arange(len(ks_s))
    ax.bar(x, sys_e, width=0.58, color="#D55E00", label=r"Systematic negative-margin ($\mu^* \leq 0$)")
    ax.bar(x, var_e, width=0.58, bottom=sys_e, color="#56B4E9", label=r"Variance-driven ($\mu^* > 0$, pooling-correctable)")
    for i, (se, ve) in enumerate(zip(sys_e, var_e)):
        tot = se + ve
        pct_sys = 100.0 * se / tot
        ax.text(i, tot + 1.1, f"{tot:.1f}%\n({pct_sys:.0f}% sys)", ha="center", va="bottom", fontsize=6.9, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(ks_s)
    ax.set_xlabel("Decision kernel count, k")
    ax.set_ylabel("Mean balanced error rate (%)")
    ax.set_title("Variance-to-systematic error transition (14 SWIR)")
    ax.set_ylim(0, 62)
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    panel_label(ax, "B")

    # Panel C: Correct- vs misclassified-kernel worst-rival margin contributions
    ax = axes[1, 0]
    merged = sf.merge(cf[["configuration", "mean_pos_kernel_contrib", "mean_wrong_kernel_drag"]], on="configuration")
    # Plot larger markers first so smaller markers (e.g., SVM median behind ET median) remain visible on top
    merged = merged.sort_values("group_ba_kAll", ascending=False)
    for idx_m, (_, r) in enumerate(merged.iterrows()):
        ax.scatter(
            r["mean_pos_kernel_contrib"], -r["mean_wrong_kernel_drag"],
            s=30 + (r["group_ba_kAll"] - 0.75) * 320,
            color=CLF_COLORS[r["classifier"]], marker=REP_MARKERS[r["representation"]],
            edgecolor="white", lw=0.65, alpha=0.94, zorder=4 + idx_m * 0.05,
        )
    m_idx = merged.set_index("configuration")
    callouts_4c = [
        ("S_cnn1d_mean", "1D-CNN (mean)\n95.3% All (μ*=0.332)", (0.268, 0.084), "left", "center"),
        ("S_logistic_mean_std", "LR (m+SD)\n89.8% All", (0.315, 0.036), "left", "center"),
        ("S_extra_trees_mean", "ET (mean): 93.8% All\n(low misclass. drag)", (0.134, 0.079), "left", "center"),
        ("S_rbf_svm_median", "SVM (med)\n77.5% All", (0.035, 0.004), "left", "bottom"),
        ("S_rbf_svm_mean_std", "SVM (m+SD)\n89.7% All", (0.228, 0.009), "left", "center"),
    ]
    for cfg, lbl, xy_txt, ha_v, va_v in callouts_4c:
        r = m_idx.loc[cfg]
        ax.annotate(
            lbl,
            xy=(r["mean_pos_kernel_contrib"], -r["mean_wrong_kernel_drag"]),
            xytext=xy_txt,
            ha=ha_v,
            va=va_v,
            fontsize=6.6,
            color="#222222",
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.75, shrinkA=2, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.86),
        )
    ax.text(0.98, 0.03, "Marker size ∝ full-tray BA", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.4, style="italic", color="#555555")
    ax.set_xlabel(r"Correctly classified kernel contribution $\mu_{\mathrm{corr}}^*$")
    ax.set_ylabel(r"Misclassified kernel margin drag $|\mu_{\mathrm{mis}}^*|$")
    ax.set_title("Two margin pathways to high full-tray accuracy")
    ax.set_xlim(0.015, 0.46)
    ax.set_ylim(-0.001, 0.095)
    ax.grid(alpha=0.25, ls="--")
    h_clf = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLF_COLORS[k], ms=5.8, label=CLF_LABELS[k]) for k in CLF_COLORS]
    ax.legend(handles=h_clf, loc="upper left", frameon=True, framealpha=0.94, fontsize=6.7)
    panel_label(ax, "C")

    # Panel D: Exact additive multimodal margin shift Δd_c across 67 primary trays
    ax = axes[1, 1]
    show_schemes = [
        ("MSD_raw__S_logistic_median", "LR (median)", "#0072B2"),
        ("MSD_raw__S_logistic_mean_std", "LR (mean+SD)", "#56B4E9"),
        ("MSD_raw__S_extra_trees_mean_std", "ET (mean+SD)", "#009E73"),
        ("MSD_raw__S_cnn1d_mean", "1D-CNN (mean)", "#CC79A7"),
    ]
    for i, (sch, lbl, col) in enumerate(show_schemes):
        g = comp[comp["scheme"].eq(sch)]
        vals = g["delta_d_on_md_rival"].to_numpy()
        rng = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ok = g["outcome"].eq("both_correct").to_numpy()
        res = g["outcome"].eq("rescued").to_numpy()
        bw = g["outcome"].eq("both_wrong").to_numpy()
        ax.scatter(np.full(ok.sum(), i) + jitter[ok], vals[ok], s=14, color=col, alpha=0.55, edgecolors="none")
        if res.any():
            ax.scatter(np.full(res.sum(), i) + jitter[res], vals[res], s=55, marker="*", color="#D55E00", edgecolor="black", lw=0.4, zorder=6, label="Rescued tray" if i == 0 else None)
        if bw.any():
            ax.scatter(np.full(bw.sum(), i) + jitter[bw], vals[bw], s=32, marker="X", color="#222222", zorder=6, label="Unresolved error" if i == 0 else None)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals), np.median(vals)], color="black", lw=1.8, zorder=5)
    ax.axhline(0, color="#555555", ls="--", lw=0.9)
    ax.set_xticks(range(len(show_schemes)))
    ax.set_xticklabels([s[1] for s in show_schemes])
    ax.set_ylabel(r"SWIR margin shift $\Delta d_{c^*_{\mathrm{MD}}} = \frac{1}{3}(\mu_{S} - \mu_{\mathrm{MD}})$")
    ax.set_title("Exact additive SWIR margin shift (67 trays)")
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    panel_label(ax, "D")

    save_fig(fig, "Figure_4_margin_diagnostics")


def build_figure_5() -> None:
    """Figure 5: Seven pooling and nested calibration strategies across Primary
    and August cohorts, including scale-matched post-pooling transfer.
    """
    p3 = pd.read_csv(PBP / "stage3_calibration/primary_pooling_comparison.csv")
    temps = pd.read_csv(PBP / "stage3_calibration/fitted_temperatures.csv")
    p1 = pd.read_csv(PBP / "stage5_strengthening/p1_post_pooling_transfer.csv")

    FACTORIAL_SWIR = [
        f"S_{clf}_{rep}"
        for clf in ("logistic", "rbf_svm", "extra_trees")
        for rep in ("median", "mean", "mean_std", "distribution")
    ] + ["S_cnn1d_median", "S_cnn1d_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    plt.subplots_adjust(wspace=0.30, hspace=0.38)

    strat_meta = [
        ("arith_raw", "Arithmetic mean (raw)", "#0072B2", "o-"),
        ("cal_kernel_pre", r"Pre-pool $T_{\mathrm{kernel}}^*$", "#009E73", "s--"),
        ("cal_group_pre", r"Pre-pool $T_{\mathrm{group}}^*$", "#CC79A7", "^-."),
        ("log_prob_pool", "Log-probability pool", "#E69F00", "D:"),
        ("majority_vote", "Majority voting", "#D55E00", "v-"),
    ]

    ax = axes[0, 0]
    p3_s = p3[p3["family"].eq("SWIR_factorial")]
    agg = p3_s.groupby("strategy")[["group_ba_k1", "group_ba_k3", "group_ba_k5", "group_ba_k10", "group_ba_k20", "group_ba_kAll"]].mean()
    base = agg.loc["arith_raw"]
    kx = np.arange(6)
    for st, lbl, col, sty in strat_meta[1:]:
        diff = (agg.loc[st] - base).to_numpy() * 100
        ax.plot(kx, diff, sty, color=col, lw=1.8, ms=5.5, label=lbl)
    ax.axhline(0, color="#333333", lw=0.9, ls="--")
    ax.set_xticks(kx)
    ax.set_xticklabels(["1", "3", "5", "10", "20", "All"])
    ax.set_xlabel("Decision kernel count, k")
    ax.set_ylabel("Mean Δ Group BA vs. arithmetic raw (pp)")
    ax.set_title("Pooling & calibration impact across k (14 SWIR)")
    ax.set_ylim(-4.5, 2.3)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.94, fontsize=6.8)
    ax.annotate("Log-pool scale crossover:\n+1.38 pp (k=3) → −2.08 pp (All)", xy=(1, 1.38), xytext=(1.7, 1.45),
                arrowprops=dict(arrowstyle="->", color="#E69F00", lw=0.8), fontsize=7.2, color="#B37400", fontweight="bold")
    panel_label(ax, "A")

    ax = axes[0, 1]
    t4 = pd.read_csv(PBP / "stage7_final_polish/table4_complete_all7.csv")
    agg_all = (
        t4[(t4["cohort"] == "primary") & (t4["k"] == "All")]
        .groupby("strategy")[["group_ba", "group_brier", "group_ece"]]
        .mean()
    )
    all_strats = strat_meta + [("post_cal_AllT", r"Post-pool $T_{\mathrm{post}}(\mathrm{All})$", "#222222", "*")]
    offsets = {
        "arith_raw": (10, 7, "left"),
        "cal_kernel_pre": (-8, -25, "left"),
        "cal_group_pre": (10, -4, "left"),
        "log_prob_pool": (-10, -14, "right"),
        "majority_vote": (10, -4, "left"),
        "post_cal_AllT": (6, -26, "right"),
    }
    for st, lbl, col, sty in all_strats:
        r = agg_all.loc[st]
        mk = "*" if st == "post_cal_AllT" else sty[0]
        sz = 135 if st == "post_cal_AllT" else 65
        ax.scatter(r["group_brier"], r["group_ba"] * 100, color=col, marker=mk, s=sz, edgecolor="white", lw=0.5, zorder=5)
        dx, dy, ha = offsets[st]
        ax.annotate(f"{lbl}\n(ECE={r['group_ece']:.3f})", (r["group_brier"], r["group_ba"] * 100),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=6.8,
                    color=col if col != "#E69F00" else "#A66A00")
    ax.set_xlabel("Full-tray Group Brier score (lower = better)")
    ax.set_ylabel("Full-tray Group BA (%) (higher = better)")
    ax.set_title("Full-tray accuracy vs. probability reliability (14 SWIR)")
    ax.set_xlim(0.08, 0.62)
    ax.set_ylim(87.0, 91.8)
    ax.invert_xaxis()
    ax.grid(alpha=0.25, ls="--")
    ax.text(0.04, 0.08, "← Higher Brier    Better reliability →", transform=ax.transAxes, fontsize=7.2, color="#555555")
    panel_label(ax, "B")

    ax = axes[1, 0]
    show_cfgs = [
        ("M_raw", "M_raw"),
        ("S_logistic_median", "LR (med)"),
        ("S_logistic_mean_std", "LR (m+SD)"),
        ("S_rbf_svm_mean_std", "SVM (m+SD)"),
        ("S_extra_trees_mean_std", "ET (m+SD)"),
        ("S_cnn1d_mean", "CNN (mean)"),
    ]
    ts = temps.groupby("config_id")[["T_kernel", "T_group", "T_post"]].agg(["mean", "std"])
    x = np.arange(len(show_cfgs))
    w = 0.26
    for j, (col_n, lbl, col) in enumerate([("T_kernel", r"$T_{\mathrm{kernel}}^*$", "#009E73"),
                                            ("T_group", r"$T_{\mathrm{group}}^*$", "#CC79A7"),
                                            ("T_post", r"$T_{\mathrm{post}}(\mathrm{All})$", "#222222")]):
        vals = [ts.loc[c[0], (col_n, "mean")] for c in show_cfgs]
        errs = [ts.loc[c[0], (col_n, "std")] for c in show_cfgs]
        ax.bar(x + (j - 1) * w, vals, width=w, yerr=errs, capsize=2, color=col, alpha=0.88, label=lbl)
    ax.axhline(1.0, color="#666666", ls="--", lw=0.9, label="Uncalibrated (T = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in show_cfgs], rotation=15)
    ax.set_ylabel("Nested fitted temperature T (±1 fold SD)")
    ax.set_title(r"Objective divergence: $T_{\mathrm{kernel}}^*$ vs. $T_{\mathrm{group}}^*$")
    ax.set_ylim(0, 2.35)
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=6.8)
    panel_label(ax, "C")

    ax = axes[1, 1]
    p1_s = p1[p1["config_id"].isin(FACTORIAL_SWIR)]
    groups = [
        ("Primary\n(k=3)", p1_s[(p1_s["cohort"] == "primary") & (p1_s["k"] == "3")]),
        ("Primary\n(All)", p1_s[(p1_s["cohort"] == "primary") & (p1_s["k"] == "All")]),
        ("Aug 6-cond\n(k=3)", p1_s[(p1_s["cohort"] == "august") & (p1_s["k"] == "3")]),
        ("Aug 6-cond\n(All)", p1_s[(p1_s["cohort"] == "august") & (p1_s["k"] == "All")]),
        ("Aug BASE\n(All)", p1_s[(p1_s["cohort"] == "august") & (p1_s["condition"] == "BASE") & (p1_s["k"] == "All")]),
    ]
    x = np.arange(len(groups))
    w = 0.25
    strats_d = [
        ("arith_raw", "Uncalibrated raw", "#7F7F7F"),
        ("post_cal_AllT", r"Post-pool $T_{\mathrm{post}}(\mathrm{All})$", "#D55E00"),
        ("post_cal_matched_kT", r"Scale-matched $T_{\mathrm{post}}(k)$", "#0072B2"),
    ]
    for j, (st, lbl, col) in enumerate(strats_d):
        vals_brier = [g[g["strategy"] == st]["group_brier"].mean() for _, g in groups]
        ax.bar(x + (j - 1) * w, vals_brier, width=w, color=col, alpha=0.9, label=lbl)
    for i, (_, g) in enumerate(groups):
        nll_raw = g[g["strategy"] == "arith_raw"]["group_nll"].mean()
        nll_mt = g[g["strategy"] == "post_cal_matched_kT"]["group_nll"].mean()
        nll_all = g[g["strategy"] == "post_cal_AllT"]["group_nll"].mean()
        top_y = max(g[g["strategy"].isin(["arith_raw", "post_cal_AllT", "post_cal_matched_kT"])].groupby("strategy")["group_brier"].mean())
        if i in (0, 2):
            ax.text(i, top_y + 0.025, f"NLL\n{nll_all:.2f}→{nll_mt:.2f}", ha="center", va="bottom", fontsize=6.0, color="#005B8F", fontweight="bold", linespacing=1.1)
        elif i == 4:
            ax.text(i, top_y + 0.025, f"NLL\n{nll_raw:.2f}→{nll_mt:.2f}*", ha="center", va="bottom", fontsize=6.0, color="#B84A00", fontweight="bold", linespacing=1.1)
        else:
            ax.text(i, top_y + 0.025, f"NLL\n{nll_raw:.2f}→{nll_mt:.2f}", ha="center", va="bottom", fontsize=6.0, color="#333333", linespacing=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.3)
    ax.set_ylabel("Mean Group Brier score (14 SWIR; lower = better)")
    ax.set_title("Scale-matched post-pooling calibration transfer")
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.7)
    panel_label(ax, "D")

    save_fig(fig, "Figure_5_calibration_and_pooling")


def build_figure_6() -> None:
    """Figure 6: Controlled SNV vs. non-SNV perturbation ablation, unified RGB
    comparator suite (M_raw, D, MD_raw), and six-condition RGB-SWIR robustness.
    """
    snv_a = pd.read_csv(S6 / "p1_snv_ablation_august.csv")
    pm = pd.read_csv(PBP / "stage2_diagnostics/primary_metrics.csv").set_index("configuration")
    ec = pd.read_csv(PBP / "stage2_diagnostics/external_6cond_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    plt.subplots_adjust(wspace=0.38, hspace=0.38)

    # Panel A (NEW): Controlled SNV vs non-SNV SWIR-only ablation across key August conditions
    ax = axes[0, 0]
    sa_swir = snv_a[snv_a["mode"].eq("SWIR_only")]
    # Plot mean across 4 classifiers (bars) + individual classifier markers for BASE, FLIP, LEFT_OFF, RIGHT_OFF!
    show_conds = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
    x = np.arange(len(show_conds))
    w = 0.34
    mean_snv = [sa_swir[(sa_swir["preprocessing"] == "SNV") & (sa_swir["condition"] == c)]["group_ba_kAll"].mean() * 100 for c in show_conds]
    mean_raw = [sa_swir[(sa_swir["preprocessing"] == "non_SNV") & (sa_swir["condition"] == c)]["group_ba_kAll"].mean() * 100 for c in show_conds]
    ax.bar(x - w / 2, mean_raw, width=w, color="#A0AEC0", edgecolor="#4A5568", lw=0.7, label="non-SNV SWIR (4-clf mean)")
    ax.bar(x + w / 2, mean_snv, width=w, color="#0072B2", alpha=0.88, edgecolor="#1A365D", lw=0.7, label="SNV SWIR (4-clf mean)")
    # Overlay individual classifier markers
    clfs = ["logistic", "rbf_svm", "extra_trees", "cnn1d"]
    offsets_c = [-0.075, -0.025, 0.025, 0.075]
    for ci, clf in enumerate(clfs):
        y_r = [sa_swir[(sa_swir["classifier"] == clf) & (sa_swir["preprocessing"] == "non_SNV") & (sa_swir["condition"] == c)]["group_ba_kAll"].iloc[0] * 100 for c in show_conds]
        y_s = [sa_swir[(sa_swir["classifier"] == clf) & (sa_swir["preprocessing"] == "SNV") & (sa_swir["condition"] == c)]["group_ba_kAll"].iloc[0] * 100 for c in show_conds]
        ax.scatter(x - w / 2 + offsets_c[ci], y_r, s=22, color=CLF_COLORS[clf], edgecolor="white", lw=0.4, zorder=5)
        ax.scatter(x + w / 2 + offsets_c[ci], y_s, s=22, color=CLF_COLORS[clf], edgecolor="white", lw=0.4, zorder=5, label=CLF_LABELS[clf])
    ax.annotate("Lamp-loss drop\nwithout SNV\n(0–3/10 trays)", xy=(4 - w / 2, 14.0), xytext=(4 - w / 2, 108),
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#B84A00", lw=0.9), fontsize=6.4, color="#B84A00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#E0C0A8", alpha=0.96))
    ax.set_xticks(x)
    ax.set_xticklabels(["BASE", "REP", "REPOS", "FLIP", "LEFT\nOFF", "RIGHT\nOFF"])
    ax.set_xlabel("Paired August imaging condition (mean421 SWIR)")
    ax.set_ylabel("Full-tray accuracy (%)")
    ax.set_title("SNV vs. non-SNV ablation across perturbations")
    ax.set_ylim(0, 136)
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, fontsize=5.7, columnspacing=0.6, handlelength=1.4)
    panel_label(ax, "A")

    # Panel B: Unified RGB comparators (M_raw, D, MD_raw, MD_jpeg) vs MSD_raw across k (Primary)
    ax = axes[0, 1]
    ks_cols = ["group_ba_k1", "group_ba_k3", "group_ba_k5", "group_ba_k10", "group_ba_k20", "group_ba_kAll"]
    x_pts = np.array([0, 1, 2, 3, 4, 5.4])
    curves_b = [
        ("RGB_M_raw", "M_raw → 67/67", "#56B4E9", "s:"),
        ("RGB_D_phys", "D (JPEG EffNet) → 61/67", "#999999", "^--"),
        ("RGB_MD_raw", "MD_raw (M_raw + D) → 65/67", "#0072B2", "o-"),
        ("MSD_raw__S_logistic_mean_std", "MSD_raw + LR (m+SD) → 66/67", "#D55E00", "D-"),
        ("MSD_raw__S_rbf_svm_mean_std", "MSD_raw + SVM (m+SD) → 65/67", "#E69F00", "v-"),
        ("MSD_raw__S_cnn1d_mean", "MSD_raw + 1D-CNN (mean) → 66/67", "#CC79A7", "P-"),
    ]
    for cfg, lbl, col, sty in curves_b:
        y = [pm.loc[cfg, c] * 100 for c in ks_cols]
        ax.plot(x_pts[:5], y[:5], sty, color=col, lw=1.65, ms=4.8, label=lbl)
        ax.scatter([x_pts[5]], [y[5]], color=col, marker=sty[0], s=44, zorder=5)
        ax.plot([x_pts[4], x_pts[5]], [y[4], y[5]], ":", color=col, lw=0.9, alpha=0.6)
    ax.axvline(4.7, color="#999999", ls="--", lw=0.8)
    ax.set_xticks(x_pts)
    ax.set_xticklabels(["1", "3", "5", "10", "20", "All"])
    ax.set_xlabel("Decision kernel count, k (Primary: 67 trays)")
    ax.set_ylabel("Group balanced accuracy (%)")
    ax.set_title("Primary: RGB comparators vs. RGB–SWIR across k")
    ax.set_ylim(44, 105)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower right", frameon=True, framealpha=0.93, fontsize=6.5)
    ax.annotate("+10.3 to +12.1 pp\nSWIR gain at k=3", xy=(1.0, 90.0), xytext=(0.05, 96.5),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.85), fontsize=6.8, color="#B84A00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E0C0A8", alpha=0.93))
    panel_label(ax, "B")

    # Panel C: 6-condition full-tray heatmap including M_raw, D, MD_raw, non-SNV vs SNV MSD_raw
    ax = axes[1, 0]
    conds = ["BASE", "REP", "REPOSITION", "FLIP", "LEFT_OFF", "RIGHT_OFF"]
    # Assemble rows from ec and snv_a (MSD_raw non-SNV!)
    ec_piv = ec.pivot(index="configuration", columns="condition", values="group_ba_kAll")[conds] * 100
    sa_msd_nosnv = snv_a[(snv_a["mode"] == "MSD_raw") & (snv_a["preprocessing"] == "non_SNV")].pivot(
        index="classifier", columns="condition", values="group_ba_kAll"
    )[conds] * 100

    hm_rows = [
        ("M_raw", ec_piv.loc["RGB_M_raw"].to_numpy()),
        ("D (JPEG EfficientNet)", ec_piv.loc["RGB_D_phys"].to_numpy()),
        ("MD_jpeg (M_jpeg + D)", ec_piv.loc["RGB_MD_jpeg"].to_numpy()),
        ("MD_raw (M_raw + D)", ec_piv.loc["RGB_MD_raw"].to_numpy()),
        ("SWIR: 1D-CNN (SNV)", ec_piv.loc["S_cnn1d_mean"].to_numpy()),
        ("SWIR: RBF-SVM (SNV m+SD)", ec_piv.loc["S_rbf_svm_mean_std"].to_numpy()),
        ("MSD_raw + LR (SNV m+SD)", ec_piv.loc["MSD_raw__S_logistic_mean_std"].to_numpy()),
        ("MSD_raw + CNN (non-SNV)", sa_msd_nosnv.loc["cnn1d"].to_numpy()),
        ("MSD_raw + ET (SNV m+SD)", ec_piv.loc["MSD_raw__S_extra_trees_mean_std"].to_numpy()),
        ("MSD_raw + SVM (SNV m+SD)", ec_piv.loc["MSD_raw__S_rbf_svm_mean_std"].to_numpy()),
        ("MSD_raw + CNN (SNV mean)", ec_piv.loc["MSD_raw__S_cnn1d_mean"].to_numpy()),
    ]
    mat = np.vstack([r[1] for r in hm_rows])
    mat_full = np.hstack([mat, mat.mean(axis=1, keepdims=True)])
    ax.imshow(mat_full, cmap="YlGnBu", vmin=10, vmax=100, aspect="auto")
    for r_i in range(mat_full.shape[0]):
        for c_i in range(mat_full.shape[1]):
            val = mat_full[r_i, c_i]
            txt = f"{val:.1f}" if c_i == 6 else f"{int(round(val/10))}/10"
            ax.text(c_i, r_i, txt, ha="center", va="center", fontsize=6.6,
                    color="white" if val >= 65 else "#111111", fontweight="bold" if c_i == 6 else "normal")
    ax.axvline(5.5, color="white", lw=2.0)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["BASE", "REP", "REPOS", "FLIP", "LEFT\nOFF", "RIGHT\nOFF", "Mean\n(%)"], fontsize=7.1)
    ax.set_yticks(range(len(hm_rows)))
    ax.set_yticklabels([r[0] for r in hm_rows], fontsize=7.0)
    ax.set_title("Full-tray accuracy across 6 paired imaging conditions")
    panel_label(ax, "C")

    # Panel D: Kernel-level perturbation trajectories (RGB comparators + SNV vs non-SNV 1D-CNN & MSD_raw)
    ax = axes[1, 1]
    ec_k = ec.pivot(index="configuration", columns="condition", values="kernel_ba")[conds] * 100
    sa_k = snv_a.pivot(index=["mode", "classifier", "preprocessing"], columns="condition", values="kernel_ba")[conds] * 100
    x = np.arange(len(conds))
    traj = [
        ("M_raw", ec_k.loc["RGB_M_raw"].to_numpy(), "#56B4E9", "s--"),
        ("D (JPEG EffNet)", ec_k.loc["RGB_D_phys"].to_numpy(), "#8C8C8C", "^:"),
        ("MD_raw (M_raw + D)", ec_k.loc["RGB_MD_raw"].to_numpy(), "#0072B2", "o-"),
        ("SWIR CNN (non-SNV)", sa_k.loc[("SWIR_only", "cnn1d", "non_SNV")].to_numpy(), "#E69F00", "v--"),
        ("SWIR CNN (SNV)", sa_k.loc[("SWIR_only", "cnn1d", "SNV")].to_numpy(), "#009E73", "D-."),
        ("MSD_raw + CNN (non-SNV)", sa_k.loc[("MSD_raw", "cnn1d", "non_SNV")].to_numpy(), "#D55E00", "x--"),
        ("MSD_raw + CNN (SNV)", sa_k.loc[("MSD_raw", "cnn1d", "SNV")].to_numpy(), "#CC79A7", "P-"),
    ]
    for lbl, vals, col, sty in traj:
        ax.plot(x, vals, sty, color=col, lw=1.6, ms=4.8, label=lbl)
    ax.axvspan(-0.35, 1.35, color="#EEEEEE", alpha=0.6, zorder=0)
    ax.text(0.5, 64.5, "Baseline / repeat ref.", ha="center", fontsize=6.5, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(["BASE", "REP", "REPOS", "FLIP", "LEFT\nOFF", "RIGHT\nOFF"])
    ax.set_xlabel("Paired imaging condition (280 physical kernels)")
    ax.set_ylabel("Single-kernel balanced accuracy (%)")
    ax.set_title("Kernel-level robustness: SNV invariance under lamp loss")
    ax.set_ylim(2, 69)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.94, fontsize=6.1, columnspacing=0.8)
    panel_label(ax, "D")

    save_fig(fig, "Figure_6_snv_ablation_and_robustness")


def main() -> None:
    # Remove legacy Figure_1..5 stems that were renamed so figures/ stays spotless
    for old_stem in [
        "Figure_1_optical_characterization",
        "Figure_2_factorial_scale_decoupling",
        "Figure_3_margin_mechanism",
        "Figure_4_calibration_and_pooling",
        "Figure_5_raw_multimodal_robustness",
    ]:
        for ext in (".pdf", ".png"):
            p = FIG_DIR / f"{old_stem}{ext}"
            if p.exists():
                p.unlink()

    build_figure_1()
    build_figure_2()
    build_figure_3()
    build_figure_4()
    build_figure_5()
    build_figure_6()
    print("Rebuilt all 6 IPT figures (PDF + 600-dpi PNG) in", FIG_DIR)


if __name__ == "__main__":
    main()
