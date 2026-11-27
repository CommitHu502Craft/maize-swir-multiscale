# Multi-scale aggregation of SWIR hyperspectral information for waxy-maize cultivar identification: spectral representation, decision pooling, and robustness

Companion code, nested evaluation schedules, and archived summary results for the manuscript submitted to ***Infrared Physics & Technology***.

## Repository Contents

| Directory / File | Description | Count / Scope |
| :--- | :--- | :--- |
| [`scripts/`](scripts/) | End-to-end modeling, training, evaluation, calibration, optical perturbation, and figure-generation scripts (`00`–`08`, plus plotting modules) | 12 scripts (`9` Python pipeline stages + `3` figure builders) |
| [`results/`](results/) | Complete archived CSV and JSON summary outputs backing Main Tables 1–5, Supplementary Tables S1–S12, and Figures 1–6 | 48 files across `stage1`–`stage8` and `audit_manifest.json` |
| [`schedules/`](schedules/) | Shared 500 nested kernel permutation schedules across the 67 primary trays (`batch2_schedules.npz`) and 2,000 variety-stratified whole-tray bootstrap weights (`batch2_bootstrap.npz`) | 2 files (~3.5 MB total) |
| [`LICENSE`](LICENSE) | MIT license for code/scripts and CC BY 4.0 license for summary CSV/JSON outputs and documentation | — |

---

## Companion Dataset and Reproducibility Scope

- **Companion Dataset:** The paired RGB–SWIR dataset has been deposited in Zenodo under reserved [**DOI 10.5281/zenodo.23567396**](https://doi.org/10.5281/zenodo.23567396) and will become publicly accessible upon publication (private peer-review access is provided directly to editors and reviewers in the accompanying cover letter).
- **What Runs Directly Here:** Inspecting all archived numerical tables (including complete un-truncated Supplementary Tables S7 and S9) and redrawing Figures 3–6 directly from [`results/`](results/) (see below).
- **Full End-to-End Reruns (`00`–`08`):** Scripts [`00_prepare_and_audit.py`](scripts/00_prepare_and_audit.py)–[`08_final_technical_verification.py`](scripts/08_final_technical_verification.py) contain the complete Python implementations of factorial model training (Logistic, RBF-SVM, ExtraTrees, 1D-CNN), Protocol 1 and Protocol 2 scale-matched early pooling, controlled SNV ablation, unified RGB comparators (`M_raw`, `D`, `MD_raw`, `MSD_raw`), wavelength-resolved perturbation spectroscopy, counterfactual rival-slot permutation, margin decomposition, and nested temperature scaling. Full end-to-end retraining requires the raw hyperspectral/RGB/mask data matrices (hosted on Zenodo upon publication; excluded from GitHub due to file-size limits) and a Python environment with NumPy, pandas, SciPy, scikit-learn, OpenCV, joblib, Matplotlib, Pillow, and PyTorch.

---

## Script Pipeline (`scripts/`)

| Stage / Module | Script | Function |
| :--- | :--- | :--- |
| Stage 00 | [`00_prepare_and_audit.py`](scripts/00_prepare_and_audit.py) | Prepare spectral representations (`median`, `mean`, `mean+SD`, `distribution`), enforce tray-grouped splits without leakage, and align 500 shared schedules. |
| Stage 01 | [`01_run_factorial_models.py`](scripts/01_run_factorial_models.py) | Fit the 14 single-modality SWIR configurations across 4 outer tray-grouped folds and 3 nested inner folds (producing inner-OOF and outer-OOF scores). |
| Stage 02 | [`02_margin_and_scale_diagnostics.py`](scripts/02_margin_and_scale_diagnostics.py) | Compute scale-decoupling curves, worst-rival tray margin decomposition, and additive RGB–SWIR multimodal complementarity (`MSD_raw` vs. `MD_raw`). |
| Stage 03 | [`03_calibration_and_pooling.py`](scripts/03_calibration_and_pooling.py) | Fit nested temperature scaling (`T_kernel`, `T_group`, `T_post`) and compare pooling rules (`arith_raw`, `majority_vote`, `log_prob_pool`). |
| Stage 04 | [`04_optical_and_gate_report.py`](scripts/04_optical_and_gate_report.py) | Quantify cultivar reflectance, within-kernel spatial SD (`~9×` tray SD), pixel subsampling curves, and six-condition perturbation statistics. |
| Stage 05 | [`05_p0_p1_strengthening.py`](scripts/05_p0_p1_strengthening.py) | Run 2,000-draw cluster bootstraps, rival-slot permutations, 7-temperature sharpness sweeps (`Table S9`), and affine-residual perturbation RMSE/SAM checks. |
| Stage 06 | [`06_ipt_signal_processing_revision.py`](scripts/06_ipt_signal_processing_revision.py) | Compute unified tray-equal rankings, Protocol 1 & Protocol 2 (`early1`, `early2`), controlled `SNV` vs. `non-SNV` ablation, uniform `M_raw` comparators, and 421-band affine residuals. |
| Stage 07 | [`07_ipt_final_polish_checks.py`](scripts/07_ipt_final_polish_checks.py) | Build complete 7-strategy calibration (`Table 4`), entity-tray paired uncertainty (`Table S8`), PCA variance contraction (`Table S5`), `early3`, and subset-count sensitivity (`Table S6`). |
| Stage 08 | [`08_final_technical_verification.py`](scripts/08_final_technical_verification.py) | Audit 127-tray orientation NCC, run Protocol 2 controlled PCA/weight/$C$ factorizations (`Table S7`), and verify the exact `early1` SNV algebra. |
| Figures 2–6 | [`build_figures.py`](scripts/build_figures.py) | Generate publication-grade PDF and 600-dpi PNG figures (`Figure 5B` plots unsmoothed Group Brier score; `Figure 6D` labels baseline reference band). |
| Figure 1 | [`build_figure1_svg.py`](scripts/build_figure1_svg.py), [`export_figure1.cjs`](scripts/export_figure1.cjs) | Assemble schematic + SWIR spectral inset SVG and export vector PDF / 600-dpi PNG via headless Chromium/Edge. |

---

## Manuscript Table & Figure Provenance (`results/`)

| Manuscript Item | Description | Source File(s) in `results/` |
| :--- | :--- | :--- |
| **Table 1** | Cohorts and paired SWIR perturbation spectroscopy | `audit_manifest.json`; `stage4_optical/cultivar_optical_heterogeneity.csv`; `stage5_strengthening/p0_optical_perturbation_physics.csv`; `stage6_ipt_revision/p1_wavelength_affine_summary.csv` |
| **Table 2** | Representative SWIR scale decoupling & rival permutation | `stage2_diagnostics/primary_metrics.csv`; `stage5_strengthening/p0_counterfactual_margins.csv`; `stage6_ipt_revision/p0_swir14_unified_summary.csv` |
| **Table 3** | Spectral early pooling vs. probability late pooling (`mean421`) | `stage6_ipt_revision/p1_early_vs_late_primary.csv`, `p1_early_vs_late_august.csv`, `p1_early_vs_late_contrasts.csv` |
| **Table 4** | Seven pooling & calibration strategies (Primary + August) | `stage7_final_polish/table4_complete_all7.csv`; `stage3_calibration/primary_pooling_comparison.csv`; `stage5_strengthening/p1_post_pooling_transfer.csv` |
| **Table 5** | Controlled SNV ablation & visible comparators vs. `MSD_raw` | `stage6_ipt_revision/p1_snv_ablation_primary.csv`, `p1_snv_ablation_august.csv`, `p1_snv_ablation_contrasts.csv`, `p1_unified_rgb_multimodal_summary.csv` |
| **Table S1** | 12-cultivar composition, fold splits & optical spread | `audit_manifest.json`; `stage4_optical/cultivar_optical_heterogeneity.csv` |
| **Table S2** | 5-window wavelength-dependent affine residuals | `stage6_ipt_revision/p1_wavelength_affine_summary.csv`, `p1_wavelength_affine_residuals.csv` |
| **Table S3** | Complete 14-configuration SWIR factorial matrix | `stage1_summary.csv`; `stage2_diagnostics/primary_metrics.csv`; `stage6_ipt_revision/p0_swir14_unified_summary.csv` |
| **Table S4** | 2,000-draw cluster-bootstrap Spearman $\rho$ & Kendall $\tau$ | `stage6_ipt_revision/p0_unified_tray_equal_rankings.csv` |
| **Table S5** | Early-pooling PCA-60 contraction & `early3` control | `stage7_final_polish/early_pooling_pca_shift.csv`, `early3_resnv_control.csv` |
| **Table S6** | Subset-count sensitivity ($M \in \{1,5,10,25,50\}$) | `stage7_final_polish/subset_count_sensitivity_logistic_k20.csv` |
| **Table S7** *(full 40 rows)* | Protocol 2 PCA-rank, tray-weight ($1/M$) & wide-$C$ controls | [`stage8_final_verification/protocol2_pca_weight_c_controls.csv`](results/stage8_final_verification/protocol2_pca_weight_c_controls.csv); `protocol2_subset_audit.json` |
| **Table S8** | Entity-tray paired uncertainty ($n=10$ trays) | `stage7_final_polish/august_entity_tray_paired_uncertainty.csv`, `august_per_entity_tray_6cond.csv` |
| **Table S9** *(full 14 models)* | 7-temperature sharpness sweep at fixed top-1 labels | [`stage5_strengthening/p0_sharpness_sweep.csv`](results/stage5_strengthening/p0_sharpness_sweep.csv) |
| **Table S10** | Additive multimodal margin complementarity (all 14 models) | `stage2_diagnostics/complementarity_summary_primary.csv`, `multimodal_margin_complementarity_primary.csv` |
| **Table S11** | Nested fitted temperatures (`inner-OOF`) & bound-hit audit | `stage3_calibration/fitted_temperatures.csv`; `stage5_strengthening/p1_post_temp_bound_audit.csv` |
| **Table S12** | Complete controlled `SNV` vs. `non-SNV` ablation | `stage6_ipt_revision/p1_snv_ablation_primary.csv`, `p1_snv_ablation_august.csv`, `p1_snv_ablation_contrasts.csv` |
| **Figure 1** | Experimental setup & dual-scale workflow schematic | `build_figure1_svg.py`, `export_figure1.cjs`; `audit_manifest.json` (+ setup photos & spectrum `.npz` on Zenodo) |
| **Figure 2** | Optical characterization, pixel subsampling & affine residuals | `stage4_optical/intrakernel_pixel_sampling_summary.csv`; `stage5_strengthening/p0_optical_perturbation_physics.csv`; `stage6_ipt_revision/p1_wavelength_affine_residuals.csv` (+ cultivar spectra `.npz` on Zenodo) |
| **Figures 3–6** | Scale decoupling, margins, calibration, SNV & robustness | Redrawn directly from [`results/`](results/) CSVs via `scripts/build_figures.py` |

---

## Standalone Redraw of Figures 3–6 from `results/`

From the repository root (requiring Python, NumPy, pandas, Pillow, and Matplotlib):

```python
from pathlib import Path
from runpy import run_path

plots = run_path("scripts/build_figures.py")
g = plots["build_figure_3"].__globals__
g["PBP"] = Path("results").resolve()
g["S6"] = g["PBP"] / "stage6_ipt_revision"
g["FIG_DIR"] = Path("figures_out").resolve()
g["FIG_DIR"].mkdir(parents=True, exist_ok=True)
for idx in range(3, 7):
    plots[f"build_figure_{idx}"]()
```
