"""Stage 0: Extract primary inner-mask SWIR cache, verify all external/RGB caches,
and audit outer + inner tray-grouped splits for Path B+.

Read-only with respect to all raw inputs, annotations, outputs/, and frozen phases.
All artifacts are written strictly inside analysis/path_b_plus/.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PILOT = PROJECT / "analysis" / "roi_pixel_pilot_20261113"
FOLLOWUP = PILOT / "controlled_followup"
CROSSDATE = PROJECT / "analysis" / "cross_date_attribution_20261114"
FINAL = PROJECT / "analysis" / "phase7_final_run"
LONG = PROJECT / "analysis" / "long_term_validation" / "data"
DATA = PROJECT / "outputs" / "datasets" / "Maize_RGB_HSI_v2.0"
ARCHIVE = PROJECT / "analysis" / "manuscript_v4" / "_local" / "zenodo_upload_test" / "temporary_archives" / "00_metadata_annotations.zip"

sys.path.insert(0, str(PROJECT))
from night_pilot.dataset_freeze import build_inner_labels
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE
from night_pilot.pipeline import parse_hdr
from night_pilot.robustness import assert_no_group_leakage
from night_pilot.seed_lots import assert_lot_declared

LABELS = np.asarray(FROZEN_LABEL_SPACE)
INNER_SEED = 20261126


def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_primary_tray(
    record: str,
    block: pd.DataFrame,
    baseline_snv: np.ndarray,
    inventory: pd.DataFrame,
) -> tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    member = f"annotations/primary_cohort/{record}/hsi_labels.png"
    with zipfile.ZipFile(ARCHIVE) as zf:
        payload = zf.read(member)
    frozen_row = inventory.loc[f"{record}/hsi_labels.png"]
    assert len(payload) == int(frozen_row["bytes"]), f"Mask size mismatch: {record}"
    assert hashlib.sha256(payload).hexdigest() == frozen_row["sha256"], f"Mask sha256 mismatch: {record}"
    labels = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    inner = build_inner_labels(labels)

    first = block.iloc[0]
    hdr = parse_hdr(Path(first.hsi_path).with_suffix(".hdr"))
    waves = np.asarray(hdr["wavelength"], dtype=np.float32)
    keep = (waves >= 950.0) & (waves <= 1650.0)
    assert int(keep.sum()) == 421, f"Unexpected kept bands {keep.sum()} for {record}"
    shape = (int(hdr["lines"]), int(hdr["bands"]), int(hdr["samples"]))
    assert inner.shape == (shape[0], shape[2]), f"Shape mismatch for {record}"

    raw = np.memmap(first.hsi_path, dtype="<u2", mode="r", shape=shape)
    dark = np.fromfile(first.hsi_dark_path, dtype="<u2").reshape(shape[1:]).astype(np.float32)
    white = np.fromfile(first.hsi_white_path, dtype="<u2").reshape(shape[1:]).astype(np.float32)
    den = np.where((white - dark) > 0, white - dark, np.nan)

    seed_ids: list[str] = []
    medians: list[np.ndarray] = []
    stats: list[np.ndarray] = []
    qcs: list[dict] = []

    for row in block.itertuples():
        yy, xx = np.nonzero(inner == int(row.hsi_seed_index))
        assert len(yy) == int(row.pixels_inner) and len(yy) > 0, f"Inner pixel mismatch: {row.seed_id}"
        allraw = np.asarray(raw[yy, :, xx], dtype=np.float32)
        vals = ((allraw - dark[:, xx].T) / den[:, xx].T)[:, keep]
        med = np.nanmedian(vals, axis=0)
        diff = float(np.max(np.abs(snv(med[None])[0] - baseline_snv[row.Index])))
        assert diff <= 1e-6, f"Frozen SNV replay error {diff} for {row.seed_id}"
        valid = np.isfinite(vals).all(axis=1)
        assert valid.any(), f"No finite pixels for {row.seed_id}"
        v = vals[valid]
        st = np.stack([v.mean(0), v.std(0), *np.quantile(v, [0.1, 0.5, 0.9], axis=0)]).astype(np.float32)
        seed_ids.append(str(row.seed_id))
        medians.append(med.astype(np.float32))
        stats.append(st)
        qcs.append(
            dict(
                seed_id=str(row.seed_id),
                record_id=record,
                variety_id=str(row.variety_id),
                variety_name=str(row.variety_name),
                fold=int(row.fold),
                inner_pixels=int(len(yy)),
                valid_pixels=int(valid.sum()),
                finite_fraction=float(np.isfinite(vals).mean()),
                reflectance_mean=float(v.mean()),
                spectral_spatial_sd=float(v.std(0).mean()),
                median_snv_max_abs_error=diff,
                quality_flag=str(row.quality_flag),
            )
        )
    del raw
    return (
        record,
        seed_ids,
        np.asarray(medians, dtype=np.float32),
        np.asarray(stats, dtype=np.float32),
        waves[keep],
        qcs,
    )


def main() -> None:
    t0 = time.monotonic()
    (ROOT / "cache").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "stage0.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # 1. Load canonical cohort metadata
    primary_rows = pd.read_csv(PILOT / "cohort.csv", dtype={"record_id": str})
    external_rows = pd.read_csv(FOLLOWUP / "external_rows.csv", dtype={"record_id": str})
    assert len(primary_rows) == 3627 and primary_rows["record_id"].nunique() == 67
    assert primary_rows["variety_name"].nunique() == 12
    assert set(primary_rows["variety_name"]) == set(LABELS)
    assert len(external_rows) == 1680 and external_rows["entity_seed_id"].nunique() == 280
    assert external_rows["entity_tray_id"].nunique() == 10
    assert sorted(external_rows["condition"].unique().tolist()) == [
        "BASE",
        "FLIP",
        "LEFT_OFF",
        "REP",
        "REPOSITION",
        "RIGHT_OFF",
    ]
    logging.info("Loaded primary (%d seeds, %d trays) and external (%d obs, 280 physical seeds x 6 conds)",
                 len(primary_rows), primary_rows["record_id"].nunique(), len(external_rows))

    # 2. Extract or verify primary inner-mask spectra cache
    primary_cache = ROOT / "cache" / "primary_spectra.npz"
    if primary_cache.exists():
        with np.load(primary_cache) as a:
            assert a["seed_ids"].tolist() == primary_rows["seed_id"].tolist()
            assert a["median"].shape == (3627, 421) and a["statistics"].shape == (3627, 5, 421)
            waves = a["waves"]
        logging.info("Verified existing primary_spectra.npz cache.")
    else:
        feat_rows = pd.read_parquet(FINAL / "derived/board_normalized/second_batch/feature_rows.parquet")
        pos = pd.Series(np.arange(len(feat_rows)), index=feat_rows["seed_id"]).loc[primary_rows["seed_id"]].to_numpy()
        baseline_snv = np.load(FINAL / "derived/board_normalized/second_batch/hsi_snv.npy", mmap_mode="r")[pos]
        inventory = pd.read_csv(DATA / "annotation_snapshot_inventory.csv").set_index("relative_path")

        all_ids: list[str] = []
        all_med: list[np.ndarray] = []
        all_stats: list[np.ndarray] = []
        all_qc: list[dict] = []
        waves_ref = None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {
                pool.submit(extract_primary_tray, rec, blk, baseline_snv, inventory): rec
                for rec, blk in primary_rows.groupby("record_id", sort=True)
            }
            done = 0
            for fut in as_completed(futs):
                rec, sids, med, st, wv, qcs = fut.result()
                all_ids.extend(sids)
                all_med.append(med)
                all_stats.append(st)
                all_qc.extend(qcs)
                if waves_ref is None:
                    waves_ref = wv
                else:
                    assert np.allclose(waves_ref, wv, atol=1e-4)
                done += 1
                if done % 15 == 0 or done == 67:
                    logging.info("Extracted primary HSI trays: %d/67", done)

        order = pd.Series(np.arange(len(all_ids)), index=all_ids).loc[primary_rows["seed_id"]].to_numpy()
        med_arr = np.concatenate(all_med, axis=0)[order]
        stat_arr = np.concatenate(all_stats, axis=0)[order]
        assert np.isfinite(med_arr).all() and np.isfinite(stat_arr).all()
        np.savez_compressed(
            primary_cache,
            seed_ids=primary_rows["seed_id"].to_numpy(dtype=str),
            waves=waves_ref,
            median=med_arr,
            statistics=stat_arr,
        )
        pd.DataFrame(all_qc).to_csv(ROOT / "cache" / "primary_extraction_qc.csv", index=False)
        waves = waves_ref
        max_err = max(q["median_snv_max_abs_error"] for q in all_qc)
        logging.info("Saved primary_spectra.npz (max frozen SNV replay diff = %.3g)", max_err)

    # 3. Assemble verified external 6-condition inner-mask spectra cache
    ext_cache = ROOT / "cache" / "external_spectra.npz"
    if not ext_cache.exists():
        ext_med: list[np.ndarray] = []
        ext_stat: list[np.ndarray] = []
        ext_keys: list[tuple[str, str]] = []
        for rec, blk in external_rows.groupby("record_id", sort=True):
            with np.load(FOLLOWUP / "external_cache" / f"{rec}.npz") as a:
                assert a["ids"].tolist() == blk["entity_seed_id"].tolist()
                ext_med.append(a["median"])
                ext_stat.append(a["statistics"])
                ext_keys.extend(zip([rec] * len(blk), a["ids"].tolist()))
        lookup = {k: i for i, k in enumerate(ext_keys)}
        order = [lookup[k] for k in zip(external_rows["record_id"], external_rows["entity_seed_id"])]
        med_ext = np.concatenate(ext_med, axis=0)[order]
        stat_ext = np.concatenate(ext_stat, axis=0)[order]
        assert med_ext.shape == (1680, 421) and stat_ext.shape == (1680, 5, 421)
        assert np.isfinite(med_ext).all() and np.isfinite(stat_ext).all()
        np.savez_compressed(
            ext_cache,
            record_ids=external_rows["record_id"].to_numpy(dtype=str),
            entity_seed_ids=external_rows["entity_seed_id"].to_numpy(dtype=str),
            waves=waves,
            median=med_ext,
            statistics=stat_ext,
        )
        logging.info("Saved verified external_spectra.npz (1680 x 421).")

    # 4. Verify RGB features (JPEG-M, RAW-M) and frozen D_phys probabilities
    jr = pd.read_parquet(FINAL / "derived/board_normalized/second_batch/feature_rows.parquet")
    ji = pd.Series(np.arange(len(jr)), index=jr["seed_id"]).loc[primary_rows["seed_id"]].to_numpy()
    m_jpeg_july = np.asarray(np.load(FINAL / "derived/board_normalized/second_batch/rgb_basic.npy")[ji], dtype=np.float64)
    m_jpeg_aug = np.asarray(
        np.load(FINAL / "derived/board_normalized/third_batch/rgb_basic.npy")[external_rows["_row_position"].to_numpy()],
        dtype=np.float64,
    )
    with np.load(CROSSDATE / "cache" / "RAW_fixedWB_features.npz") as rf:
        m_raw_july = np.asarray(rf["july"], dtype=np.float64)
        m_raw_aug = np.asarray(rf["august"], dtype=np.float64)
    assert m_jpeg_july.shape == (3627, 28) and m_raw_july.shape == (3627, 28)
    assert m_jpeg_aug.shape == (1680, 28) and m_raw_aug.shape == (1680, 28)
    assert np.allclose(m_jpeg_july[:, :8], m_raw_july[:, :8], atol=1e-10), "Geometry mismatch in July RAW vs JPEG"
    assert np.allclose(m_jpeg_aug[:, :8], m_raw_aug[:, :8], atol=1e-10), "Geometry mismatch in August RAW vs JPEG"

    d_rows_july = (
        pd.read_parquet(FINAL / "d_phys/second_batch_common_oof_rows_and_D_phys.parquet")
        .set_index("seed_id")
        .loc[primary_rows["seed_id"]]
    )
    assert d_rows_july["fold"].to_numpy().tolist() == primary_rows["fold"].to_numpy().tolist()
    d_prob_july = d_rows_july[[f"prob_{lbl}" for lbl in LABELS]].to_numpy(dtype=np.float64)

    d_rows_aug_full = pd.read_parquet(FINAL / "d_phys/third_batch_prediction_rows.parquet")
    full_tab_aug = pd.read_parquet(FINAL / "tabular/batch3_prediction_rows.parquet")
    assert list(zip(d_rows_aug_full["record_id"], d_rows_aug_full["entity_seed_id"])) == list(
        zip(full_tab_aug["record_id"], full_tab_aug["entity_seed_id"])
    )
    d_prob_aug = np.load(FINAL / "d_phys/third_batch_probabilities_D_phys.npy")[external_rows["_row_position"].to_numpy()]

    frozen_m_july = np.load(FINAL / "tabular/batch2_oof_probabilities_M_phys.npy")
    frozen_s_july = np.load(FINAL / "tabular/batch2_oof_probabilities_S.npy")
    frozen_m_aug = np.load(FINAL / "tabular/batch3_probabilities_M_phys.npy")[external_rows["_row_position"].to_numpy()]
    frozen_s_aug = np.load(FINAL / "tabular/batch3_probabilities_S.npy")[external_rows["_row_position"].to_numpy()]

    for arr_name, arr in [
        ("d_prob_july", d_prob_july),
        ("d_prob_aug", d_prob_aug),
        ("frozen_m_july", frozen_m_july),
        ("frozen_s_july", frozen_s_july),
        ("frozen_m_aug", frozen_m_aug),
        ("frozen_s_aug", frozen_s_aug),
    ]:
        assert arr.shape[1] == 12 and np.isfinite(arr).all() and np.allclose(arr.sum(1), 1.0, atol=1e-5), arr_name

    np.savez_compressed(
        ROOT / "cache" / "rgb_and_frozen_refs.npz",
        m_jpeg_july=m_jpeg_july,
        m_jpeg_aug=m_jpeg_aug,
        m_raw_july=m_raw_july,
        m_raw_aug=m_raw_aug,
        d_prob_july=d_prob_july,
        d_prob_aug=d_prob_aug,
        frozen_m_july=frozen_m_july,
        frozen_s_july=frozen_s_july,
        frozen_m_aug=frozen_m_aug,
        frozen_s_aug=frozen_s_aug,
    )

    # 5. Audit outer 4 folds + construct leak-free inner 3-fold tray-grouped splits inside each outer_train
    y_july = primary_rows["variety_name"].to_numpy()
    rec_july = primary_rows["record_id"].to_numpy()
    folds_july = primary_rows["fold"].to_numpy()

    inner_splits: dict[str, dict] = {}
    for fold in range(4):
        outer_idx = np.flatnonzero(folds_july != fold)
        test_idx = np.flatnonzero(folds_july == fold)
        outer_trays = set(rec_july[outer_idx])
        test_trays = set(rec_july[test_idx])
        assert not (outer_trays & test_trays)
        assert_no_group_leakage(outer_trays, test_trays)
        assert_lot_declared(outer_trays, test_trays)
        assert len(np.unique(y_july[outer_idx])) == 12 and len(np.unique(y_july[test_idx])) == 12

        # Stratified 3-fold tray-grouped inner CV on outer_train (~50 trays)
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=INNER_SEED + fold)
        inner_fold_id = np.full(len(outer_idx), -1, dtype=int)
        sub_audit = []
        for g, (tr_rel, va_rel) in enumerate(sgkf.split(outer_idx, y_july[outer_idx], rec_july[outer_idx])):
            tr_abs = outer_idx[tr_rel]
            va_abs = outer_idx[va_rel]
            tr_trays = set(rec_july[tr_abs])
            va_trays = set(rec_july[va_abs])
            assert not (tr_trays & va_trays)
            assert not (va_trays & test_trays)
            assert_no_group_leakage(tr_trays, va_trays)
            assert_lot_declared(tr_trays, va_trays)
            assert len(np.unique(y_july[tr_abs])) == 12 and len(np.unique(y_july[va_abs])) == 12
            inner_fold_id[va_rel] = g
            sub_audit.append(
                dict(
                    inner_fold=g,
                    n_train_seeds=int(len(tr_abs)),
                    n_val_seeds=int(len(va_abs)),
                    n_train_trays=int(len(tr_trays)),
                    n_val_trays=int(len(va_trays)),
                    val_trays=sorted(va_trays),
                )
            )
        assert (inner_fold_id >= 0).all()
        inner_splits[f"fold{fold}"] = dict(
            outer_train_trays=sorted(outer_trays),
            outer_test_trays=sorted(test_trays),
            inner_cv=sub_audit,
        )
        # Save exact inner fold assignments aligned to primary_rows
        primary_rows[f"inner_fold_for_outer{fold}"] = -1
        primary_rows.loc[outer_idx, f"inner_fold_for_outer{fold}"] = inner_fold_id

    # Verify schedule alignment
    b2_rows = pd.read_csv(LONG / "batch2_rows.csv", dtype={"record_id": str})
    b2_trays = pd.read_csv(LONG / "batch2_trays.csv", dtype={"record_id": str})
    b2_sched = np.load(LONG / "batch2_schedules.npz")
    assert set(b2_rows["seed_id"]) == set(primary_rows["seed_id"])
    assert len(b2_trays) == 67 and b2_sched["permutations"].shape[0] == 500

    b3_rows = pd.read_csv(LONG / "batch3_rows.csv", dtype={"record_id": str})
    b3_base = external_rows[external_rows["condition"].eq("BASE")]
    assert set(b3_rows["entity_seed_id"]) == set(b3_base["entity_seed_id"])

    primary_rows.to_csv(ROOT / "cache" / "primary_rows.csv", index=False)
    external_rows.to_csv(ROOT / "cache" / "external_rows.csv", index=False)

    manifest = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_seconds=round(time.monotonic() - t0, 2),
        primary_seeds=int(len(primary_rows)),
        primary_trays=int(primary_rows["record_id"].nunique()),
        external_observations=int(len(external_rows)),
        external_physical_seeds=int(external_rows["entity_seed_id"].nunique()),
        external_entity_trays=int(external_rows["entity_tray_id"].nunique()),
        bands_retained=421,
        wavelength_min_nm=float(waves.min()),
        wavelength_max_nm=float(waves.max()),
        label_space=LABELS.tolist(),
        inner_cv_splits=3,
        inner_cv_seed=INNER_SEED,
        d_phys_calibration_note="D_phys outer OOF and external predictions are fixed; no independent inner-val D logits exist without retraining 12 inner EfficientNet models on CPU.",
        splits=inner_splits,
    )
    save_json(ROOT / "audit_manifest.json", manifest)
    logging.info("Stage 0 COMPLETE in %.1fs. All leakage, lot, mask SHA-256, and schedule assertions PASSED.",
                 time.monotonic() - t0)


if __name__ == "__main__":
    main()
