"""Stage 1: Unified cross-factorial SWIR (4 classifiers x 4 representations; 1D-CNN x 2)
and handcrafted RGB (JPEG-M, RAW-M) models with strictly leak-free inner-OOF scores
and per-fold 6-condition August predictions.

Outputs saved under analysis/path_b_plus/models_and_preds/.
Resumable per configuration.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
sys.path.insert(0, str(PROJECT))
from night_pilot.phase5.model_bundle import FROZEN_LABEL_SPACE

LABELS = np.asarray(FROZEN_LABEL_SPACE)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
SEED = 20261126
EPS = 1e-12


def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)


def build_spectral_reps(median: np.ndarray, stats: np.ndarray) -> dict[str, np.ndarray]:
    # stats order from Stage 0: [mean, std, q10, q50, q90]
    mean = stats[:, 0]
    std = stats[:, 1]
    iqr90_10 = stats[:, 4] - stats[:, 2]
    scale = np.maximum(mean.std(axis=1, keepdims=True), 1e-6)
    center = snv(mean)
    return {
        "median": snv(median).astype(np.float64),
        "mean": center.astype(np.float64),
        "mean_std": np.concatenate([center, std / scale], axis=1).astype(np.float64),
        "distribution": np.concatenate([center, std / scale, iqr90_10 / scale], axis=1).astype(np.float64),
    }


def aligned_probs_and_scores(model, X: np.ndarray, has_raw_logits: bool) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(model.classes_)
    order = [int(np.flatnonzero(classes == lbl).item()) for lbl in LABELS]
    probs = np.asarray(model.predict_proba(X), dtype=np.float64)[:, order]
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    if has_raw_logits:
        scores = np.asarray(model.decision_function(X), dtype=np.float64)[:, order]
    else:
        scores = np.log(np.clip(probs, EPS, 1.0))
    return probs, scores


def make_classical_estimator(clf_name: str, params: dict, n_features: int, n_train: int):
    if clf_name in ("logistic", "logistic_fixedC1", "m_logistic"):
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
            n_jobs=6,
        )
    raise ValueError(clf_name)


def fit_and_predict_classical(
    clf_name: str,
    params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    rec_tr: np.ndarray,
    X_eval_list: list[np.ndarray],
    cal_splits: int,
    seed_offset: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    base = make_classical_estimator(clf_name, params, X_tr.shape[1], len(X_tr))
    if clf_name == "rbf_svm":
        sgkf = StratifiedGroupKFold(n_splits=cal_splits, shuffle=True, random_state=SEED + seed_offset)
        cv_idx = list(sgkf.split(X_tr, y_tr, rec_tr))
        for a, b in cv_idx:
            assert not (set(rec_tr[a]) & set(rec_tr[b]))
        model = CalibratedClassifierCV(base, cv=cv_idx, n_jobs=min(3, cal_splits), ensemble=True)
        model.fit(X_tr, y_tr)
        has_logits = False
    else:
        model = base
        model.fit(X_tr, y_tr)
        has_logits = clf_name in ("logistic", "logistic_fixedC1", "m_logistic")
    return [aligned_probs_and_scores(model, Xe, has_logits) for Xe in X_eval_list]


def run_classical_config(
    config_id: str,
    clf_name: str,
    grid: list[dict],
    X_july: np.ndarray,
    X_aug: np.ndarray,
    rows_july: pd.DataFrame,
    out_dir: Path,
) -> None:
    target = out_dir / f"{config_id}.npz"
    meta_path = out_dir / f"{config_id}_meta.json"
    if target.exists() and meta_path.exists():
        logging.info("SKIP existing %s", config_id)
        return

    t0 = time.monotonic()
    y_july = rows_july["variety_name"].to_numpy()
    rec_july = rows_july["record_id"].to_numpy()
    folds = rows_july["fold"].to_numpy()

    outer_oof_prob = np.zeros((len(rows_july), 12), dtype=np.float64)
    outer_oof_score = np.zeros((len(rows_july), 12), dtype=np.float64)
    ext_fold_prob = np.zeros((4, len(X_aug), 12), dtype=np.float64)
    ext_fold_score = np.zeros((4, len(X_aug), 12), dtype=np.float64)
    save_dict: dict[str, np.ndarray] = {}
    fold_meta = []

    for fold in range(4):
        outer_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)
        inner_col = rows_july[f"inner_fold_for_outer{fold}"].to_numpy()
        inner_vals = inner_col[outer_idx]
        assert (inner_vals >= 0).all()

        # Hyperparameter selection strictly via 3-fold inner tray-grouped OOF BA
        if len(grid) == 1:
            chosen = grid[0]
            cand_records = [dict(params=chosen, inner_oof_ba=None)]
        else:
            cand_records = []
            best_ba = -1.0
            chosen = grid[0]
            for params in grid:
                pred_inner = np.empty(len(outer_idx), dtype=object)
                for g in range(3):
                    tr_rel = np.flatnonzero(inner_vals != g)
                    va_rel = np.flatnonzero(inner_vals == g)
                    tr_abs = outer_idx[tr_rel]
                    va_abs = outer_idx[va_rel]
                    est = make_classical_estimator(clf_name, params, X_july.shape[1], len(tr_abs))
                    est.fit(X_july[tr_abs], y_july[tr_abs])
                    pred_inner[va_rel] = est.predict(X_july[va_abs])
                sc = float(balanced_accuracy_score(y_july[outer_idx], pred_inner))
                cand_records.append(dict(params=params, inner_oof_ba=sc))
                if sc > best_ba:
                    best_ba = sc
                    chosen = params

        # Generate strictly independent inner-OOF probabilities & scores across ALL outer_train trays
        inner_oof_p = np.zeros((len(outer_idx), 12), dtype=np.float64)
        inner_oof_s = np.zeros((len(outer_idx), 12), dtype=np.float64)
        for g in range(3):
            tr_rel = np.flatnonzero(inner_vals != g)
            va_rel = np.flatnonzero(inner_vals == g)
            tr_abs = outer_idx[tr_rel]
            va_abs = outer_idx[va_rel]
            # Inside inner_train (2/3 of outer_train), smallest variety has >=2 trays -> cal_splits=2
            [(vp, vs)] = fit_and_predict_classical(
                clf_name, chosen, X_july[tr_abs], y_july[tr_abs], rec_july[tr_abs],
                [X_july[va_abs]], cal_splits=2, seed_offset=fold * 10 + g,
            )
            inner_oof_p[va_rel] = vp
            inner_oof_s[va_rel] = vs

        # Refit on full outer_train -> predict outer_test and August external (6 conditions)
        [(tp, ts), (ep, es)] = fit_and_predict_classical(
            clf_name, chosen, X_july[outer_idx], y_july[outer_idx], rec_july[outer_idx],
            [X_july[test_idx], X_aug], cal_splits=3, seed_offset=fold * 100 + 7,
        )
        outer_oof_prob[test_idx] = tp
        outer_oof_score[test_idx] = ts
        ext_fold_prob[fold] = ep
        ext_fold_score[fold] = es

        save_dict[f"inner_oof_idx_fold{fold}"] = outer_idx.astype(np.int32)
        save_dict[f"inner_oof_prob_fold{fold}"] = inner_oof_p.astype(np.float64)
        save_dict[f"inner_oof_score_fold{fold}"] = inner_oof_s.astype(np.float64)

        inner_ba = float(balanced_accuracy_score(y_july[outer_idx], LABELS[inner_oof_p.argmax(1)]))
        test_ba = float(balanced_accuracy_score(y_july[test_idx], LABELS[tp.argmax(1)]))
        fold_meta.append(dict(fold=fold, chosen=chosen, inner_oof_ba=inner_ba, outer_test_ba=test_ba, candidates=cand_records))

    overall_ba = float(balanced_accuracy_score(y_july, LABELS[outer_oof_prob.argmax(1)]))
    save_dict["outer_oof_prob"] = outer_oof_prob
    save_dict["outer_oof_score"] = outer_oof_score
    save_dict["ext_fold_prob"] = ext_fold_prob
    save_dict["ext_fold_score"] = ext_fold_score
    np.savez_compressed(target, **save_dict)

    meta = dict(
        config_id=config_id,
        clf_name=clf_name,
        score_type="raw_logits" if clf_name in ("logistic", "logistic_fixedC1", "m_logistic") else "log_prob",
        overall_oof_ba=overall_ba,
        elapsed_seconds=round(time.monotonic() - t0, 2),
        folds=fold_meta,
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("DONE %-32s OOF BA=%.4f (%.1fs)", config_id, overall_ba, time.monotonic() - t0)


def run_cnn1d_config(
    config_id: str,
    X_july: np.ndarray,
    X_aug: np.ndarray,
    rows_july: pd.DataFrame,
    out_dir: Path,
    max_epochs: int = 80,
) -> None:
    target = out_dir / f"{config_id}.npz"
    meta_path = out_dir / f"{config_id}_meta.json"
    if target.exists() and meta_path.exists():
        logging.info("SKIP existing %s", config_id)
        return

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

    def train_one(
        tr_x: np.ndarray,
        tr_y: np.ndarray,
        eval_x_list: list[np.ndarray],
        epochs: int,
        seed: int,
        val_x: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        scaler = StandardScaler().fit(tr_x)
        tr_t = torch.tensor(scaler.transform(tr_x), dtype=torch.float32)
        yt = torch.tensor(tr_y, dtype=torch.long)
        val_t = torch.tensor(scaler.transform(val_x), dtype=torch.float32) if val_x is not None else None

        counts = np.bincount(tr_y, minlength=12)
        weights = torch.tensor(len(tr_y) / (12.0 * counts), dtype=torch.float32)
        model = CNN1D()
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

        best_ba = -1.0
        best_epoch = 1
        best_val_logits = None
        for ep in range(1, epochs + 1):
            model.train()
            order = np.random.default_rng(seed + ep).permutation(len(tr_y))
            for j in range(0, len(order), 64):
                b = order[j : j + 64]
                opt.zero_grad()
                loss = loss_fn(model(tr_t[b]), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if val_t is not None and val_y is not None:
                model.eval()
                with torch.no_grad():
                    vl = torch.cat([model(val_t[j : j + 256]) for j in range(0, len(val_t), 256)]).numpy()
                ba = float(balanced_accuracy_score(val_y, vl.argmax(1)))
                if ba > best_ba:
                    best_ba = ba
                    best_epoch = ep
                    best_val_logits = vl.astype(np.float64)

        if val_t is not None:
            return best_epoch, best_ba, best_val_logits

        model.eval()
        outputs = []
        with torch.no_grad():
            for ex in eval_x_list:
                et = torch.tensor(scaler.transform(ex), dtype=torch.float32)
                logits = torch.cat([model(et[j : j + 256]) for j in range(0, len(et), 256)]).numpy().astype(np.float64)
                z = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(z)
                probs /= probs.sum(axis=1, keepdims=True)
                outputs.append((probs, logits))
        return outputs

    t0 = time.monotonic()
    y_idx = np.array([LABEL_TO_IDX[v] for v in rows_july["variety_name"]], dtype=np.int64)
    folds = rows_july["fold"].to_numpy()

    outer_oof_prob = np.zeros((len(rows_july), 12), dtype=np.float64)
    outer_oof_score = np.zeros((len(rows_july), 12), dtype=np.float64)
    ext_fold_prob = np.zeros((4, len(X_aug), 12), dtype=np.float64)
    ext_fold_score = np.zeros((4, len(X_aug), 12), dtype=np.float64)
    save_dict: dict[str, np.ndarray] = {}
    fold_meta = []

    for fold in range(4):
        outer_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)
        inner_vals = rows_july[f"inner_fold_for_outer{fold}"].to_numpy()[outer_idx]

        inner_oof_s = np.zeros((len(outer_idx), 12), dtype=np.float64)
        inner_oof_p = np.zeros((len(outer_idx), 12), dtype=np.float64)
        chosen_epochs = []
        for g in range(3):
            tr_rel = np.flatnonzero(inner_vals != g)
            va_rel = np.flatnonzero(inner_vals == g)
            tr_abs = outer_idx[tr_rel]
            va_abs = outer_idx[va_rel]
            be, bba, v_logits = train_one(
                X_july[tr_abs], y_idx[tr_abs], [], epochs=max_epochs,
                seed=SEED + fold * 100 + g * 10, val_x=X_july[va_abs], val_y=y_idx[va_abs],
            )
            chosen_epochs.append(be)
            z = v_logits - v_logits.max(axis=1, keepdims=True)
            vp = np.exp(z)
            vp /= vp.sum(axis=1, keepdims=True)
            inner_oof_s[va_rel] = v_logits
            inner_oof_p[va_rel] = vp

        selected_ep = int(np.round(np.mean(chosen_epochs)))
        selected_ep = max(10, min(max_epochs, selected_ep))
        [(tp, ts), (ep, es)] = train_one(
            X_july[outer_idx], y_idx[outer_idx], [X_july[test_idx], X_aug],
            epochs=selected_ep, seed=SEED + fold * 1000 + 99,
        )
        outer_oof_prob[test_idx] = tp
        outer_oof_score[test_idx] = ts
        ext_fold_prob[fold] = ep
        ext_fold_score[fold] = es

        save_dict[f"inner_oof_idx_fold{fold}"] = outer_idx.astype(np.int32)
        save_dict[f"inner_oof_prob_fold{fold}"] = inner_oof_p
        save_dict[f"inner_oof_score_fold{fold}"] = inner_oof_s

        inner_ba = float(balanced_accuracy_score(y_idx[outer_idx], inner_oof_p.argmax(1)))
        test_ba = float(balanced_accuracy_score(y_idx[test_idx], tp.argmax(1)))
        fold_meta.append(dict(fold=fold, inner_best_epochs=chosen_epochs, selected_epochs=selected_ep,
                              inner_oof_ba=inner_ba, outer_test_ba=test_ba))
        logging.info("  %s fold%d epochs=%s->%d inner_ba=%.4f test_ba=%.4f",
                     config_id, fold, chosen_epochs, selected_ep, inner_ba, test_ba)

    overall_ba = float(balanced_accuracy_score(y_idx, outer_oof_prob.argmax(1)))
    save_dict["outer_oof_prob"] = outer_oof_prob
    save_dict["outer_oof_score"] = outer_oof_score
    save_dict["ext_fold_prob"] = ext_fold_prob
    save_dict["ext_fold_score"] = ext_fold_score
    np.savez_compressed(target, **save_dict)

    meta = dict(
        config_id=config_id,
        clf_name="cnn1d",
        score_type="raw_logits",
        overall_oof_ba=overall_ba,
        elapsed_seconds=round(time.monotonic() - t0, 2),
        folds=fold_meta,
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("DONE %-32s OOF BA=%.4f (%.1fs)", config_id, overall_ba, time.monotonic() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["all", "classical", "neural", "rgb"], default="all")
    args = ap.parse_args()

    out_dir = ROOT / "models_and_preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "stage1.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    rows_july = pd.read_csv(ROOT / "cache" / "primary_rows.csv", dtype={"record_id": str})
    with np.load(ROOT / "cache" / "primary_spectra.npz") as psp:
        reps_july = build_spectral_reps(psp["median"], psp["statistics"])
    with np.load(ROOT / "cache" / "external_spectra.npz") as esp:
        reps_aug = build_spectral_reps(esp["median"], esp["statistics"])
    with np.load(ROOT / "cache" / "rgb_and_frozen_refs.npz") as rgbf:
        m_jpeg_july = rgbf["m_jpeg_july"]
        m_jpeg_aug = rgbf["m_jpeg_aug"]
        m_raw_july = rgbf["m_raw_july"]
        m_raw_aug = rgbf["m_raw_aug"]

    if args.group in ("all", "rgb"):
        run_classical_config("M_jpeg", "m_logistic", [{"C": 1.0}], m_jpeg_july, m_jpeg_aug, rows_july, out_dir)
        run_classical_config("M_raw", "m_logistic", [{"C": 1.0}], m_raw_july, m_raw_aug, rows_july, out_dir)

    if args.group in ("all", "classical"):
        grids = {
            "logistic_fixedC1": [{"C": 1.0}],
            "logistic": [{"C": c} for c in (0.1, 1.0, 10.0)],
            "rbf_svm": [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01)],
            "extra_trees": [{"leaf": l, "max_features": f} for l in (2, 5) for f in ("sqrt", 0.3)],
        }
        for clf_name, grid in grids.items():
            for rep_name in ("median", "mean", "mean_std", "distribution"):
                run_classical_config(
                    f"S_{clf_name}_{rep_name}",
                    clf_name,
                    grid,
                    reps_july[rep_name],
                    reps_aug[rep_name],
                    rows_july,
                    out_dir,
                )

    if args.group in ("all", "neural"):
        for rep_name in ("median", "mean"):
            run_cnn1d_config(
                f"S_cnn1d_{rep_name}",
                reps_july[rep_name],
                reps_aug[rep_name],
                rows_july,
                out_dir,
                max_epochs=80,
            )

    # Summarize all completed configurations
    summary_rows = []
    for mp in sorted(out_dir.glob("*_meta.json")):
        d = json.loads(mp.read_text(encoding="utf-8"))
        summary_rows.append(
            dict(
                config_id=d["config_id"],
                clf_name=d["clf_name"],
                score_type=d["score_type"],
                overall_oof_ba=d["overall_oof_ba"],
                elapsed_seconds=d["elapsed_seconds"],
            )
        )
    pd.DataFrame(summary_rows).to_csv(ROOT / "stage1_summary.csv", index=False)
    logging.info("Stage 1 (%s) COMPLETE. Total saved configs: %d", args.group, len(summary_rows))


if __name__ == "__main__":
    main()
