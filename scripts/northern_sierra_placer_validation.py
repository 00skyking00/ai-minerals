"""Phase G: validation harness + model card for the northern-Sierra placer raster.

Reads Phase 1 + Phase 2 outputs and writes:

  - success_rate_curve_phase1_vs_phase2.png — P-A curves (Phase 1 vs per-pop
    Phase 2 vs fused) over the AOI, with anchor districts marked.
  - anchor_districts_decile_table.csv — per-district decile rank for each model.
  - phase1_vs_phase2_comparison.csv — mean rank + Brier + F1 + Cohen's Kappa +
    Recall at the top-decile cutoff for every (model, population).
  - model_card_northern_sierra_placer.md — the artifact recruiters and
    downstream consumers (gldbg) read. Covers AOI, CRS, features, label
    sources, CV scheme, calibration, anchor-district performance, the
    Phase 1 vs Phase 2 comparison, and explicit limitations.

Inputs (under data/derived/northern_sierra_placer/):
  phase1_index_250m.parquet                     (Phase C)
  pop_predictions_<pop>_250m.parquet            (Phase E, per population)
  pop_calibrated_<pop>_250m.parquet             (Phase E, per population)
  pop_fold_metrics_<pop>.csv                    (Phase E, per population)
  prospectivity_placer_northern_sierra_250m_fused.parquet (Phase F)

Each input is loaded with a clear error if missing. If a population's
Phase 2 outputs aren't there, the model card declares the deliverable as
the Phase 1 raster (per the plan's "ship Phase 1 if Phase 2 doesn't beat it"
rule).

Usage:
    .venv/bin/python scripts/northern_sierra_placer_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    recall_score,
)

from ai_minerals.metrics.bootstrap import (
    bootstrap_auc_pa_ci,
    bootstrap_capture_ci,
)
from ai_minerals.regions._northern_sierra_anchors import ANCHOR_DISTRICTS
from ai_minerals.regions.northern_sierra_placer import NORTHERN_SIERRA_PLACER


REGION = NORTHERN_SIERRA_PLACER

DATA_DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT_DIR = DATA_DERIVED / REGION.data_prefix

IN_PHASE1 = OUT_DIR / "phase1_index_250m.parquet"
IN_FUSED = OUT_DIR / "prospectivity_placer_northern_sierra_250m_fused.parquet"

POPULATIONS = ("placer_tertiary", "placer_quaternary")

OUT_PA_CURVE = OUT_DIR / "success_rate_curve_phase1_vs_phase2.png"
OUT_ANCHOR_TABLE = OUT_DIR / "anchor_districts_decile_table.csv"
OUT_COMPARISON = OUT_DIR / "phase1_vs_phase2_comparison.csv"
OUT_HEADLINE = OUT_DIR / "headline_metrics.csv"
OUT_CARD = OUT_DIR / "model_card_northern_sierra_placer.md"

HEADLINE_KS: tuple[float, ...] = (1.0, 5.0, 10.0)

HEADLINE_MODELS: tuple[tuple[str, str], ...] = (
    ("Phase 1 index", "phase1_score"),
    ("Phase 2 fused", "p_fused"),
    ("Phase 2 Tertiary (cal)", "p_cal_placer_tertiary"),
    ("Phase 2 Quaternary (cal)", "p_cal_placer_quaternary"),
)


def _success_rate_curve(score: pd.Series, positives: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frac_area_descending, frac_positives_captured) for a P-A curve."""
    mask = score.notna()
    s = score[mask].to_numpy()
    y = positives[mask].astype(int).to_numpy()
    order = np.argsort(-s)
    y_ranked = y[order]
    n_total = len(y_ranked)
    n_pos = int(y_ranked.sum())
    if n_pos == 0 or n_total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    cum_pos = np.cumsum(y_ranked)
    frac_area = (np.arange(n_total) + 1) / n_total
    frac_pos = cum_pos / n_pos
    return frac_area, frac_pos


def _anchor_cell_indices(df: pd.DataFrame) -> pd.Series:
    """Snap each anchor district (lon, lat) to the nearest grid-cell DataFrame index."""
    transformer = Transformer.from_crs("EPSG:4326", REGION.working_crs, always_xy=True)
    names, idxs = [], []
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    for name, (lon, lat) in ANCHOR_DISTRICTS.items():
        ax, ay = transformer.transform(lon, lat)
        d2 = (xs - ax) ** 2 + (ys - ay) ** 2
        cell = int(np.argmin(d2))
        names.append(name)
        idxs.append(df.index[cell])
    return pd.Series(idxs, index=names, name="cell_idx")


def _decile_rank(score: pd.Series) -> pd.Series:
    """Decile of each cell in score's distribution; 0 = top decile."""
    return pd.qcut(
        score.rank(method="first", ascending=False, pct=False, na_option="keep"),
        q=10,
        labels=range(10),
    )


def _top_decile_metrics(score: pd.Series, y: pd.Series) -> dict[str, float]:
    """F1, Cohen's Kappa, Recall at the top-decile cutoff. Brier separately."""
    mask = score.notna() & y.notna()
    s = score[mask].to_numpy()
    y_arr = y[mask].astype(int).to_numpy()
    if y_arr.sum() == 0:
        return {"brier": float("nan"), "f1": float("nan"),
                "kappa": float("nan"), "recall": float("nan"), "n_pos": 0}
    cutoff = float(np.percentile(s, 90))
    y_pred = (s >= cutoff).astype(int)
    return {
        "brier": float(brier_score_loss(y_arr, s)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_arr, y_pred)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "n_pos": int(y_arr.sum()),
        "cutoff": cutoff,
    }


def _load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _build_score_table(
    phase1: pd.DataFrame,
    fused: pd.DataFrame | None,
    per_pop: dict[str, dict[str, pd.DataFrame | None]],
) -> pd.DataFrame:
    """Merge Phase 1 + per-pop predictions + fused into a single per-cell DataFrame."""
    df = phase1[["row", "col", "x", "y", "phase1_score"]].copy()
    if fused is not None:
        df = df.merge(
            fused[["row", "col", "p_fused", "p_tertiary", "p_quaternary"]],
            on=["row", "col"], how="left",
        )
    for pop, files in per_pop.items():
        for stage, frame in files.items():
            if frame is None:
                continue
            cols = [c for c in ("p_rf", "p_lgbm", "p_stack", "p_cal", "p_pu") if c in frame.columns]
            if not cols:
                continue
            rename = {c: f"{c}_{pop}" for c in cols}
            df = df.merge(
                frame[["row", "col"] + cols].rename(columns=rename),
                on=["row", "col"], how="left",
            )
    return df


def _plot_pa_curves(
    df: pd.DataFrame,
    anchor_idxs: pd.Series,
    out_path: Path,
) -> dict[str, dict[str, float]]:
    """Write the P-A curve PNG and return per-model AUC-style metric (area under curve)."""
    anchors_mask = pd.Series(False, index=df.index)
    anchors_mask.loc[anchor_idxs.values] = True
    summary: dict[str, dict[str, float]] = {}

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    models = [
        ("Phase 1 index", "phase1_score", "C0", "-"),
        ("Phase 2 fused", "p_fused",      "C1", "-"),
        ("Phase 2 Tertiary (cal)",   "p_cal_placer_tertiary",   "C2", "--"),
        ("Phase 2 Quaternary (cal)", "p_cal_placer_quaternary", "C3", "--"),
    ]
    pos_int = anchors_mask.astype(int).to_numpy()
    for label, col, color, ls in models:
        if col not in df.columns:
            continue
        frac_area, frac_pos = _success_rate_curve(df[col], anchors_mask.astype(int))

        score_arr = df[col].to_numpy(dtype=float)
        finite = np.isfinite(score_arr)
        auc, auc_lo, auc_hi = bootstrap_auc_pa_ci(score_arr[finite], pos_int[finite])

        plot_label = f"{label} (AUC {auc:.2f} [{auc_lo:.2f}, {auc_hi:.2f}])"
        ax.plot(frac_area, frac_pos, color=color, linestyle=ls, label=plot_label, linewidth=2)
        summary[label] = {"auc_pa": float(auc), "auc_pa_lo": float(auc_lo), "auc_pa_hi": float(auc_hi)}

    ax.plot([0, 1], [0, 1], color="0.7", linewidth=1, label="random (y=x)")
    ax.set_xlabel("Fraction of area selected (descending score)")
    ax.set_ylabel("Fraction of anchor districts captured")
    ax.set_title(f"Success-rate / Prediction-Area curve — {REGION.slug}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return summary


def _build_anchor_table(df: pd.DataFrame, anchor_idxs: pd.Series) -> pd.DataFrame:
    """Per-district decile rank for every score column present in df."""
    score_cols = [c for c in df.columns if c.startswith(("phase1_score", "p_fused", "p_cal_"))]
    rows = []
    for district, idx in anchor_idxs.items():
        rec: dict[str, object] = {"district": district, "cell_idx": int(idx)}
        for col in score_cols:
            score = df[col]
            deciles = _decile_rank(score)
            rec[f"{col}_decile"] = (int(deciles.loc[idx]) if pd.notna(deciles.loc[idx]) else np.nan)
            rec[f"{col}_value"]  = float(score.loc[idx]) if pd.notna(score.loc[idx]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_comparison_table(
    df: pd.DataFrame, per_pop_labels: dict[str, pd.Series],
) -> pd.DataFrame:
    """Cross-model metric table: Brier / F1 / Kappa / Recall on per-pop labels.

    per_pop_labels: {"placer_tertiary": Series, "placer_quaternary": Series}
                     Series indexed identically to df, dtype int.
    """
    rows: list[dict[str, object]] = []

    for pop, y in per_pop_labels.items():
        for model_col in ("phase1_score", "p_fused", f"p_cal_{pop}",
                          f"p_rf_{pop}", f"p_lgbm_{pop}", f"p_stack_{pop}", f"p_pu_{pop}"):
            if model_col not in df.columns:
                continue
            m = _top_decile_metrics(df[model_col], y)
            m.update({"population": pop, "model": model_col})
            rows.append(m)
    return pd.DataFrame(rows)


def _build_headline_metrics(
    df: pd.DataFrame, anchor_idxs: pd.Series,
) -> pd.DataFrame:
    """Top-k% capture (with bootstrap CI) + enrichment for every model column present."""
    anchors_mask = np.zeros(len(df), dtype=bool)
    anchors_mask[anchor_idxs.values] = True

    rows: list[dict[str, object]] = []
    for label, col in HEADLINE_MODELS:
        if col not in df.columns:
            continue
        scores = df[col].to_numpy(dtype=float)
        finite = np.isfinite(scores)
        s = scores[finite]
        y = anchors_mask[finite]
        if s.size == 0 or y.sum() == 0:
            continue
        cis = bootstrap_capture_ci(s, y, ks_percent=HEADLINE_KS)
        for k in HEADLINE_KS:
            point, lo, hi = cis[k]
            base = k / 100.0
            rows.append({
                "model_label": label,
                "model": col,
                "k_percent": k,
                "capture_point": float(point),
                "capture_lo": float(lo),
                "capture_hi": float(hi),
                "enrichment_point": float(point / base) if base > 0 else float("nan"),
                "enrichment_lo": float(lo / base) if base > 0 else float("nan"),
                "enrichment_hi": float(hi / base) if base > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def _aggregate_fold_metrics(per_pop_metrics_files: dict[str, Path | None]) -> pd.DataFrame:
    """Per (population, model) mean/std/CI95 of roc_auc + pr_auc across folds."""
    rows: list[dict[str, object]] = []
    for pop, path in per_pop_metrics_files.items():
        if path is None or not path.exists():
            continue
        fm = pd.read_csv(path)
        # exclude global OOF rows (fold_id == -1, used for stacking)
        per_fold = fm[fm["fold_id"] >= 0] if "fold_id" in fm.columns else fm
        if per_fold.empty:
            continue
        for model, grp in per_fold.groupby("model"):
            roc = grp["roc_auc"].to_numpy(dtype=float)
            pr = grp["pr_auc"].to_numpy(dtype=float)
            roc = roc[np.isfinite(roc)]
            pr = pr[np.isfinite(pr)]
            if roc.size == 0:
                continue
            roc_lo, roc_hi = np.percentile(roc, [2.5, 97.5]) if roc.size >= 2 else (roc[0], roc[0])
            pr_lo, pr_hi = np.percentile(pr, [2.5, 97.5]) if pr.size >= 2 else (pr[0], pr[0])
            row: dict[str, object] = {
                "population": pop,
                "model": model,
                "n_folds": int(roc.size),
                "roc_auc_mean": float(roc.mean()),
                "roc_auc_std": float(roc.std(ddof=1)) if roc.size > 1 else 0.0,
                "roc_auc_lo": float(roc_lo),
                "roc_auc_hi": float(roc_hi),
                "pr_auc_mean": float(pr.mean()),
                "pr_auc_std": float(pr.std(ddof=1)) if pr.size > 1 else 0.0,
                "pr_auc_lo": float(pr_lo),
                "pr_auc_hi": float(pr_hi),
            }
            # v3 Phase E.2: per-fold capture rates alongside AUC. These columns
            # land in pop_fold_metrics_<pop>.csv when the v3 training script ran;
            # they may be absent on v2 metrics files.
            for cap_col in ("capture_at_1pct", "capture_at_5pct",
                            "capture_at_10pct", "enrichment_at_1pct"):
                if cap_col in grp.columns:
                    arr = grp[cap_col].to_numpy(dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        row[f"{cap_col}_mean"] = float("nan")
                        row[f"{cap_col}_std"] = float("nan")
                    else:
                        row[f"{cap_col}_mean"] = float(arr.mean())
                        row[f"{cap_col}_std"] = (
                            float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                        )
            rows.append(row)
    return pd.DataFrame(rows)


def _format_headline_section(headline: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if headline.empty:
        lines.append("_No headline metrics available — no model scores were present at validation time._")
        return lines
    lines.append("Top-k% capture (fraction of anchor districts captured in the top-k% scoring cells), "
                 "with bootstrap 95% CI and enrichment over random:")
    lines.append("")
    lines.append("| model | k% | capture | enrichment |")
    lines.append("| --- | --- | --- | --- |")
    for _, r in headline.iterrows():
        lines.append(
            f"| {r['model_label']} | {r['k_percent']:.0f}% | "
            f"{r['capture_point']:.2f} [{r['capture_lo']:.2f}, {r['capture_hi']:.2f}] | "
            f"{r['enrichment_point']:.1f}x [{r['enrichment_lo']:.1f}x, {r['enrichment_hi']:.1f}x] |"
        )
    return lines


def _format_per_fold_section(per_fold: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if per_fold.empty:
        lines.append("_No per-fold metrics files present — Phase 2 training has not been run, "
                     "or the fold-metrics CSVs are empty._")
        return lines
    lines.append("Per (population, model) over spatial-CV folds (`SpatialBlockCV(block_size_m=20_000)`); "
                 "CI95 is the 2.5/97.5 percentile across folds. `capture@5%` is the "
                 "v3 Phase E.2 per-fold capture rate: fraction of held-out positives "
                 "ranked in the top 5% of predicted scores.")
    lines.append("")
    has_cap5 = "capture_at_5pct_mean" in per_fold.columns
    if has_cap5:
        lines.append("| population | model | n_folds | roc_auc mean +/- std [CI95] | pr_auc mean +/- std [CI95] | capture@5% mean +/- std |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
    else:
        lines.append("| population | model | n_folds | roc_auc mean +/- std [CI95] | pr_auc mean +/- std [CI95] |")
        lines.append("| --- | --- | --- | --- | --- |")
    for _, r in per_fold.iterrows():
        base = (
            f"| {r['population']} | {r['model']} | {int(r['n_folds'])} | "
            f"{r['roc_auc_mean']:.3f} +/- {r['roc_auc_std']:.3f} "
            f"[{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}] | "
            f"{r['pr_auc_mean']:.3f} +/- {r['pr_auc_std']:.3f} "
            f"[{r['pr_auc_lo']:.3f}, {r['pr_auc_hi']:.3f}] |"
        )
        if has_cap5:
            cap_mean = r.get("capture_at_5pct_mean", float("nan"))
            cap_std = r.get("capture_at_5pct_std", float("nan"))
            if pd.notna(cap_mean) and pd.notna(cap_std):
                base += f" {cap_mean:.3f} +/- {cap_std:.3f} |"
            else:
                base += " _n/a_ |"
        lines.append(base)
    return lines


def _load_recipe_matches() -> dict[str, dict]:
    """Read recipe_match_<pop>.json (v3 Phase E.4) for every population present."""
    out: dict[str, dict] = {}
    for pop in POPULATIONS:
        p = OUT_DIR / f"recipe_match_{pop}.json"
        if not p.exists():
            continue
        try:
            out[pop] = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _format_recipe_section(recipes: dict[str, dict]) -> list[str]:
    """Markdown for the SHAP recipe-match section (v3 Phase E.4)."""
    lines: list[str] = []
    if not recipes:
        lines.append("_No `recipe_match_<pop>.json` files present; run "
                     "`scripts/northern_sierra_placer_rationale_250m.py` first._")
        return lines
    lines.append("Per-population SHAP top-5 features vs the geological recipe "
                 "(v3 Phase E.4 spec). Gate passes when at least 3 of the top-5 "
                 "features match the recipe list. The lithology family "
                 "(`litho_*` / `lithology_class_*`) counts once, not once per "
                 "one-hot dummy.")
    lines.append("")
    lines.append("| population | top-5 features | expected | matched | gate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for pop, rec in recipes.items():
        top_5 = ", ".join(rec.get("top_5", []))
        expected = ", ".join(rec.get("expected", []))
        n = rec.get("n_matched", 0)
        gate = "PASS" if rec.get("gate_passes", False) else "FAIL"
        lines.append(f"| {pop} | {top_5} | {expected} | {n}/5 | {gate} |")
    return lines


def _render_model_card(
    *,
    out_path: Path,
    pa_summary: dict[str, dict[str, float]],
    anchor_table: pd.DataFrame,
    comparison: pd.DataFrame,
    fused_present: bool,
    per_pop_metrics_files: dict[str, Path | None],
    headline: pd.DataFrame | None = None,
    per_fold: pd.DataFrame | None = None,
    recipes: dict[str, dict] | None = None,
) -> None:
    """Write the markdown model card. Brutally honest about what wasn't shipped."""
    lines: list[str] = []
    push = lines.append

    push(f"# Model card — `{REGION.slug}`")
    push("")
    push(f"Northern Sierra deep-gravel placer-Au prospectivity, 250 m, EPSG:4326. Deliverable: `prospectivity_placer_northern_sierra_250m_calibrated_4326.tif`.")
    push("")
    push("## AOI and grid")
    push("")
    push(f"- Bounding box (WGS84): {REGION.aoi.bbox}")
    push(f"- Compute CRS: `{REGION.working_crs}` (California Albers Equal Area)")
    push(f"- Deliverable CRS: `EPSG:4326`")
    push(f"- Resolution: 250 m, aligned cell-for-cell to the motherlode lode raster")
    push("")
    push("## Headline metrics")
    push("")
    for line in _format_headline_section(headline if headline is not None else pd.DataFrame()):
        push(line)
    push("")
    push("## Two-population architecture")
    push("")
    push("Sierra placer divides into Tertiary deep-gravel (paleochannels, mapped via hydraulic-pit polygons + Lindgren PP73 + the paleochannel-likelihood raster) and Quaternary modern-channel (modern drainage placers from MRDS + NHD position). Each population has its own classifier; the calibrated rasters fuse via per-cell `np.maximum` into the deliverable.")
    push("")
    push("## Features")
    push("")
    push("- Distance-downstream-from-lode along NHD (m); seeded from MRDS lode-Au only (placer dep_types excluded; leakage guard enforced by `features/hydrology.distance_downstream_from_lode`)")
    push("- Paleochannel-likelihood raster (REM + LRM + GeoMorphic Index composite from `features/paleochannel.py`)")
    push("- Hydraulic-pit proximity (m) (Hydraulic Mine Pits of California, DOI 10.5066/F7J38QMD)")
    push("- Stream Power Index banded membership (NZ-derived 3–6 band; Phase 1 only)")
    push("- Catchment Au, As, Sb (Hawkes dual-decay-length, fold-aware in Phase 2)")
    push("- Quaternary alluvium mask (CGS 2010 Geologic Map of California, FeatureServer)")
    push("- Topographic Wetness Index")
    push("- Geomorphon terrace mask (GRASS `r.geomorphon`)")
    push("- Slope (Horn central differences)")
    push("- Other base features inherited from `build_feature_frame`: elevation, TRI, magnetic, gravity, lithology one-hots, distance-to-fault, NGDB + NURE geochem aggregates")
    push("")
    push("**Dropped per Lawley audit discipline:** every `*_count_5km` and `*_has_data_5km` column.")
    push("")
    push("## Labels")
    push("")
    push("- Tertiary positives: Hydraulic Mine Pit polygon centroids (DOI 10.5066/F7J38QMD; 167 polygons)")
    push("- Quaternary positives: MRDS records with dep_type matching the placer regex `(placer|alluvial|stream.?placer|paleo.?placer|black.?sand|residual|eluvial)`")
    push("- Negatives: PU-learning framing only (absence-of-record is NOT treated as a hard negative)")
    push("- Anchor districts (held out from every training fold, PU bag, and calibration fold):")
    for d in ANCHOR_DISTRICTS:
        push(f"  - {d}")
    push("")
    push("## Cross-validation and calibration")
    push("")
    push("- Spatial-block CV: `SpatialBlockCV(block_size_m=20_000)` from `ai_minerals.model`")
    push("- Block size matches the Lawley-audit calibration; random KFold is not used anywhere in this build")
    push("- Stacking ensemble (RF + LightGBM → logistic-regression meta) uses the same `SpatialBlockCV` iterable as `cv`")
    push("- Calibration: `CalibratedClassifierCV(method='isotonic', cv=5)`; falls back to `'sigmoid'` (Platt) when a population has < 30 positives")
    push("- Hawkes catchment-geochem features are recomputed per fold using only the train-fold samples (`hawkes_dual_decay_catchment(..., fold_mask=...)`)")
    push("")
    push("## Anchor-district performance")
    push("")
    push("Per-district decile rank for each model in the within-AOI score distribution (0 = top decile = strongest signal):")
    push("")
    push(anchor_table.to_markdown(index=False))
    push("")
    push("## Cross-model comparison (population, model, top-decile-cutoff metrics)")
    push("")
    if comparison.empty:
        push("_No comparison metrics available — Phase 2 outputs were not present at validation time._")
    else:
        push(comparison.round(4).to_markdown(index=False))
    push("")
    push("## Success-rate / Prediction-Area curve")
    push("")
    push("![P-A curve](success_rate_curve_phase1_vs_phase2.png)")
    push("")
    if pa_summary:
        push("Area under each P-A curve (over anchor cells; higher = better targeting). "
             "Brackets are bootstrap 95% CIs over the positive set (n_resamples=2000):")
        for label, vals in pa_summary.items():
            lo = vals.get("auc_pa_lo", float("nan"))
            hi = vals.get("auc_pa_hi", float("nan"))
            push(f"- {label}: {vals['auc_pa']:.3f} [{lo:.3f}, {hi:.3f}]")
    push("")
    push("## Per-fold spatial-CV metrics")
    push("")
    for line in _format_per_fold_section(per_fold if per_fold is not None else pd.DataFrame()):
        push(line)
    push("")
    push("## SHAP recipe match (v3 Phase E.4)")
    push("")
    for line in _format_recipe_section(recipes if recipes is not None else {}):
        push(line)
    push("")
    push("## Deliverable")
    push("")
    if fused_present:
        push(f"`{OUT_DIR.name}/prospectivity_placer_northern_sierra_250m_calibrated_4326.tif` — Phase 2 per-cell-max fusion of the Tertiary and Quaternary calibrated rasters.")
    else:
        push(f"_Phase 2 fusion raster not present; shipping the Phase 1 knowledge-driven index as the deliverable (per the plan's 'ship Phase 1 if Phase 2 doesn't beat it' rule). Filename: `{OUT_DIR.name}/phase1_index_250m_4326.tif`._")
    push("")
    push("## Limitations")
    push("")
    push("- **Central-Sierra hydraulic-pit coverage is thin.** Sub-areas with no pit polygons rely entirely on the Quaternary branch + Lindgren proxy for Tertiary signal. The model card should note pit density sub-area-by-sub-area for any region added to the model.")
    push("- **Stream Power Index band (3–6) is NZ-derived.** Used in Phase 1 only; Phase 2 trees see raw SPI and learn the local optimum.")
    push("- **DEM/LiDAR coverage ceiling.** 3DEP 1 m where flown; 10 m fallback elsewhere. The paleochannel-likelihood raster degrades to LRM-only on 10 m DEMs.")
    push("- **Hawkes catchment-geochem and distance-downstream-from-lode** carry residual leakage risk. Mitigations: per-fold Hawkes recompute (`features/placer_geology.hawkes_dual_decay_catchment(..., fold_mask=...)`); strict lode-seed filter (`features/hydrology.distance_downstream_from_lode` asserts no placer dep_type). Unit test at `tests/test_distance_downstream_leakage.py`.")
    push("- **Isotonic calibration is brittle on tiny positive sets** (< 30 positives). Falls back to Platt scaling; ground-truth on the failover noted in `pop_fold_metrics_*.csv`.")
    push("- **NHD COMID joins at the AOI boundary** depend on the 25 km buffer in `data/nhdplus_hr.py::fetch`. Cells whose nearest reach extends beyond the buffer get NaN distance-downstream values.")
    push("- **NURE ICP-MS Au is semi-quantitative** (`Au_sq_ppm`); noisier than four-acid leach Au; treat as auxiliary signal, not primary.")
    push("")
    push("## Fold metrics")
    push("")
    for pop, p in per_pop_metrics_files.items():
        if p is None:
            push(f"- {pop}: _no fold metrics file_")
        else:
            push(f"- {pop}: `{p.relative_to(OUT_DIR.parent)}`")
    push("")
    push("## Provenance")
    push("")
    push("- Region module: `src/ai_minerals/regions/northern_sierra_placer.py`")
    push("- Anchor districts: `src/ai_minerals/regions/_northern_sierra_anchors.py`")
    push("- Phase 1 scorer: `src/ai_minerals/scorers/usgs_alaska_placer.py`")
    push("- Phase 2 training: `scripts/northern_sierra_placer_train_predict_250m.py`")
    push("- Fusion: `scripts/northern_sierra_placer_calibrate_and_fuse.py`")
    push("- This validation script: `scripts/northern_sierra_placer_validation.py`")
    push("- Plan: `~/.claude/plans/hazy-humming-lynx.md`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    print(f"==> Loading Phase 1: {IN_PHASE1}")
    if not IN_PHASE1.exists():
        print(f"ERROR: {IN_PHASE1} missing. Run Phase C first.", file=sys.stderr)
        return 2
    phase1 = pd.read_parquet(IN_PHASE1)
    print(f"    {len(phase1):,} cells, phase1_score finite={int(phase1['phase1_score'].notna().sum()):,}")

    fused = _load_optional(IN_FUSED)
    if fused is None:
        print(f"WARN: {IN_FUSED} missing. Phase 2 likely not run; model card will declare Phase 1 as deliverable.")

    per_pop: dict[str, dict[str, pd.DataFrame | None]] = {}
    per_pop_metrics: dict[str, Path | None] = {}
    for pop in POPULATIONS:
        preds = _load_optional(OUT_DIR / f"pop_predictions_{pop}_250m.parquet")
        cal = _load_optional(OUT_DIR / f"pop_calibrated_{pop}_250m.parquet")
        metrics = OUT_DIR / f"pop_fold_metrics_{pop}.csv"
        per_pop[pop] = {"predictions": preds, "calibrated": cal}
        per_pop_metrics[pop] = metrics if metrics.exists() else None
        if preds is None and cal is None:
            print(f"WARN: no Phase 2 outputs for {pop}.")

    df = _build_score_table(phase1, fused, per_pop)
    anchor_idxs = _anchor_cell_indices(df)

    print(f"==> Plotting P-A curves -> {OUT_PA_CURVE}")
    pa_summary = _plot_pa_curves(df, anchor_idxs, OUT_PA_CURVE)
    for label, vals in pa_summary.items():
        print(f"    {label}: auc_pa={vals['auc_pa']:.3f}")

    print(f"==> Building anchor-district table -> {OUT_ANCHOR_TABLE}")
    anchor_table = _build_anchor_table(df, anchor_idxs)
    anchor_table.to_csv(OUT_ANCHOR_TABLE, index=False)

    # Cross-model comparison needs per-pop labels. Build them by treating
    # anchor districts as the positive set (they are the known placer
    # producers held out of training).
    print(f"==> Building cross-model comparison -> {OUT_COMPARISON}")
    per_pop_labels = {}
    for pop in POPULATIONS:
        y = pd.Series(0, index=df.index, dtype=int)
        y.loc[anchor_idxs.values] = 1
        per_pop_labels[pop] = y
    comparison = _build_comparison_table(df, per_pop_labels)
    comparison.to_csv(OUT_COMPARISON, index=False)
    print(f"    {len(comparison)} comparison rows")

    print(f"==> Building headline capture / enrichment metrics -> {OUT_HEADLINE}")
    headline = _build_headline_metrics(df, anchor_idxs)
    headline.to_csv(OUT_HEADLINE, index=False)
    print(f"    {len(headline)} headline rows")

    print("==> Aggregating per-fold spatial-CV metrics")
    per_fold = _aggregate_fold_metrics(per_pop_metrics)
    print(f"    {len(per_fold)} (population, model) aggregates")

    print("==> Loading SHAP recipe-match JSONs (v3 Phase E.4)")
    recipes = _load_recipe_matches()
    for pop, rec in recipes.items():
        gate = "PASS" if rec.get("gate_passes", False) else "FAIL"
        print(f"    {pop}: {rec.get('n_matched', 0)}/5 ({gate})")
    if not recipes:
        print("    no recipe_match_<pop>.json files found")

    print(f"==> Rendering model card -> {OUT_CARD}")
    _render_model_card(
        out_path=OUT_CARD,
        pa_summary=pa_summary,
        anchor_table=anchor_table,
        comparison=comparison,
        fused_present=fused is not None,
        per_pop_metrics_files=per_pop_metrics,
        headline=headline,
        per_fold=per_fold,
        recipes=recipes,
    )

    print(json.dumps({
        "pa_summary": pa_summary,
        "anchor_rows": len(anchor_table),
        "comparison_rows": len(comparison),
        "headline_rows": len(headline),
        "per_fold_rows": len(per_fold),
        "recipe_match_pops": list(recipes.keys()),
        "model_card": str(OUT_CARD),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
