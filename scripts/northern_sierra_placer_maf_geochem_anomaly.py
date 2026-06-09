"""EW1: Masked Autoregressive Flow anomaly detection on Au pathfinder geochem.

Direct port of Scheidt, Mathieu, Yin, Wang, Caers (2024) ([DOI 10.1007/s11053-024-10409-2](https://doi.org/10.1007/s11053-024-10409-2))
to the Northern Sierra placer AOI, with Au pathfinder elements (As, Sb,
Bi, W, Cu) substituted for the original paper's LCT-pegmatite indicator
chemistry.

Pipeline:

1. Load USGS NGDB stream-sediment samples for the Northern Sierra AOI.
2. Filter to rows with complete-case measurements across the 5 pathfinder
   elements. Geochemical assays are right-skewed; log-transform before
   fitting.
3. Fit a Masked Autoregressive Flow on the log-transformed feature
   matrix (model_maf_anomaly.fit_maf_anomaly).
4. Score every sample under the fitted flow. Negative-log-likelihood is
   the anomaly score; high score = candidate pathfinder anomaly.
5. Write a sidecar parquet with sample coords + anomaly scores +
   per-element values, ready for downstream rasterization in v3.5.LF.
6. Produce a sanity-check figure: anomaly score map.

Outputs:

- ``data/derived/northern_sierra_placer/maf_geochem_anomaly_scores.parquet``
- ``data/derived/northern_sierra_placer/fig_maf_anomaly_map.png``

Honest caveat: NGDB Northern Sierra has ~1900 samples with ~600
complete-case rows across the 5 pathfinder elements. Scheidt 2024 uses
~10k samples with 30+ elements. This is the small-n, low-d regime; the
fitted MAF density is correspondingly noisier than the paper's. The
anomaly ranking is still useful as a relative-ranking signal but
absolute scores are less stable.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_minerals.model_maf_anomaly import fit_maf_anomaly


NGDB_PATH = Path("data/raw/ngdb/ngdb_sediment_northern_sierra.gpkg")
OUT_DIR = Path("data/derived/northern_sierra_placer")
OUT_PARQUET = OUT_DIR / "maf_geochem_anomaly_scores.parquet"
OUT_FIG = OUT_DIR / "fig_maf_anomaly_map.png"

# Au pathfinder elements per Goldfarb 2013 and the Cox & Singer Au-bearing
# placer model. Cu is included because Au often shows weak Cu association
# in orogenic-Au; As, Sb, Bi, W are the classical low-temp pathfinders.
PATHFINDER_COLS = ["As_ppm", "Sb_ppm", "Bi_ppm", "W_ppm", "Cu_ppm"]

# MAF hyperparameters, sized for the small-n / 5-feature regime
N_FLOWS = 4
HIDDEN_UNITS = 32
N_ITER = 2000
LR = 1e-3


def main() -> None:
    print(f"[maf-geochem] Loading NGDB samples from {NGDB_PATH}", flush=True)
    df = gpd.read_file(NGDB_PATH)
    print(f"[maf-geochem]   {len(df):,} total samples (CRS: {df.crs})", flush=True)
    # NGDB GPKG is in EPSG:3310 California Albers (metres); reproject to
    # WGS84 for lat/lon-labeled output that downstream consumers
    # (placer.qmd visualization, goldbug per-claim sampling) can use
    # directly.
    df = df.to_crs("EPSG:4326")

    # Filter to complete-case pathfinder rows
    mask = df[PATHFINDER_COLS].notna().all(axis=1)
    df_cc = df[mask].copy().reset_index(drop=True)
    print(f"[maf-geochem]   {len(df_cc):,} complete-case rows across {PATHFINDER_COLS}", flush=True)
    if len(df_cc) < 100:
        raise RuntimeError(
            f"Only {len(df_cc)} complete-case rows; MAF needs more"
        )

    # Log-transform (add small epsilon to handle zeros; the geochem
    # detection limits are non-zero, so this just nudges below-detection
    # values away from log(0))
    eps = 1e-3
    X = df_cc[PATHFINDER_COLS].to_numpy()
    X = np.log10(X + eps)
    print(f"[maf-geochem]   feature range after log10:", flush=True)
    for i, c in enumerate(PATHFINDER_COLS):
        print(f"    {c:>10s}: {X[:, i].min():+.3f} … {X[:, i].max():+.3f}", flush=True)

    # Train MAF
    print(f"[maf-geochem] Fitting MAF: n_flows={N_FLOWS}, hidden={HIDDEN_UNITS}, n_iter={N_ITER}", flush=True)
    model = fit_maf_anomaly(
        X,
        feature_names=PATHFINDER_COLS,
        n_flows=N_FLOWS,
        hidden_units=HIDDEN_UNITS,
        n_iter=N_ITER,
        learning_rate=LR,
        verbose=True,
    )

    # Score every sample (training-set scoring is normal practice for
    # density-based anomaly detection; the in-sample score still
    # discriminates outliers — see Scheidt 2024 section 3.4)
    print(f"[maf-geochem] Scoring {len(df_cc):,} samples...", flush=True)
    log_density = model.score_samples(X)
    anomaly = -log_density  # higher = more anomalous

    df_cc["maf_log_density"] = log_density
    df_cc["maf_anomaly_score"] = anomaly

    # Clip outlier scores so the visualization doesn't get dominated by
    # one freak sample
    p99 = float(np.quantile(anomaly, 0.99))
    print(f"[maf-geochem] Anomaly score: median={np.median(anomaly):.3f}, "
          f"p90={np.quantile(anomaly, 0.90):.3f}, "
          f"p99={p99:.3f}, max={anomaly.max():.3f}", flush=True)
    df_cc["maf_anomaly_score_clipped"] = np.minimum(anomaly, p99)

    # Save sidecar parquet (drop geometry for parquet portability;
    # keep lat/lon columns from the centroid)
    out = pd.DataFrame({
        "lon": df_cc.geometry.x,
        "lat": df_cc.geometry.y,
        "lab_id": df_cc["lab_id"],
        **{c: df_cc[c] for c in PATHFINDER_COLS},
        "maf_log_density": df_cc["maf_log_density"],
        "maf_anomaly_score": df_cc["maf_anomaly_score"],
        "maf_anomaly_score_clipped": df_cc["maf_anomaly_score_clipped"],
    })
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"[maf-geochem] Wrote {OUT_PARQUET}  ({len(out):,} rows)", flush=True)

    # Sanity-check figure: anomaly score on a map
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(
        out["lon"], out["lat"],
        c=out["maf_anomaly_score_clipped"],
        s=12, cmap="magma", alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="MAF anomaly score (clipped at p99)")
    # Highlight the top-decile anomalies
    top_decile = out["maf_anomaly_score"] >= out["maf_anomaly_score"].quantile(0.90)
    ax.scatter(
        out.loc[top_decile, "lon"], out.loc[top_decile, "lat"],
        marker="o", facecolors="none", edgecolors="white", linewidths=0.8,
        s=80, label=f"Top decile (n={top_decile.sum()})",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Au pathfinder MAF anomaly — NGDB stream sediment, Northern Sierra "
        f"({len(out):,} samples × {len(PATHFINDER_COLS)} elements)\n"
        f"Following Scheidt et al. 2024 (DOI 10.1007/s11053-024-10409-2)"
    )
    ax.legend(loc="best")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"[maf-geochem] Wrote {OUT_FIG}", flush=True)

    print("[maf-geochem] Done.", flush=True)


if __name__ == "__main__":
    main()
