"""Digitize the Albion fault (documented georeference) and test dist-to-Albion.

The Albion fault is the NE-trending structure that hosts the Rock Creek sheeted-
vein deposit (Otto, Piekenbrock & Odden 2009, Econ. Geol. 104:945). It is a
deposit-scale structure absent from the district GeMS (PDF 94-39), so the round-2
lode model never carried it. Round-2 deferred it because pixel-tracing a paper
figure unattended is error-prone. This builds it instead from the paper's
EXPLICIT quantitative description plus an independently-coordinated anchor, so the
geometry is documented, not guessed:

  control point   ARDF occurrence "Albion" (model_code 36a), EPSG:3338
                  (-542902, 1675320). Cross-checked against Otto et al. 2009
                  Fig. 3 (UTM Zone 3 NAD83): the anchor converts to UTM3N
                  N=7,166,061, landing on the figure's top northing tick
                  (7166000 N). The control point sits on the paper's own grid.
  strike          azimuth 045 deg ("strikes north 45 deg east", Otto 2009 p.951).
                  At Nome (lon -165.4, ~the UTM zone-3 central meridian) the grid
                  convergence is -0.37 deg, so true bearing 045 ~ UTM3 grid 045.
  extent          20 km NE ("the structure continues to the northeast for about
                  20 km", Otto 2009 p.951) + 3 km SW of the anchor through the
                  deposit (the SW continuation is shorter and less constrained;
                  3 km is a conservative documented choice, flagged as such).

The trace is built in UTM Zone 3 NAD83 (EPSG:26903), where bearing maps to grid
azimuth with negligible convergence, then reprojected to EPSG:3338. Control points
and provenance are written to albion_fault_control.json next to the geometry.

Then dist_to_albion is added on top of the round-2 winner (struct_groves, the
0.712 arm) and graded under the F1 leak-guarded CV on the typed 36a labels, with
a paired bootstrap on the marginal delta. A null is a fine result: the Albion
fault is parallel and adjacent to the Anvil fault, which the NE-oriented bands
already carry.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_albion_cv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, mapping

from ai_minerals.data import nome_structure as ns
from ai_minerals.spatial_cv import default_rf_factory, spatial_cv_oof, subset_auc
from scripts.nome_placer.lode_structure_sharpen_cv import (
    RNG, TEMPLATE, ensure_structure_bands, load_base, lode_positives,
)
from scripts.nome_placer.newlayers_bootstrap import N_BOOT, boot_delta
from scripts.nome_placer.newlayers_geophys_rebaseline import make_cv, sample_bands

OUT_DIR = Path("data/derived/nome_geophys/albion_fault")
RES_OUT = Path("data/derived/nome_placer/lode_albion")

# Documented control parameters (see module docstring for the sources).
ANCHOR_3338 = (-542902.0, 1675320.0)   # ARDF "Albion" (36a), EPSG:3338
STRIKE_DEG = 45.0                      # Otto 2009 p.951 "strikes north 45 deg east"
EXTENT_NE_M = 20000.0                  # Otto 2009 p.951 "about 20 km" NE
EXTENT_SW_M = 3000.0                   # conservative SW continuation through the deposit


def build_albion_trace() -> LineString:
    """Construct the documented Albion fault LineString in EPSG:3338 + write provenance."""
    to_utm = Transformer.from_crs("EPSG:3338", "EPSG:26903", always_xy=True)
    to_3338 = Transformer.from_crs("EPSG:26903", "EPSG:3338", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)
    e0, n0 = to_utm.transform(*ANCHOR_3338)
    th = np.radians(STRIKE_DEG)
    de, dn = np.sin(th), np.cos(th)  # NE unit vector in grid metres
    ne = (e0 + EXTENT_NE_M * de, n0 + EXTENT_NE_M * dn)
    sw = (e0 - EXTENT_SW_M * de, n0 - EXTENT_SW_M * dn)
    pts_utm = [sw, (e0, n0), ne]
    pts_3338 = [to_3338.transform(e, n) for e, n in pts_utm]
    line = LineString(pts_3338)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = {
        "feature": "Albion fault (NE-trending host of the Rock Creek deposit)",
        "method": "documented analytic construction from Otto, Piekenbrock & Odden 2009 "
                  "(Econ. Geol. v.104, p.945-960, doi 10.2113/econgeo.104.7.945); NOT a pixel trace",
        "control_point": {
            "source": "ARDF occurrence 'Albion' (model_code 36a)",
            "epsg_3338": list(ANCHOR_3338),
            "utm_zone3_nad83": [round(e0), round(n0)],
            "lonlat": [round(v, 4) for v in to_ll.transform(*ANCHOR_3338)],
            "cross_check": "Otto 2009 Fig.3 is UTM Zone 3 NAD83; anchor N=7,166,061 lands on "
                           "the figure's 7166000 N tick",
        },
        "strike_deg": STRIKE_DEG, "strike_source": "Otto 2009 p.951 'strikes north 45 deg east'",
        "extent_ne_m": EXTENT_NE_M, "extent_ne_source": "Otto 2009 p.951 'continues to the northeast for about 20 km'",
        "extent_sw_m": EXTENT_SW_M, "extent_sw_note": "conservative; SW continuation less constrained in the paper",
        "grid_convergence_deg": -0.371, "convergence_note": "Nome ~ UTM zone-3 central meridian; bearing 045 ~ grid 045",
        "endpoints_3338": {"sw": [round(v) for v in pts_3338[0]], "ne": [round(v) for v in pts_3338[2]]},
        "limitations": "straight-line idealization at the stated strike; real trace may bend; "
                       "SW extent approximate. A documented first geometry, not a surveyed trace.",
    }
    (OUT_DIR / "albion_fault_control.json").write_text(json.dumps(control, indent=2))
    (OUT_DIR / "albion_fault_3338.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3338"}},
        "features": [{"type": "Feature", "properties": {"name": "Albion fault (documented)"},
                      "geometry": mapping(line)}]}, indent=2))
    print(f"Albion trace: SW {control['endpoints_3338']['sw']} -> NE {control['endpoints_3338']['ne']} "
          f"(length {line.length / 1000:.1f} km)")
    return line


def build_dist_to_albion(line: LineString) -> Path:
    dist, prof = ns._dist_to([line], TEMPLATE)
    return ns._write(dist, prof, OUT_DIR / "dist_to_albion.tif")


def main() -> None:
    RES_OUT.mkdir(parents=True, exist_ok=True)
    line = build_albion_trace()
    albion_path = build_dist_to_albion(line)

    sp = ensure_structure_bands()
    with rasterio.open(TEMPLATE) as ds:
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    px, py = lode_positives(bounds, "36a")
    base_df, y, coords, in_box = load_base((px, py))
    X_base = base_df.to_numpy(np.float32)
    ex, ny = coords[:, 0], coords[:, 1]

    st = sample_bands(sp, ex, ny)
    mapped = (st["gems_extent"] > 0.5).to_numpy()
    add = st.fillna(-999.0)
    alb = sample_bands({"dist_to_albion": albion_path}, ex, ny)["dist_to_albion"].fillna(-999.0).to_numpy(np.float32)
    cv, cvmeta = make_cv(X_base, y, coords)

    groves = ns.GROVES_BANDS + ["dist_fold_hinge", "carbonaceous_host"]

    def oof(struct_cols, with_albion):
        cols = [add[c].to_numpy(np.float32) for c in struct_cols]
        if with_albion:
            cols.append(alb)
        XX = X_base if not cols else np.column_stack([X_base] + cols)
        return spatial_cv_oof(XX.astype(np.float32), y, coords, cv, model_factory=default_rf_factory(seed=RNG))

    oof_base = oof([], False)
    oof_groves = oof(groves, False)
    oof_groves_alb = oof(groves, True)

    def row(o):
        return {"auc_gems": subset_auc(y, o, mask=mapped),
                "auc_placer_core": subset_auc(y, o, mask=in_box), "auc_full": subset_auc(y, o)}

    res = {"base": row(oof_base), "struct_groves": row(oof_groves), "struct_groves_albion": row(oof_groves_alb)}
    # marginal of dist_to_albion ON TOP of struct_groves (the "does it help" test)
    boot_marg = boot_delta(y, oof_groves, oof_groves_alb, mapped)
    # struct_groves_albion vs base, for the absolute
    boot_abs = boot_delta(y, oof_base, oof_groves_alb, mapped)

    n_pos_gems = int(y[mapped].sum())
    print(f"\n36a n={len(y)} pos={int(y.sum())} (gems={n_pos_gems})  n_boot={N_BOOT}")
    for arm, r in res.items():
        print(f"  {arm:22s} auc_gems={r['auc_gems']:.4f}  full={r['auc_full']:.4f}")
    print(f"\n  dist_to_albion marginal over struct_groves (auc_gems): "
          f"d={boot_marg['point']} 95%CI={boot_marg['ci95']} P(d>0)={boot_marg['p_gt_0']} (n_pos={boot_marg['n_pos']})")

    out = {
        "question": "Does a documented dist-to-Albion-fault feature help the round-2 lode "
                    "winner (struct_groves, auc_gems 0.712) under F1 leak-guarded CV?",
        "albion_construction": json.loads((OUT_DIR / "albion_fault_control.json").read_text()),
        "scheme": {**cvmeta, "estimator": "RandomForest(300, balanced, seed=42)", "n_boot": N_BOOT},
        "n": int(len(y)), "n_pos": int(y.sum()), "n_pos_gems_mapped": n_pos_gems,
        "arms": res,
        "albion_marginal_over_struct_groves": {"subset": "gems", **boot_marg},
        "struct_groves_albion_vs_base": {"subset": "gems", **boot_abs},
    }
    (RES_OUT / "lode_albion.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RES_OUT / 'lode_albion.json'}")
    print(f"wrote {OUT_DIR / 'albion_fault_control.json'} + albion_fault_3338.geojson")


if __name__ == "__main__":
    main()
