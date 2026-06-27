"""Round 4A: does the Groves splay/intersection gain clear zero in the EAST once
the eastern low-sulfide Au-quartz (36a) label set is expanded with disciplined typing?

Round 3 found the peninsula-wide structural control generalized (mapped-cell gain
clears zero), but the EASTERN subset specifically was underpowered: 9 eastern 36a
positives, struct_groves-over-base east gain +0.104, 95% CI [-0.073, +0.282] still
spanning zero. This closes that by expanding the eastern label set under the SAME
typed-label discipline (no occurrence-derived features, which was the F3 leak), then
re-grading the splay/intersection gain on the larger eastern set with a bootstrap CI.

Label expansion (disciplined typing, NOT a loosened keyword net):
  * keep every ARDF model_code 36a (Cox-Singer low-sulfide Au-quartz): the typed core.
  * recover gold-bearing-quartz-vein occurrences ARDF left uncoded (model_code None)
    whose deposit-model text reads as low-sulfide Au-quartz, EXCLUDING anything
    antimony/stibnite/scheelite/tungsten/skarn/polymetallic/galena/base-metal/
    fluorite/calcite/placer/replacement. On the eastern Seward Peninsula this recovers
    exactly the two Council-area Crooked Creek lodes ("Gold-bearing quartz veins in
    schistose marble"); the 22c polymetallic and Sb/W/galena veins are rejected.
  * MRDS adds nothing: the only local Alaska MRDS extract is Interior/eastern AK,
    with zero Seward Peninsula coverage.
  * The named Big Hurrah and Casadepaga lode sites sit WEST of the central-district
    east edge (x = -473300), i.e. already inside the central training set, so they
    are not eastern hold-out positives. Otter Creek (a 36a at x = -394085) sits
    beyond the round-3 wider-grid east edge; it is a single positive and is reported
    as an available extension, not folded into the comparable test.

Result: eastern positives 9 -> 11, all on RI 2024-7 / SIM 3131 mapped ground. Same
wider grid, structure bands and F1 leak-guarded CV as round 3, so the only thing
that moves is the eastern label count. Either outcome is reportable.

Run: PYTHONPATH=src .venv/bin/python -m scripts.nome_placer.lode_eastern_splay_confirm
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from scripts.nome_placer.lode_peninsula_generalization import (
    ARDF_FULL, CENTRAL_RIGHT_EDGE, CENTRAL_TPL, STRUCT_DIR,
    build_features, build_wider_template, grade,
)

OUT_DIR = Path("data/derived/nome_placer/lode_eastern_splay_confirm")

# Disciplined low-sulfide Au-quartz typing. The recovery clause only fires on
# records ARDF left uncoded (or coded non-36a) whose text is unambiguously a
# gold-bearing quartz vein; the disqualifier clause drops every other vein family.
GOLDQ_RE = ("gold-bearing quartz|auriferous quartz|au-quartz|gold quartz|"
            "gold.bearing.{0,8}quartz|low.sulfide.{0,12}quartz")
DISQUALIFY_RE = ("stibnite|antimony|scheelite|tungsten|skarn|polymetallic|galena|"
                 "base.metal|massive sulfide|sulfide-rich|fluorite|calcite|placer|"
                 "porphyry|greisen|replacement|\\btin\\b|\\bsb\\b|\\bw\\b")


def _ardf_typed() -> tuple[gpd.GeoDataFrame, pd.Series, pd.Series]:
    """Statewide ARDF (EPSG:3338) plus the 36a and disciplined-recovery masks."""
    ardf = gpd.read_file(ARDF_FULL).to_crs("EPSG:3338")
    mc = ardf.get("model_code", pd.Series([""] * len(ardf))).astype(str).str.lower()
    txt = (ardf.get("dep_model", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("geol_desc", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("site_type", pd.Series([""] * len(ardf))).astype(str) + " "
           + ardf.get("ore", pd.Series([""] * len(ardf))).astype(str)).str.lower()
    is36a = mc.str.startswith("36a")
    recover = (~is36a
               & txt.str.contains(GOLDQ_RE, regex=True, na=False)
               & ~txt.str.contains(DISQUALIFY_RE, regex=True, na=False))
    return ardf, is36a, recover


def disciplined_lode_positives(bounds):
    """(px, py) for 36a + disciplined Au-quartz recovery, clipped to ``bounds``."""
    ardf, is36a, recover = _ardf_typed()
    sel = ardf[is36a | recover].cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    return sel.geometry.x.to_numpy(), sel.geometry.y.to_numpy()


def label_audit(bounds) -> dict:
    """Provenance of the disciplined positives inside ``bounds`` (and east of the
    central edge), so the report can name what the recovery clause added."""
    ardf, is36a, recover = _ardf_typed()
    box = ardf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    bi, br = is36a[box.index], recover[box.index]
    east = box.geometry.x > CENTRAL_RIGHT_EDGE
    recovered_east = box[br & east]
    return {
        "n_36a": int(bi.sum()),
        "n_recovered": int(br.sum()),
        "n_36a_east": int((bi & east).sum()),
        "n_recovered_east": int((br & east).sum()),
        "recovered_east_sites": [
            {"site": str(r.site), "model_code": str(r.model_code),
             "x": round(float(r.geometry.x)), "y": round(float(r.geometry.y)),
             "dep_model": str(r.dep_model)[:70]}
            for r in recovered_east.itertuples()
        ],
    }


def east_verdict(arms: dict) -> dict:
    g = arms["struct_groves"]
    be = g.get("bootstrap_east") or {}
    ci = be.get("ci95")
    clears = bool(ci is not None and ci[0] is not None and ci[0] > 0)
    return {
        "auc_east": g.get("auc_east"),
        "d_auc_east": g.get("d_auc_east"),
        "east_ci95": ci,
        "east_p_gt_0": be.get("p_gt_0"),
        "east_n_pos": be.get("n_pos"),
        "clears_zero_in_east": clears,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wider_tpl = build_wider_template(STRUCT_DIR / "_wider_grid_template_3338.tif")

    with rasterio.open(wider_tpl) as ds:
        wb = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    with rasterio.open(CENTRAL_TPL) as ds:
        cb = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

    audit_wider = label_audit(wb)
    print("label audit (wider grid):", json.dumps(audit_wider, indent=2))

    print("\nbuilding features with disciplined 36a+Au-quartz labels (wider, then central) ...")
    wider = build_features("wider_100m", wider_tpl, disciplined_lode_positives(wb))
    central = build_features("central_100m", CENTRAL_TPL, disciplined_lode_positives(cb))

    out = {
        "question": "Does the Groves splay/intersection gain over a lithology+fault base "
                    "clear zero in the EASTERN Seward Peninsula once the 36a label set is "
                    "expanded with disciplined typing, under the F1 leak-guarded CV?",
        "round3_baseline": {
            "eastern_36a_positives": 9,
            "east_d_auc": 0.104,
            "east_ci95": [-0.0734, 0.2824],
            "east_p_gt_0": 0.872,
            "note": "round-3 east CI still spanned zero (underpowered, 9 positives)",
        },
        "label_discipline": {
            "typed_core": "ARDF model_code startswith '36a' (Cox-Singer low-sulfide Au-quartz)",
            "recovery": "uncoded gold-bearing-quartz-vein text, EXCLUDING Sb/stibnite, "
                        "scheelite/W, skarn, polymetallic/22c, galena, base-metal, "
                        "fluorite, calcite, replacement, placer",
            "recovered_east": audit_wider["recovered_east_sites"],
            "mrds": "no local Seward Peninsula MRDS coverage; 0 added",
            "named_districts": "Big Hurrah / Casadepaga lode sites are WEST of x=-473300 "
                               "(already central); Otter Creek 36a (x=-394085) is beyond "
                               "the wider-grid east edge, reported as +1 available extension",
        },
        "design": {
            "grid": "round-3 wider grid + cached SIM 3131 + RI 2024-7 structure bands "
                    "(identical to round 3 so only the eastern label count moves)",
            "base": "SIM 3131 geology one-hot + distance-to-any-regional-fault (terrain-free)",
            "struct_groves": "base + dist_fault_intersection + dist_splay + dist_fold_hinge "
                             "+ carbonaceous_host",
            "east_definition": f"cells with x > {CENTRAL_RIGHT_EDGE:.0f}",
            "cv": "F1 leak_guarded contiguous folds, residual-variogram block sizing, "
                  "1 km dead-zone, seed 42, n_boot 2000",
        },
        "wider": {"bounds_3338": [round(v) for v in wb], **grade("wider", wider)},
        "central_matched": {"bounds_3338": [round(v) for v in cb], **grade("central", central)},
    }
    out["verdict"] = {
        "eastern_positives_disciplined": out["wider"]["n_pos_east"],
        "eastern_positives_round3": 9,
        **east_verdict(out["wider"]["arms"]),
        "wider_mapped_d_auc": out["wider"]["arms"]["struct_groves"].get("d_auc_mapped"),
        "wider_mapped_ci95": (out["wider"]["arms"]["struct_groves"].get("bootstrap_mapped") or {}).get("ci95"),
    }

    (OUT_DIR / "lode_eastern_splay_confirm.json").write_text(json.dumps(out, indent=2, default=str))
    with (OUT_DIR / "lode_eastern_splay_confirm.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid", "arm", "n_features", "auc_full", "auc_mapped", "d_auc_mapped",
                    "auc_east", "d_auc_east", "east_ci95_lo", "east_ci95_hi", "east_p_gt_0", "east_n_pos"])
        for grid in ("wider", "central_matched"):
            for arm, r in out[grid]["arms"].items():
                be = r.get("bootstrap_east") or {}
                eci = be.get("ci95") or [None, None]
                w.writerow([grid, arm, r.get("n_features"), r.get("auc_full"), r.get("auc_mapped"),
                            r.get("d_auc_mapped"), r.get("auc_east"), r.get("d_auc_east"),
                            eci[0], eci[1], be.get("p_gt_0"), be.get("n_pos")])

    print("\nVERDICT:", json.dumps(out["verdict"], indent=2, default=str))
    print(f"wrote {OUT_DIR / 'lode_eastern_splay_confirm.json'}")


if __name__ == "__main__":
    main()
