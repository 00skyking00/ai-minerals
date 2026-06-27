# Tuck georef v2: the real-vs-synthetic question is settled; the offset and warp block on served-frame corner picks

Verdict first. I can settle objectively which GCP sets are real, which resolves the
question you flagged ("I can't tell which set is real without guessing"). I cannot
measure the current overlay offset or fit a real-corner warp from the artifacts on
disk, because no real pixel picks exist in the *served* raster's pixel frame. The
offset and the warp both need the 229 true corners picked directly on the served
Tuck sheets, a Sky-in-the-loop digitising pass. Evidence and the recommended path
below.

## 1. GCP-set audit (settles "which set is real")

The discriminator is the RMS residual of the best affine pixel-to-world fit: a
back-projected (synthetic) set reproduces every pair to ~0 m, so re-fitting it
returns the current georef and corrects nothing; a hand-picked set leaves the
sheet's non-affine distortion as a non-zero residual.

| GCP set | n | affine RMS | class |
|---|---|---|---|
| `v1p5_v2_gcp_candidates` (a/b/b1/c, contours, general) | 130-215 | 0.0 m | **synthetic** |
| `v1p5_v4_hammon_synthetic` (all) | 40-49 | 0.0 m | **synthetic** |
| `v1p5_v5_corrected` (all) | 40-49 | 0.0 m | **synthetic** |
| `diagnostic_mosaic/atlas_v4_diagnostic`, `…_pixelspace` | 224 | 0.0 m | **synthetic** |
| `v1p5_v2_refined_gcps/map_b` | 12 | 2.4 m | **real** |
| `v1p5_v2_refined_gcps/map_b1` | 10 | 2.1 m | **real** |
| `v1p5_v2_refined_gcps/map_a` | 5 | 2.3 m | real (few) |
| `v1p5_v2_refined_gcps/map_c` | 7 | 68.6 m | real (sparse, high distortion) |
| `v1p5_v2_refined_gcps/map_d` | 4 | 15.0 m | real (too few for order-2) |
| `v1p5_v2_refined_gcps/map_e` | 5 | 1080 m | real (bad picks or extreme distortion) |
| `diagnostic_mosaic/atlas_v4_diagnostic_manual_` | 228 | (multi-sheet) | real, atlas frame |

So the confusion resolves: `v2_gcp_candidates` is synthetic, `v2_refined_gcps` is
the real per-sheet hand-pick set, and the 228-point `diagnostic_manual` set (one
per true corner) is real but picked in mosaic-atlas pixel space, not per sheet. The
affine residuals even track your visual read: the central block is near-affine
(map_b 2.4 m), the outer/coast sheets are not (map_c 69 m, map_e 1080 m).

## 2. Why the current offset is not measurable from these artifacts

- The 229 true corners are world-only (no Tuck pixel coordinate).
- The served georef is confirmed: the ai-minerals `overlays_v1p5` map_b transform is
  identical to goldbug `data/historical/tuck1942_v1p5` map_b, to the digit.
- The only real per-sheet picks that sit on the true corners are map_b and map_b1
  (the central Dry Creek / Bear Cub block; 12/12 and 10/10 picks within 25 m of a
  corner). Applying the **served** affine to those pick pixels lands **5.6 km** off
  the true corners (bias to the SE), against the **tens of metres N-NE** you measured
  visually. A kilometre reconciliation means those picks were made on a different
  working raster, not the served sheet, so they cannot measure the served offset.
- The outer-sheet real picks (a/c/d/e) are on creek mouths and grid lines, not the
  229 corners (nearest corner 1.6 to 15 km away).

There is no served-frame pixel-to-truth pairing on disk. Your visual tens-of-metres
N-NE stands as the best estimate; I cannot sharpen it numerically without that
pairing, and I am not going to ship the 5.6 km reconciliation as if it were the
offset.

## 3. Why the warp is blocked

Two reasons, either sufficient. (a) The frame mismatch above: the real picks are not
in the served sheet frame. (b) Even ignoring that, the real picks cover only the
central block (map_b's 12 picks span a ~2 km box of a 9.5 km sheet), so a thin-plate
spline or order-2 polynomial fit from them extrapolates wildly over the rest of the
sheet. A full-sheet warp needs picks spanning each full sheet corner-to-corner. The
synthetic sets cannot substitute: they re-encode the current affine (residual 0) and
fix nothing, exactly as you noted.

## 4. Recommended path (the unblock)

One Sky-in-the-loop digitising pass. In QGIS Georeferencer, load each *served* Tuck
sheet and pick the true claim corners on the raster, snapping the map side to
bearcub's `nome_control_points.geojson` (229 corners), 12 to 20 per sheet spread
corner-to-corner. That yields real, served-frame, full-coverage GCPs. Then
`gdalwarp -tps` (or `-order 2`), with RMS measured against held-out corners. This is
the `v2_next_step` the v1p5 overlay metadata already records ("Interactive QGIS
Georeferencer with bearcub's 229 control points … TPS warp … RMS < 50 m measured").

Remapping the 228-point atlas-frame `diagnostic_manual` set into per-sheet served
pixels is the alternative, if the atlas-to-sheet tiling can be reconstructed, but
that is error-prone and I did not attempt it autonomously (the project already
treats Tuck digitising as not autonomous-precise).

## What this delivers

- `tuck_georef_v2_quantify.json`: the GCP-set audit + the per-sheet served-frame
  reconciliation check (with the frame-mismatch flags).
- `scripts/nome_placer/tuck_georef_quantify.py`: reproducible.

No warp was emitted (blocked). No prod promotion.
