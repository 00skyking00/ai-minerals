# SOURCE: Nome bedrock / Quaternary-cover contact

## Primary precise source (georeferenced, staged for manual digitizing)

USGS Mineral Investigations Field Studies Map MF-247. Hummel, C.L., 1962,
*Preliminary geologic map of the Nome C-1 quadrangle, Seward Peninsula, Alaska*,
scale 1:63,360. Scanned sheet (raster, no vector or text layer), datum NAD27.

- PDF: `research/nome_debate_library/Hummel_1962_MF247_NomeC1_geologic.pdf`
- Georeferenced by `scripts/nome_placer/mf247_georeference.py` to EPSG:3338.
  Registered on the four graticule neatline corners of the Nome C-1 quad
  (64°30' to 64°45'N, 165°00' to 165°30'W, NAD27). The corner pixels were found by
  fitting the four neatline edges and confirmed against the printed graticule
  labels at each corner. Four-corner affine residual is about 20 m, below the
  line width at 1:63,360; the delivered raster uses a thin-plate warp that fits
  the corners exactly. The 1962 sheet is NAD27, so the warp to EPSG:3338 (NAD83)
  carries the roughly 150 m datum shift.
- Raster (regenerable, gitignored):
  `data/derived/nome_placer/mf247/mf247_nomeC1_3338.tif`

### The lump-unit limitation

MF-247 maps the whole coastal plain as a single undifferentiated "Unconsolidated
deposits" unit (glacial, glaciofluvial, alluvial and beach lumped; Pleistocene
to Recent). It does not separate the Nome River drift from the beach deposits.
At the Nome type area the drift is pervasive under the coastal plain and the
beaches are reworked drift, so there is no discrete drift-versus-beach edge to
trace. The one digitizable boundary is the bedrock (Nome Group) / Quaternary-cover
contact, the inland edge of the drift-and-beach plain. The drift-on-beach feature
is built on that contact as a drift-source-proximity proxy.

## Contact vector source (used to build the feature)

`nome_bedrock_quaternary_contact_3338.geojson` is the inland (bedrock-facing)
boundary of the coastal-plain Qs polygon from the digital GeMS SIM 3131 Seward
Peninsula geologic map (1:500,000; Till, Dumoulin et al.), clipped to the Nome
C-1 footprint plus 1.5 km. The seaward (Norton Sound) edge is excluded.

Why the GeMS line rather than a hand-traced MF-247 line: the MF-247 sheet is a
monochrome, contour-dense scan with no clean separation between the patterned
bedrock and the coastal plain. Tracing the contact from it is manual cartographic
work that this unattended run cannot do reliably or verify. SIM 3131 supplies the
same contact as a clean digital line at coarser scale. Near the placer ground the
GeMS line and the MF-247-mapped boundary occupy the same zone; the 1:500,000
generalization differs from the 1:63,360 line by up to a few hundred meters. That
gap is the precision a manual digitize of the georeferenced MF-247 would add. It
does not change the result here: the placer positives sit a median 1.6 km seaward
of the contact, and the strandline-crossing feature self-restricts to the coast.

## Files in this directory

- `nome_bedrock_quaternary_contact_3338.geojson`: the contact vector (188 km within C-1)
- `dist_strandline_x_contact.tif`, `dist_to_contact.tif`: feature rasters (regenerable)
- `placer_bedrock_contact.json`, `.csv`: cross-validation numbers
- `fig_bedrock_contact_drift_on_beach.png`: contact + feature + positives on the georeferenced map
- `REPORT.md`: the short report

Build: `PYTHONPATH=.:src .venv/bin/python -m scripts.nome_placer.placer_bedrock_contact_cv`
(georeferencing: `scripts.nome_placer.mf247_georeference`).
