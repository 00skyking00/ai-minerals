# MVP proposal — ai-minerals v1 (Tanacross)

## Goal

Produce a public, runnable demonstration — a single Quarto notebook plus supporting Python module — that takes open geoscientific data for the Tanacross quadrangle of eastern Alaska, trains a simple supervised classifier, and outputs a porphyry-copper prospectivity map with honest validation and interpretation. The writeup is the primary artifact.

Audience: senior ML + geoscience hiring managers at KoBold Metals, ExploreTech, Earth AI, and similar.

Success criterion: a reader can clone the repo, execute the notebook in a fresh `uv` environment, produce the published maps, and read the writeup and come away thinking "this person could contribute credibly on our team." Nothing more ambitious than that.

## Region and commodity

**Tanacross quadrangle, eastern Alaska. Porphyry Cu-Mo-Au-Ag system** (Taurus + related prospects).

Rationale (summarized from the approved plan):
- Personal familiarity advantage (narrative authenticity).
- KoBold is explicitly active in adjacent Alaska geology (Skolai, Wrangellia Ni-Cu-Co-PGM).
- Porphyry Cu-Mo is a textbook MPM target with abundant literature.
- Clean, curated public data (AGDB4, ARDF, USGS OFR 2022-1046 Taurus SEM analyses).
- A natural external validation set exists: Kenorland Minerals' 2023–2024 Tanacross drill-hole assays postdate MRDS label cutoffs, creating a genuine temporal holdout.

**Deliberately deferred:** a Mother Lode, California orogenic-gold sequel (v2) reusing this scaffolding. Explicitly out of scope for v1.

## Deposit-model target

Porphyry Cu-Mo-Au-Ag systems, late-Cretaceous age (68–73 Ma) per the Taurus region geology. Expected geological/geochemical signatures:
- Intrusive-hosted (granodiorite, quartz monzonite).
- Hydrothermal alteration halos (potassic, phyllic, argillic, propylitic) — detectable via remote-sensing spectral indices.
- Proximal geochemistry: Cu-Mo-Au; distal halo: As, Sb, Pb, Zn, Bi, Te.
- Structural controls: fault intersections, proximity to contacts with late-stage intrusives.
- Geophysical signature: magnetic highs from magnetite-series intrusives; K/Th radiometric anomalies from K-silicate alteration; gravity lows from felsic host rocks offset by denser sulfide mineralization at depth.

The feature set below is designed to proxy these signatures from open data.

## Pipeline

### Data acquisition

All fetches are scripted and reproducible. Each dataset lives in `data/raw/<dataset>/` with a `SOURCE.md` documenting URL, retrieval date, license.

1. **USGS MRDS + ARDF** — filtered to Tanacross quadrangle and commodity = Cu, Mo, Au, Ag with deposit-type containing "porphyry" (MRDS) or equivalent ARDF deposit-model IDs.
2. **USGS AGDB4** — best-value geochemistry, rock + stream-sediment + soil, clipped to Tanacross.
3. **USGS geophysical grids** — residual aeromagnetic + K/U/Th radiometric + Bouguer gravity rasters over Tanacross. CMMI tri-national grids as a fallback if coverage is cleaner there.
4. **Sentinel-2 L2A** — summer (snow-free, low-cloud) median composite from Microsoft Planetary Computer via `pystac-client` + `stackstac`.
5. **USGS SGMC** — state geologic map compilation for the AK slice; polygons + contact linework for lithology-class and fault-proximity features.
6. **Copernicus GLO-30** DEM — from Planetary Computer.
7. **Kenorland drill collars 2023–24** — digitized from Kenorland press releases; held out from training, used as external validation only.

### Feature engineering

Define a regular analysis grid (500m or 1km pixels, CRS chosen based on Tanacross extent — probably EPSG:3338 Alaska Albers). Sample per pixel:

- **Geochemistry features.** For each pathfinder element {Cu, Mo, Au, Ag, As, Sb, Pb, Zn, Bi, Te}, aggregate (mean, max, sample count) over AGDB4 samples within a 5 km buffer. Where relevant, apply CLR (centered log-ratio) transform using `pyrolite` to elements treated as a closed system.
- **Remote-sensing features.** Sentinel-2 median-composite band ratios:
  - Iron-oxide index: B4/B2 (red / blue)
  - Ferrous-iron index: B11/B8 (SWIR1 / NIR)
  - Clay / hydroxyl index: B11/B12 (SWIR1 / SWIR2)
  - NDVI (B8-B4)/(B8+B4) — for masking heavily vegetated pixels
- **Geophysics features.** Residual aeromagnetic intensity, K, Th, U, K/Th ratio, Bouguer gravity — sampled at pixel center.
- **Geology features.** Lithology class (one-hot of top-N most-common units in the Tanacross extent), distance-to-nearest-fault derived from SGMC contact lines.
- **Topographic features.** Elevation, slope, TRI from Copernicus GLO-30.

Typical feature dimensionality: ~30–45 columns per pixel.

### Labeling

Two parallel labeling schemes, compared side by side:

**Scheme A: supervised with pseudo-negatives.**
- Positives: MRDS + ARDF porphyry-Cu occurrences.
- Negatives: random sample of pixels ≥5 km from any MRDS/ARDF Cu occurrence, stratified by lithology class to avoid trivial separation.
- Train:test ratio of positives:pseudo-negatives balanced via `imbalanced-learn` (SMOTE or random undersampling, both reported).

**Scheme B: PU (positive-unlabeled) learning.**
- Positives: same.
- No explicit negatives; all non-positive pixels treated as unlabeled.
- Simple bagging-SVM (Mordelet & Vert style) fit as a baseline. No extensive hyperparameter search — goal is to demonstrate methodological awareness, not win the benchmark.

Scheme A is the primary model; Scheme B is a side-by-side comparison in the writeup.

### Models

- **Baseline:** standardized-feature logistic regression (scikit-learn).
- **Main:** Random Forest (scikit-learn).
- **Optional, if time permits:** gradient-boosted trees (XGBoost or LightGBM).

No neural network. The feature space is tabular at ~10³–10⁴ pixels of training data; tree ensembles are the appropriate tool. A deep model here would be methodologically indefensible.

### Validation

1. **Spatial (block) cross-validation.** Partition the Tanacross extent into coarse geographic blocks (probably ~10–20 km grid cells) and perform k-fold CV where each fold holds out a whole block rather than random points. Use `spacv` or roll a simple block-CV splitter. Report **AUC-ROC + PR-AUC per fold** and the fold-averaged success-rate curve.

2. **Success-rate curve.** Plot cumulative fraction of known deposits captured vs cumulative fraction of pixels flagged prospective at decreasing probability thresholds. The standard honest metric in MPM literature under heavy class imbalance.

3. **External blind test via Kenorland drill collars.** Kenorland's 2023–24 Tanacross drill holes (including 23ETD062 returning 174.22m at 0.14% Cu, 0.02% Mo, 0.05 g/t Au) are not in MRDS and were not used in training. Compute the model's predicted probability at each hole collar; report the distribution vs. a null distribution from random non-MRDS pixels. A well-behaved model should rank the Kenorland holes visibly higher than random.

4. **Honest reporting of failure modes.** If the model performs poorly under spatial CV (a real risk with small-N porphyry occurrences), report that without editorializing. A failed-but-well-validated demo is more credible than a gamed-but-apparently-successful one.

### Interpretation

- **SHAP summary plot** on the Random Forest showing top-15 feature importances.
- **SHAP dependence plots** for the top 3–4 features, with an interpretive paragraph connecting each to known porphyry-Cu mineral-systems theory.
- **Explicit check:** if the top features don't map onto established geological theory, that's an important signal — the model may be learning confounders (e.g., field-crew sampling bias, geological-map artifacts). Call this out if it appears.

### Outputs

- Tanacross-wide prospectivity GeoTIFF at the analysis-grid resolution.
- Top-N polygons (probably N=10) of highest-probability regions, with probability bounds.
- Interactive folium map embedded in the Quarto-rendered HTML.
- Static PNG hero images for the README and writeup.

## Stack

Per [CLAUDE.md](../CLAUDE.md):

- Python 3.11+ via **uv** environment management.
- Source authored in Quarto `.qmd` (under `notebooks/`), renders to `.ipynb` (executable) and `.html` (GitHub Pages).
- Geospatial: GeoPandas, Shapely, Fiona, Rasterio, rioxarray, xarray.
- STAC: pystac-client + stackstac + planetary-computer against MS Planetary Computer.
- Geochemistry: pyrolite.
- ML: scikit-learn, imbalanced-learn, shap, optional xgboost/lightgbm.
- Viz: matplotlib, folium, contextily.

No Prefect/Dagster/MLflow/DVC. Single repo, thin `src/ai_minerals` module, one notebook per region.

## Narrative structure

The writeup drives the artifact. Sections in the Tanacross notebook:

1. **Why critical minerals, why porphyry Cu, why Alaska** — one-page framing, connects to the target companies' theses.
2. **Tanacross in context** — the Taurus Cu-Mo-Au-Ag porphyry system, published work by USGS and Kenorland, why it's a well-constrained place to demo MPM.
3. **Data inventory** — what's being pulled, where, why, limits and noise of each source.
4. **The label problem** — known deposits ≠ labeled non-deposits. Pseudo-negatives vs. PU framing.
5. **Features + geological justification** — each feature gets a one-sentence "why this matters" tied to porphyry-Cu mineral-systems theory.
6. **Model + spatial CV** — baseline + main model, why random CV is unsafe.
7. **SHAP interpretation** — does the model learn geology or noise?
8. **External validation via Kenorland** — temporal holdout, result, discussion.
9. **Outputs** — maps + top-N targets with uncertainty.
10. **Limits** — explicit boxes: "how this differs from KoBold's actual system," "how this differs from ExploreTech's decision-theoretic stack."
11. **Next steps** — SimPEG 2.5D magnetic inversion; ASTER SWIR alteration; PU-learning extension; foundation-model fine-tune (Clay/Prithvi/GFM4MPM); POMDP drill-planner sketch.

## Out of scope (don't drift)

- 3D physics-consistent geophysical inversions (SimPEG-based). Acknowledged as the natural next step.
- POMDP / decision-theoretic drill planning. Sketched in prose; not implemented.
- OCR of legacy drill logs. Referred out to the separate Bear Cub subproject.
- Production deployment, hosting, Docker, CI.
- Multi-region ensemble in v1.
- Commercial data.
- Uncertainty quantification beyond per-pixel classification probability (no Monte Carlo, no Bayesian NN).

## Timeline

Per approved plan: 3–4 h/day, 7 days, ≈ 21–28 h total.

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Research + scaffold | Research briefs ×5, design doc, git repo, Quarto stub, CLAUDE.md, pyproject.toml |
| 2 | Data acquisition | Scripts pulling MRDS/ARDF/AGDB4/SGMC/geophysics/S2 for Tanacross, saved to `data/raw/` |
| 3 | Feature engineering + baseline | Logistic-regression notebook running end-to-end with spatial CV |
| 4 | Main model + SHAP | Random Forest, SHAP summary + dependence plots, top-features prose |
| 5 | External validation | Kenorland-hole blind test + success-rate curve; map polish |
| 6 | Writeup + render | Finalize `.qmd` prose, README, figures, Quarto HTML |
| 7 | Review + publish | Public GitHub repo, Pages site live |

Mother Lode v2 sequel and Bear Cub subproject are separate engagements starting after v1 ships.

## What this doesn't prove

- That the author can build a production exploration system.
- That the author can beat a real exploration company's internal ML.
- That Tanacross has an economically viable Cu deposit where the model flags high probability.
- That the modeling choices here are optimal, rather than deliberately simple.

These limits are the point. The demo proves the author understands the problem space, respects its methodological pitfalls, can execute a data-to-map pipeline using open tools on open data, and can write about it honestly. Nothing more — but also nothing less.
