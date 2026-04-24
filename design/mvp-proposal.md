# MVP proposal — ai-minerals v1 (Eastern Alaska porphyry belt)

## Goal

Produce a public, runnable demonstration — a single Quarto notebook plus supporting Python module — that takes open geoscientific data for a three-quadrangle slice of eastern Alaska, trains a simple supervised classifier, and outputs a porphyry-copper prospectivity map with honest validation and interpretation. The writeup is the primary artifact.

Audience: senior ML + geoscience hiring managers at KoBold Metals, ExploreTech, Earth AI, and similar.

Success criterion: a reader can clone the repo, execute the notebook in a fresh `uv` environment, produce the published maps, and read the writeup and come away thinking "this person could contribute credibly on our team." Nothing more ambitious than that.

## Region and commodity

**Eastern Alaska porphyry belt — Tanacross (TC) + Mt Hayes (MH) + Nabesna (NB) 1:250,000 quadrangles. Porphyry Cu-Mo-Au-Ag systems** (Taurus + related prospects in TC, and ~40 additional occurrences across MH + NB).

The AOI spans the Wrangellia–Yukon-Tanana tectonic boundary: Mt Hayes and Nabesna sit on the Wrangellia accreted-terrane block (where KoBold's Skolai project also sits), while Tanacross sits on the Yukon-Tanana continental-arc upland. Both host Cretaceous porphyry Cu-Mo, but the structural controls differ — an ML signal we can probe via SHAP.

Rationale:
- 62 ARDF porphyry positives vs. 15 in Tanacross alone — enough for stable spatial-CV variance and defensible error bars.
- Narrative breadth: "eastern Alaska porphyry belt" reads as a real geological region, not a single map sheet.
- KoBold is active in the Wrangellia portion of this exact AOI (Skolai).
- Clean, curated public data (AGDB4, ARDF, USGS OFR 2022-1046 Taurus SEM analyses).
- A natural external validation set: Kenorland Minerals' 2023–2024 Tanacross drill-hole assays postdate MRDS label cutoffs, creating a genuine temporal holdout within TC.

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

1. **USGS MRDS + ARDF** — filtered to the AOI polygon + commodity = Cu, Mo, Au, Ag with deposit-type containing "porphyry" (MRDS) or equivalent ARDF deposit-model IDs. ARDF records from all three quadrangles (TC, MH, NB) combined then geometrically clipped.
2. **USGS AGDB4** — best-value geochemistry, rock + stream-sediment + soil, clipped to the AOI.
3. **USGS geophysical grids** — residual aeromagnetic + K/U/Th radiometric + Bouguer gravity rasters over the AOI. CMMI tri-national grids as a fallback if coverage is cleaner there.
4. **Sentinel-2 L2A** — summer (snow-free, low-cloud) composite from Microsoft Planetary Computer via `pystac-client` + `stackstac`. Mean of the 20 lowest-cloud scenes at 60 m.
5. **USGS SIM 3340 (Alaska geology)** — polygon lithology + fault/contact linework, clipped to AOI.
6. **Copernicus GLO-30** DEM — from Planetary Computer.
7. **Kenorland drill collars 2023–24** — digitized from Kenorland press releases (TC only); held out from training, used as external validation.

### Feature engineering

Define a regular analysis grid (500m or 1km pixels, CRS chosen based on AOI extent — probably EPSG:3338 Alaska Albers). Sample per pixel:

- **Geochemistry features.** For each pathfinder element {Cu, Mo, Au, Ag, As, Sb, Pb, Zn, Bi, Te}, aggregate (mean, max, sample count) over AGDB4 samples within a 5 km buffer. Where relevant, apply CLR (centered log-ratio) transform using `pyrolite` to elements treated as a closed system.
- **Remote-sensing features.** Sentinel-2 median-composite band ratios:
  - Iron-oxide index: B4/B2 (red / blue)
  - Ferrous-iron index: B11/B8 (SWIR1 / NIR)
  - Clay / hydroxyl index: B11/B12 (SWIR1 / SWIR2)
  - NDVI (B8-B4)/(B8+B4) — for masking heavily vegetated pixels
- **Geophysics features.** Residual aeromagnetic intensity, K, Th, U, K/Th ratio, Bouguer gravity — sampled at pixel center.
- **Geology features.** Lithology class (one-hot of top-N most-common units in the AOI extent), distance-to-nearest-fault derived from SGMC contact lines.
- **Topographic features.** Elevation, slope, TRI from Copernicus GLO-30.

Typical feature dimensionality: ~30–45 columns per pixel.

### Labeling

**Positive class — porphyry family (primary).** Filter ARDF by Cox & Singer deposit-model codes, not the free-text `dep_model` field. The structured code field excludes the ~5 speculative "porphyry?" records with no code and the pure Cu skarn (model 18a) records.

- **Family set** (~56 records): codes `17`, `20c`, `21a`, `21b` — covering canonical porphyry Cu-Mo (21a), plutonic porphyry Cu (17), skarn-related porphyry Cu (20c), and porphyry Mo low-F (21b). All intrusion-hosted Cu/Mo variants of the same genetic system.
- **Strict set** (~32 records, for sensitivity check at the end of Day 5): code `21a` only, the canonical Porphyry Cu-Mo model.

The family set is our primary training positive class. The sensitivity check (below) quantifies whether broadening the definition was worth the heterogeneity cost.

Two parallel labeling schemes, compared side by side:

**Scheme A: supervised with pseudo-negatives.**
- Positives: ARDF porphyry-family records (as above).
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

1. **Spatial (block) cross-validation.** Partition the AOI extent into coarse geographic blocks (probably ~10–20 km grid cells) and perform k-fold CV where each fold holds out a whole block rather than random points. Use `spacv` or roll a simple block-CV splitter. Report **AUC-ROC + PR-AUC per fold** and the fold-averaged success-rate curve.

2. **Success-rate curve.** Plot cumulative fraction of known deposits captured vs cumulative fraction of pixels flagged prospective at decreasing probability thresholds. The standard honest metric in MPM literature under heavy class imbalance.

3. **External blind test via Kenorland drill collars.** Kenorland's 2023–24 Tanacross drill holes (including 23ETD062 returning 174.22m at 0.14% Cu, 0.02% Mo, 0.05 g/t Au) are not in MRDS and were not used in training. Compute the model's predicted probability at each hole collar; report the distribution vs. a null distribution from random non-MRDS pixels. A well-behaved model should rank the Kenorland holes visibly higher than random.

4. **Honest reporting of failure modes.** If the model performs poorly under spatial CV (a real risk with small-N porphyry occurrences), report that without editorializing. A failed-but-well-validated demo is more credible than a gamed-but-apparently-successful one.

5. **Sensitivity check — strict 21a-only positives.** At the end of Day 5, retrain the Random Forest on the strict Porphyry Cu-Mo (model 21a) positive set — ~32 records vs ~56 in the family set. Compare under the same spatial-CV regime and the same feature stack. Report: (a) AUC-ROC and PR-AUC for both, (b) whether the top-5 SHAP features move, (c) whether the top-N target polygons move. If AUC changes by <0.02 and target polygons are substantially stable, the broader family definition was a free lunch — adopt it. If it swings meaningfully, the family set is learning heterogeneous sub-types and we should either stick with 21a strict or split into two per-subtype models. Report the verdict honestly in the writeup.

### Interpretation

- **SHAP summary plot** on the Random Forest showing top-15 feature importances.
- **SHAP dependence plots** for the top 3–4 features, with an interpretive paragraph connecting each to known porphyry-Cu mineral-systems theory.
- **Explicit check:** if the top features don't map onto established geological theory, that's an important signal — the model may be learning confounders (e.g., field-crew sampling bias, geological-map artifacts). Call this out if it appears.

### Outputs

- AOI-wide prospectivity GeoTIFF at the analysis-grid resolution.
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

The writeup drives the artifact. Sections in the v1 notebook:

1. **Why critical minerals, why porphyry Cu, why Alaska** — one-page framing, connects to the target companies' theses.
2. **Eastern Alaska porphyry belt in context** — Wrangellia vs. Yukon-Tanana tectonic framing, the Taurus Cu-Mo-Au-Ag system in TC, published work by USGS and Kenorland, why this belt is a well-constrained place to demo MPM.
3. **Data inventory** — what's being pulled, where, why, limits and noise of each source.
4. **The label problem** — known deposits ≠ labeled non-deposits. Pseudo-negatives vs. PU framing.
5. **Features + geological justification** — each feature gets a one-sentence "why this matters" tied to porphyry-Cu mineral-systems theory.
6. **Model + spatial CV** — baseline + main model, why random CV is unsafe.
7. **SHAP interpretation** — does the model learn geology or noise?
8. **External validation via Kenorland** — temporal holdout, result, discussion.
9. **Outputs** — maps + top-N targets with uncertainty.
10. **Limits** — explicit boxes: "how this differs from KoBold's actual system," "how this differs from ExploreTech's decision-theoretic stack."
11. **Next steps** — SimPEG 2.5D magnetic inversion; ASTER SWIR alteration; PU-learning extension; foundation-model fine-tune (Clay/Prithvi/GFM4MPM); POMDP drill-planner sketch.

## Known improvements to revisit

Items surfaced during implementation that are *not* blocking v1 completion
but are worth queuing for v1.1 or the writeup's "next steps" section.

- **`<el>_has_data` indicator columns for geochemistry.** Rare-pathfinder
  elements have high NaN rates (Te ~56%, Au ~47%, Ag ~31%) because a
  cell's 5 km neighborhood may have no sample tested for that element.
  Day-3's baseline median-imputes these, which loses information: the
  *fact* of missingness likely correlates with exploration history and
  terrain accessibility, so a binary `<el>_has_data` companion column
  would give the model an honest signal. Cheap to add (~20 lines in
  `geochem.aggregate_in_radius`); deferred so Day 4 can first compare
  the imputation-free tree models (Random Forest, HistGradientBoosting)
  that handle NaN natively and may sidestep the issue without the
  extra columns.

## Day-4/5 open questions from Day-3 diagnostics

Concrete things to check once the Random Forest + SHAP + blind test are in
place, from running the Day-3 baseline against the full AOI:

- **Exploration-bias confound in the `*_count_5km` features.**
  Cells near known porphyries have **5.2× the AGDB4 sample density** of
  cells in the pseudo-negative pool (Cu-count mean: 120 vs 23). The LR
  baseline put its top weight on `ag_count_5km` (+4.05) and
  `te_count_5km` (+2.43), followed by `cu_mean_5km` (+1.89). Dropping the
  count features drops top-1% capture from 25% → 11% but leaves top-10%
  capture nearly unchanged (89% → 77%). **Day-4 action:** retrain RF
  both with and without count features, compare spatial-CV AUC and SHAP;
  if the count features dominate RF too, either drop them, switch
  explicitly to a PU-learning framing (which doesn't have an artificial
  negative-sampling pool), or add an explicit "sample density" control
  feature.

- **Counter-intuitive negative coefficients on Zn, As, Sb, Bi, s2_iron_oxide.**
  All of these should be *positively* associated with porphyry-Cu systems
  (alteration halos, gossan). The LR is giving them negative coefs,
  suggesting either (a) they co-vary with the count features that are
  eating the positive signal, (b) the pseudo-negatives happen to have
  higher Zn/As than positives due to geochem sampling geography, or (c)
  linear logistic regression is mis-specified for non-monotonic
  relationships. **Day-4 action:** check SHAP + dependence plots on RF.
  If RF gives these features *positive* importance at low values and
  *negative* at high values (U-shape), LR just can't fit it. If RF also
  says "less Zn → more porphyry," investigate whether the pseudo-neg
  pool is geochemically unusual.

- **S2 alteration indices contribute almost nothing.**
  Novel-prediction cells have *the same* iron-oxide and clay index means
  as a random AOI sample (ratio 1.00 and 0.97). Either the model isn't
  using S2, S2 is noisy, or median imputation for the 10 Rainbow Ridge
  NaN positives flattens the signal at the training-set level. **Day-4
  action:** RF feature importance on S2 indices; if they're near zero,
  reconsider whether S2 is worth the complexity for v1 (vs. dropping and
  keeping only the stronger signals).

- **Capture-rate sensitivity to 10 NaN-S2 positives.**
  The Rainbow Ridge cluster has NaN S2 features. They're being
  median-imputed, which means the model gives them the same S2
  contribution as an "average" pixel — likely biasing their predictions
  toward the population mean probability. **Day-5 action:** retrain on
  46 positives only (drop the NaN-S2 ones), compare AUC, PR-AUC, and
  the shape of the success-rate curve. If the 46-positive model's
  metrics are substantially different, document the NaN-imputation as
  an honest limitation.

- **Positive lithology is diffuse — no single-class confound.**
  The 56 positives span 31 different lithology classes (top class has
  only 5 positives). That means the "yellow blob on one rock type"
  concern from the prospectivity-raster review is *probably* unfounded
  — no single lithology dominates positives enough for the model to
  learn "class X = porphyry." **Day-4 action:** verify via SHAP that
  `litho_*` one-hot columns have small SHAP values relative to the
  geochem and geophysics columns. If confirmed, the model is learning
  feature-space mineralogy, not rock-type shortcuts.

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
| 2 | Data acquisition | Scripts pulling MRDS/ARDF/AGDB4/SIM3340/geophysics/S2 for Eastern Alaska AOI, saved to `data/raw/` |
| 3 | Feature engineering + baseline | Logistic-regression notebook running end-to-end with spatial CV |
| 4 | Main model + SHAP | Random Forest, SHAP summary + dependence plots, top-features prose |
| 5 | External validation | Kenorland-hole blind test + success-rate curve; map polish |
| 6 | Writeup + render | Finalize `.qmd` prose, README, figures, Quarto HTML |
| 7 | Review + publish | Public GitHub repo, Pages site live |

Mother Lode v2 sequel and Bear Cub subproject are separate engagements starting after v1 ships.

## What this doesn't prove

- That the author can build a production exploration system.
- That the author can beat a real exploration company's internal ML.
- That eastern Alaska has an economically viable Cu deposit where the model flags high probability.
- That the modeling choices here are optimal, rather than deliberately simple.

These limits are the point. The demo proves the author understands the problem space, respects its methodological pitfalls, can execute a data-to-map pipeline using open tools on open data, and can write about it honestly. Nothing more — but also nothing less.
