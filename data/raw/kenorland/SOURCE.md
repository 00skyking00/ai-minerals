# Kenorland Minerals Tanacross drill collars + project centroids

- **Dataset key:** `kenorland`
- **Source URL:** <https://www.kenorlandminerals.com/projects/project-overview/?turl=tanacross>
  and the DGGS NE-Tanacross mineral-resources page
  <https://dggs.alaska.gov/minerals/northeast-tanacross.html>
- **Retrieved / extended:** 2026-04-22 (v1), 2026-04-23 (v1.1)
- **License:** Press-release disclosures by a TSXV-listed issuer, public-domain
  summaries of regulatory filings. Use is limited to research / validation
  purposes; attribute Kenorland Minerals Ltd. and the Alaska DGGS.

## Records

| Row | Type | Source | Precision |
|---|---|---|---|
| `23ETD062` | Named drill hole | 2024-03-28 Kenorland termination PR | Visually digitized from a small-scale figure. Approximately ±5–15 km; the stub may actually be a project-area centroid rather than a collar point. |
| `East_Taurus_centroid` | Property polygon centroid | DGGS NE-Tanacross mineral-resources page (linked above) | Approximate; the East Taurus target is ~1–3 km across but centroid is digitized by hand. |
| `West_Taurus_centroid` | Property polygon centroid | DGGS NE-Tanacross | Approximate. |
| `South_Taurus_centroid` | Property polygon centroid | 2025-10 Kenorland exploration update | Approximate. **Drilled 2025, no economic mineralisation intersected** — treated as an *external negative* for blind-test purposes. |
| `Bluff_centroid` | Property polygon centroid | DGGS NE-Tanacross | Approximate; limited public drilling data. |

## Notes

1. **All coordinates are approximate.** None come from NI 43-101 survey
   tables; all are visually digitized from press-release or DGGS figures
   that may predate the final drill collars. Assume ±5–15 km uncertainty
   for the centroids and ±1–5 km for the named hole. The Day-5 notebook
   contains a 15-km-disk sensitivity sweep that demonstrates the blind-
   test conclusion is robust across this uncertainty range.

2. **Deposit-model class.** Kenorland's 2023 drilling at Taurus targets
   the "porphyry Cu-Mo-Au-Ag" class, consistent with Cox & Singer model
   codes 17, 20c, 21a, 21b (family) that our model treats as positives.

3. **What's useful for validation.**
   - `23ETD062` is the only named collar in any public Kenorland
     disclosure. Post-training-cutoff (2023), making it a genuine blind.
   - `East_Taurus_centroid` and `West_Taurus_centroid` are drill-tested
     post-training positives but without per-hole precision — score the
     model at centroid and interpret as a property-scale check.
   - `South_Taurus_centroid` is the rare true *negative* blind: drilled
     2025, no economic mineralisation intersected. If our model scores
     it high, that would be a false-positive confirmed by drilling.
   - `Bluff_centroid` had 2019 Antofagasta drilling; outcomes reported
     publicly are ambiguous. Listed for completeness.

4. **What's missing and why.** Kenorland has drilled ~13,000 m across
   ~15+ holes since 2019 but publishes only the single 23ETD062 collar.
   Antofagasta drilled the earlier programs and published nothing per-
   hole. Manh Choh (Tetlin) and Pogo Mine have substantial public drill
   data but are *Au* deposits (epithermal and orogenic respectively),
   not porphyry — mis-category for this model. Orange Hill (Nabesna,
   NB015 in ARDF) has rich historical drill data in six DGGS GMC reports
   (372–377), but it was already an ARDF positive at training time and
   is therefore *not* blind. The realistic post-training blind-test set
   for porphyry in this AOI is fundamentally 1 named hole + ~4 project
   polygons.
