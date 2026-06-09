"""Two-stage lithology normalization for the BCGS dh2loop export.

Reads ``data/derived/bcgs_dh2loop/Lithology.csv`` (raw operator-format
text) and writes ``data/derived/bcgs_dh2loop/Lithology_normalized.csv``
with three additional columns:

- ``Detailed_Lithology_norm`` — operator term after stage-1 decoding
  (plain-English geological text)
- ``Lithology_Subgroup_norm`` / ``Lithology_Group_norm`` — dh2loop's
  standardized 3-level hierarchy from the 757-term thesaurus

Plus a per-interval ``norm_provenance`` column tagging where the
normalization came from: ``bcgs_decoder_high``, ``bcgs_decoder_guess``,
``dh2loop_fuzzy`` (with the rapidfuzz score), or ``unmapped``.

Pipeline:

  raw operator text
    -> (stage 1a) BCGS-specific decoder dict (hand-built; ~30 codes)
       -> plain-English geological term
       -> (stage 1b) dh2loop hierarchical thesaurus fuzzy match (rapidfuzz)
          -> (Detailed_Lithology, Lithology_Subgroup, Lithology_Group)

Discovery from this iteration: dh2loop's
``thesaurus_geology_lithology_code.csv`` is WAMEX-specific (per-company
operator-code lookup built from Western Australian exploration records).
Only 1 of the 27 BCGS-AOI top-25 codes (DI -> Diorite) matches it
directly. The hierarchical thesaurus IS portable because it fuzzy-matches
on plain-English geological terms regardless of geographic origin. So
the realistic pipeline is: hand-built BCGS decoder for the operator
shorthand we can decode, then the portable hierarchical thesaurus for
the result + the plain-English BCGS terms that need no decoding.

The hand-built BCGS decoder has 3 confidence tiers:
- ``high``: codes whose decoding is industry-standard or self-evident
  (e.g. OVB -> overburden, DI -> diorite, Sst -> sandstone)
- ``guess``: codes that probably decode a certain way but I'm
  inferring from context (e.g. DM -> mafic dike, MMD -> medium-grained
  mafic dike)
- ``unknown``: codes left as-is because I can't decode them without
  an operator-specific key (TFML, AXXX, AXXZ, AnLAT, VCCL, PPHM, ...)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

LITHO_IN = Path("data/derived/bcgs_dh2loop/Lithology.csv")
LITHO_OUT = Path("data/derived/bcgs_dh2loop/Lithology_normalized.csv")
THESAURUS = Path("data/raw/dh2loop_thesauri/thesaurus_geology_hierarchical.csv")

# Stage 1a: hand-built BCGS-shorthand decoder. Keys are lowercased
# operator codes; values are (plain_english_decoded, confidence_tag).
#
# Confidence tag controls how the result is reported in
# ``norm_provenance``. "high" = industry-standard geological abbreviation
# or self-evident expansion. "guess" = my best inference, NOT verified
# against a BCGS data dictionary; flagged in the output so a reviewer
# can audit.
BCGS_DECODER: dict[str, tuple[str, str]] = {
    # Surficial / drill artifacts (high confidence)
    "ovb":      ("overburden", "high"),
    "ovbd":     ("overburden", "high"),
    "ob":       ("overburden", "high"),
    "over":     ("overburden", "high"),
    "overburden": ("overburden", "high"),
    "wcas":     ("casing", "high"),
    "cas":      ("casing", "high"),
    "casn":     ("casing", "high"),
    "case":     ("casing", "high"),
    "casing":   ("casing", "high"),
    "casing overburden": ("overburden", "high"),
    "dhcs":     ("casing", "high"),
    "drillhole casing dhcs": ("casing", "high"),
    "till":     ("till", "high"),
    "soil":     ("soil", "high"),
    "alluvium": ("alluvium", "high"),
    # Intrusive rocks (high confidence — industry-standard abbrevs)
    "di":       ("diorite", "high"),
    "dio":      ("diorite", "high"),
    "ddrt":     ("diorite", "high"),
    "monz":     ("monzonite", "high"),
    "mnz":      ("monzonite", "high"),
    "mnzd":     ("monzodiorite", "high"),
    "mnzt":     ("monzonite", "high"),
    "qmd":      ("quartz monzodiorite", "high"),
    "qm":       ("quartz monzonite", "high"),
    "gr":       ("granite", "high"),
    "gd":       ("granodiorite", "high"),
    "pphm":     ("porphyry", "high"),
    "porphyry": ("porphyry", "high"),
    "pegmatite": ("pegmatite", "high"),
    # Volcanic (high confidence)
    "volc":     ("volcanic rock", "high"),
    "vu":       ("volcanic rock", "high"),
    "ivol":     ("intermediate volcanic", "high"),
    "iv":       ("intermediate volcanic", "high"),
    "tuff":     ("tuff", "high"),
    "andesite": ("andesite", "high"),
    "ands":     ("andesite", "high"),
    "anlt":     ("andesite lapilli tuff", "high"),
    "marble":   ("marble", "high"),
    # Sedimentary (high confidence)
    "sst":      ("sandstone", "high"),
    "slts":     ("siltstone", "high"),
    "siltstone": ("siltstone", "high"),
    "arg":      ("argillite", "high"),
    "ls":       ("limestone", "high"),
    # Skarn / mineralization (high confidence in BC porphyry context)
    "sk":       ("skarn", "high"),
    # Structural (high confidence)
    "flt":      ("fault", "high"),
    "fault":    ("fault", "high"),
    "fault ed": ("fault", "high"),  # the de-parenned "Fault(ed)"
    # Best-guess (medium confidence; flagged for review)
    "dm":       ("mafic dike", "guess"),
    "md":       ("mafic dike", "guess"),
    "df":       ("felsic dike", "guess"),
    "fp":       ("felsic porphyry", "guess"),
    "vm":       ("volcanic mafic", "guess"),
    "at":       ("andesite tuff", "guess"),
    "ivf":      ("intermediate volcanic, felsic", "guess"),
    "mmd":      ("medium-grained mafic dike", "guess"),
    "anlat":    ("andesite lapilli tuff", "guess"),
    "vccl":     ("volcanic conglomerate", "guess"),
    "vcgl":     ("volcanic conglomerate", "guess"),
    "vcl":      ("volcanic clastic", "guess"),
    "du":       ("diorite undifferentiated", "guess"),
    "spdu":     ("diorite undifferentiated", "guess"),
    "altint":   ("alteration interval", "guess"),
    "mst":      ("mudstone", "guess"),
    "mdst":     ("mudstone", "guess"),
    "mv":       ("mafic volcanic", "guess"),
    "xt":       ("exotic", "guess"),
}

# Codes I can NOT decode without an operator-specific key — these are
# tracked here for the chapter prose's "opaque vocabulary" count but
# do not get a hand decoding. They pass through to stage 1b unchanged
# and almost certainly land as ``unmapped``.
KNOWN_OPAQUE: set[str] = {
    "tfml", "axxx", "axxz",
}

# Stage 1b: dh2loop hierarchical thesaurus. The ``fuzzywuzzy_terms``
# column is a space-separated list of synonyms; we explode it into
# one synonym per row, then rapidfuzz-match each input lithology
# against the flattened synonym list.
MIN_FUZZY_SCORE = 80  # cuts off junk matches; tuned by inspecting outputs.
# Initial cutoff was 85; lowering to 80 captured "massive magnetite"
# (score 81.8 vs "magnetite ore") and "beforsite" (score 82.4 vs
# "beresite", the modern spelling of the same hydrothermal-altered
# rock). Lower values risk false matches on the long tail of
# 2-3-character operator codes, where token_set_ratio over-rewards
# letter overlap.


def load_thesaurus() -> tuple[list[str], dict[str, tuple[str, str, str]]]:
    """Returns (flat synonym list, synonym -> hierarchy triple dict)."""
    df = pd.read_csv(THESAURUS)
    synonym_to_hierarchy: dict[str, tuple[str, str, str]] = {}
    for _, row in df.iterrows():
        if pd.isna(row["fuzzywuzzy_terms"]):
            continue
        triple = (row["detailed_lithology"], row["lithology_subgroup"],
                  row["lithology_group"])
        for syn in str(row["fuzzywuzzy_terms"]).split():
            # Thesaurus synonyms are joined with underscores ("Mafic_Dike");
            # normalize to spaces for matching against free text.
            key = syn.replace("_", " ").strip().lower()
            if key and key not in synonym_to_hierarchy:
                synonym_to_hierarchy[key] = triple
    print(f"[thesaurus] loaded {len(df):,} hierarchy rows; "
          f"{len(synonym_to_hierarchy):,} unique synonym keys")
    return list(synonym_to_hierarchy.keys()), synonym_to_hierarchy


def _pre_clean(raw: str) -> str:
    """Strip embedded newlines/carriage returns/parentheses, convert
    hyphens and slashes to spaces, collapse runs of whitespace, lowercase.

    Many BCGS lithology entries embed structural punctuation that breaks
    the downstream fuzzy match: ``"qtz-biotite-diorite"`` (hyphens),
    ``"Massive\\nMagnetite"`` (literal newline from OCR or paste),
    ``"Fault(ed)"`` (parens), ``"syenodiorite/monzonite"`` (slash). All
    of these are plain geological text once the punctuation is normalized
    to spaces.
    """
    if not isinstance(raw, str):
        return ""
    s = re.sub(r"[\n\r\-/()]+", " ", raw)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def stage_1a_decode(raw: str) -> tuple[str, str]:
    """BCGS-specific decoder. Returns (decoded_text, provenance_tag)."""
    clean = _pre_clean(raw)
    if not clean:
        return ("", "unmapped")
    # Try whole-string match against the decoder.
    if clean in BCGS_DECODER:
        text, conf = BCGS_DECODER[clean]
        return (text, f"bcgs_decoder_{conf}")
    if clean in KNOWN_OPAQUE:
        return (clean, "bcgs_opaque")
    # Pass-through: the cleaned text is either plain-English geological
    # vocabulary (e.g. "quartz biotite gneiss", "andesite") or an
    # unknown operator code we haven't catalogued yet.
    return (clean, "passthrough")


def stage_1b_thesaurus_match(text: str, synonyms: list[str],
                             lookup: dict[str, tuple[str, str, str]]
                             ) -> tuple[str | None, str | None, str | None, int]:
    """rapidfuzz match against the dh2loop hierarchical thesaurus."""
    if not text:
        return (None, None, None, 0)
    result = process.extractOne(text, synonyms, scorer=fuzz.token_set_ratio,
                                score_cutoff=MIN_FUZZY_SCORE)
    if result is None:
        return (None, None, None, 0)
    matched_synonym, score, _ = result
    triple = lookup[matched_synonym]
    return (triple[0], triple[1], triple[2], int(score))


def main() -> None:
    litho = pd.read_csv(LITHO_IN)
    print(f"[litho] loaded {len(litho):,} intervals")

    synonyms, lookup = load_thesaurus()

    # Stage 1a on every row.
    decoded = litho["Detailed_Lithology"].apply(stage_1a_decode)
    litho["Detailed_Lithology_norm"] = [d[0] for d in decoded]
    litho["stage1a_provenance"] = [d[1] for d in decoded]

    # Stage 1b: thesaurus match. Apply to every UNIQUE decoded string
    # (much faster than per-row apply since there are ~2k distinct
    # strings vs 100k+ rows).
    unique_decoded = litho["Detailed_Lithology_norm"].unique()
    print(f"[match] running fuzzy match on {len(unique_decoded):,} distinct strings...")
    cache = {}
    for s in unique_decoded:
        cache[s] = stage_1b_thesaurus_match(s, synonyms, lookup)
    litho["thes_detailed"] = litho["Detailed_Lithology_norm"].map(lambda s: cache[s][0])
    litho["thes_subgroup"] = litho["Detailed_Lithology_norm"].map(lambda s: cache[s][1])
    litho["thes_group"]    = litho["Detailed_Lithology_norm"].map(lambda s: cache[s][2])
    litho["thes_score"]    = litho["Detailed_Lithology_norm"].map(lambda s: cache[s][3])

    # Final norm_provenance combines stage 1a + 1b outcomes.
    def _provenance(row: pd.Series) -> str:
        if pd.notna(row["thes_detailed"]):
            if row["stage1a_provenance"].startswith("bcgs_decoder_"):
                return f"{row['stage1a_provenance']}+dh2loop_fuzzy"
            return "dh2loop_fuzzy"
        return row["stage1a_provenance"] if row["stage1a_provenance"] != "passthrough" else "unmapped"
    litho["norm_provenance"] = litho.apply(_provenance, axis=1)

    # Write.
    LITHO_OUT.parent.mkdir(parents=True, exist_ok=True)
    out = litho[["CollarID", "FromDepth", "ToDepth", "Detailed_Lithology",
                 "Detailed_Lithology_norm", "thes_detailed",
                 "thes_subgroup", "thes_group", "thes_score",
                 "norm_provenance", "Comments"]]
    out.to_csv(LITHO_OUT, index=False)
    print(f"[litho] wrote {LITHO_OUT}  ({len(out):,} rows)")

    # Report.
    print()
    print("=" * 60)
    print("NORMALIZATION COVERAGE")
    print("=" * 60)
    counts = litho["norm_provenance"].value_counts()
    total = len(litho)
    for prov, n in counts.items():
        print(f"  {prov:<48s}  {n:>7,}  ({100*n/total:5.1f}%)")
    print()
    decoded_total = (~litho["thes_detailed"].isna()).sum()
    print(f"Total with standardized hierarchy: {decoded_total:,} of {total:,}  "
          f"({100*decoded_total/total:.1f}%)")
    print()
    print("Top 15 unmapped operator strings (by interval count):")
    unmapped = litho[litho["norm_provenance"].isin(["unmapped", "bcgs_opaque"])]
    for term, n in unmapped["Detailed_Lithology"].value_counts().head(15).items():
        print(f"  {n:>6,}  {term!r}")


if __name__ == "__main__":
    main()
