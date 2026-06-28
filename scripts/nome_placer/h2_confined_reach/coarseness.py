"""Gold-bound ordinal coarseness miner for the confined-reach test.

The round-5 miner used bare ``\\bfine\\b`` / ``\\bcoarse\\b`` cues. On the richer
mrdata narratives that over-fires badly: "485 fine ounces" (an assay term),
"fine micaceous sand" / "fine creek gravels" (sediment), and "fine-grained
graphitic schist" (bedrock) all match "fine" without describing fine GOLD. This
miner binds every size word to gold and blacklists the sediment / bedrock /
assay contexts, so a class is assigned only when the narrative describes the
size of the gold itself.

Ordinal scale (coarsest evidence wins -- the coarse fraction is the proximity
diagnostic; fines occur at every distance, nuggets only near the source):
  3 proximal  : nuggets, quartz-attached / rough / angular / wire gold
  2 coarse    : gold described as coarse (and not "coarse gravel/sand")
  1 distal    : only fine / flaky / flour / flat gold described
  None        : no gold-size descriptor in the narrative

`max(found)` realizes coarsest-wins: a reach with fine gold AND nuggets is
class 3, because the nuggets cannot have travelled far.
"""
from __future__ import annotations

import re

# Nouns that, right after a size word, mean it modifies sediment/rock/assay, not gold.
_NONGOLD = (r"grained|sand|sandy|gravel|gravels|silt|muck|micaceous|stratified|"
            r"material|schist|rock|clay|quartz sand|ounce|ounces|dwt")
_NONGOLD_AFTER = re.compile(rf"[\s,-]{{0,3}}(?:{_NONGOLD})\b")
# A size word counts only if "gold" (or "nugget", itself gold) is in its clause-window.
_GOLD = re.compile(r"gold|nugget")


def _gold_near(clause: str, start: int, end: int, window: int = 28) -> bool:
    return bool(_GOLD.search(clause[max(0, start - window):end + window]))


def mine_gold_coarseness(text: str) -> int | None:
    """Ordinal gold coarseness {3,2,1} or None, mined clause by clause."""
    t = re.sub(r"\s+", " ", str(text).lower())
    found: set[int] = set()
    for clause in re.split(r"[.;]", t):
        has_gold = bool(_GOLD.search(clause) or re.search(r"\bplacer\b", clause))

        # --- PROXIMAL (3) ---
        for m in re.finditer(r"nugget", clause):
            head = clause[m.start():m.start() + 24]
            if re.match(r"nugget\s+(creek|gulch|river|hill|mountain|peak|bench|claim)", head):
                continue  # place name, not a gold nugget
            if re.search(r"ounce|oz\b|\$|recover|found|weigh|worth|gold|placer|report", clause):
                found.add(3)
        if re.search(r"(rough|angular|wire|crystalline|sharp)\s+gold", clause):
            found.add(3)
        if re.search(r"gold[^.]{0,30}(rough|angular|wire|crystalline)\b", clause):
            found.add(3)
        if ("gold" in clause or "nugget" in clause) and re.search(
            r"attached\s+quartz|quartz[- ]attached|with\s+quartz", clause
        ):
            found.add(3)
        if re.search(r"coarse and rough|rough and coarse", clause):
            found.add(3)

        # --- COARSE (2) ---
        # In a gold-context clause "coarse" almost always describes the gold;
        # the only common non-gold use is "coarse gravel/sand", excluded here.
        if has_gold:
            for m in re.finditer(r"\bcoars\w*\b", clause):
                if _NONGOLD_AFTER.match(clause[m.end():m.end() + 14]):
                    continue
                found.add(2)
                break

        # --- DISTAL (1) ---
        # "fine/flat/flaky/flour" attach to sediment (and "flat" to bench/
        # terrace) far more than to gold, so each needs the gold-proximity bind
        # on top of the sediment/landform-context exclusion.
        if has_gold:
            for m in re.finditer(r"\b(fine|flaky|flour|flat)\b", clause):
                tail = clause[m.end():m.end() + 18]
                if re.search(rf"\b(?:{_NONGOLD})\b", tail):
                    continue
                if m.group(1) == "flat" and re.match(
                    r"[\s,-]*(bench|terrace|lying|country|area|topped|tundra|ground)", tail
                ):
                    continue
                if _gold_near(clause, m.start(), m.end(), window=32):
                    found.add(1)
                    break

    return max(found) if found else None
