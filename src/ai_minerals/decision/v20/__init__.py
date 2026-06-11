"""bcgt-v2.0 POMDP machinery: correlated draws + noisy sensors + multi-hypothesis.

Extends `src/ai_minerals/decision/` (v1.0, iid Bernoulli prior + noiseless
sensor + single hypothesis) with:

- B.1: correlated-prior (thresholded Gaussian random field over the v3 RF
       posterior surface) + importance-weighted particle filter for belief
       updates.
- B.2: retrospective BCGS validation against pre-2010 BCGS drill records.
- C.1: Bernoulli noisy sensor (α, β) + sensitivity sweep.
- C.2: multi-hypothesis belief + null-hypothesis falsification check.

See `research/pomdp_v20_implementation_plan.md` for parameter choices
and `research/bcgt_v20_b0_walkthrough.md` for the Julia reference
crosswalk.

NOTHING IN THIS DIRECTORY IS FUNCTIONAL YET. The skeleton was committed
on 2026-06-10 while v3.7 was still training; B.1 implementation kicks
off after H2.7 ships ai-minerals-v1.1.0.
"""
