# Round-4 rule (written 2026-11-12 BEFORE any downstream result of the rolling-guidance variants)
Candidates: PICK6 (TD2+bil A4 b0 g0.5) and the RG variants in variants_ds.py round 10 (PICK6 + 1-3 further
programmed bilateral blocks whose edge guide is re-read from the filtered stream; chosen on DINOv3-free look proxies).
Dev: COCO-dev probe, SPair-val (30/cat), DAVIS-train(12) J, look SELECT2 (dev_lookP/Q).
Eligible: COCO-dev >= stock (0.5248), DAVIS J >= stock (0.6360), SPair-val >= stock - 0.010 (0.5474).
  (Relaxed SPair tolerance is a stated trade: the goal of this round was a better LOOK; every spatial filter so far cost
  SPair 0-1 pt, and the fresh tests report the cost.)
Pick = eligible variant with the highest SELECT2; within 0.002 of the best, the one with higher SPair-val.
FRESH TEST (once): ADE20K probe split fresh3 (3000 unused training images, 2000 probe / 1000 test);
SPair test pairs seed 3 (disjoint from seeds 0-2, 60/cat); DAVIS-17 val J&F (3rd use, never used for selection);
DINOv3 look on 48 ADE training images in NO probe draw (seed 4) + 4 frames of an unused scan of ours.

Addendum (17:38, before ANY downstream or look result of these): candidates also include the round-11 "TTR ..."
variants (same filters + programmed test-time register). Same eligibility, same pick criterion.
Addendum 2 (before any result for them): also "TTR mild A4b8g0.3/0.5" (one weak guided block after PICK6's filter).
Observed so far (dev, round 10): every RG variant fails SPair-val (0.41-0.53 vs 0.5474 floor) -> ineligible.
RESULT (dev): eligible = TTR TD2 (SELECT2 .8860), PICK6 (.9014), TTR PICK6 (.9040), TTR mild A4b8g0.3 (.9101; SPair .5476,
J .6417, COCO .5434). PICK7 = TTR mild A4b8g0.3. Fresh test also reports "TTR RG A2b8g1x2" as best-look NON-pick.

# Round-5 rule (written after dev SPair/COCO/DAVIS of the pos variants were seen, BEFORE their look scores and
# before any fresh result): candidates = PICK7 + positional projector variants with all three dev tasks measured
# (r13: pos4/16/64/256/16a0.5; r15: pos32a0.5, pos64a0.5, TTR TD2+pos16a0.5 no-filter).
# Eligible: COCO-dev >= 0.5248, DAVIS J >= 0.6360, SPair-val >= 0.5574 (stock; no tolerance this time).
# Pick = highest SELECT2. Fresh test: as round 4 (ADE fresh3, SPair seed 3, DAVIS val, look test6).
# Note: ADE fresh3 was partly run (stock .3954, PICK6 .4150, PICK7 .4138, RG .4061) before the round-4 eval was
# stopped; those numbers did not drive this edit (it targets position, chosen from SPair-val).
RESULT (round 5): eligible with all 3 dev tasks: PICK7+pos16 (SELECT2 .9110; COCO .5250, J .6387, SPair .5629),
pos16 a0.5 (.9080), pos32 a0.5 (.9081); pos64 a0.5 / no-filter COCO not finished but their SELECT2 (.9072/.8802)
cannot win. PICK8 = PICK7+pos16.
