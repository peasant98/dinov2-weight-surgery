# Round-17 rule (written 2026-11-18 13:05 BEFORE any result of these variants). Goal: make the features look better.
Base = PICK9 (PICK6+fisher b0.5+D23mlp0). Candidates = PICK9 with more / stronger programmed bilateral smoothing:
  P9+mild g0.3 / g0.5   one extra guided bilateral block (A4, beta 8, gamma .3 / .5) after PICK6's filter
  P9+RG A2b8g1x2        two rolling-guidance blocks (best-look non-pick of round 4)
  P9 main g1.0          PICK6's own filter at gamma 1.0 instead of 0.5
Look metric = SELECT2 on dev (search.py; DINOv3-free; same as rounds 4-5). Eligible = dev floors as rounds 5/6
(SPair-val >= .5574, COCO-dev >= .5248, DAVIS-train(12) J >= .6360). Pick = eligible candidate with the HIGHEST
SELECT2, and only if it beats PICK9's SELECT2; else PICK9 stays. (Stated trade: the look goal may spend part of
PICK9's dev margin on SPair/COCO/DAVIS, but never below stock.)
Fresh report (once): look test6 (CKA / sim-map vs DINOv3), SPair test_fresh5, DAVIS val, ADE fresh3 (all reported before
for PICK9, never used for selection).
RESULT (dev): DAVIS J mild g0.3 .6332, mild g0.5 .6274, RG .5789, main g1.0 .6346 -> ALL below the .6360 floor (smoothing
blurs boundaries that propagation needs); SPair .5870/.5782/.4456/.5855. None eligible -> PICK9 stays. COCO not run.
