# DINOv3 weight-space edit rule (written 2026-11-12 before any task result of an edited DINOv3)
Candidates: wsp_v3.VARIANTS_V3 (positional-subspace projector, k in {4,16,64}, alpha in {.25,.5,1}).
Dev: COCO-dev probe, SPair-val (30/cat), DAVIS-train(12) J - same splits as for DINOv2.
Eligible: every dev task >= stock DINOv3 dev. Pick = eligible variant with the largest mean dev gain over the 3 tasks.
If none is eligible, report that DINOv3 was not improved. Fresh test once: ADE fresh3, SPair seed-3 pairs, DAVIS val.
Addendum (before any result of them): candidates also include the v3:white* variants (programmed partial whitening).
DINOv2 whitening variants (PICK6+white*, TD2+white) are judged by SELECTION_RULE_v4 round-5 eligibility (>= stock on all
3 dev tasks) and must also beat PICK6 on mean dev gain to replace it.
Addendum 2 (before any result): v3:fisher b{0.25,0.5,1.0} (programmed Fisher discriminant from SPair TRAIN keypoints).
Addendum 3: also PICK6+fisher b0.25/b0.5 (DINOv2).
Addendum 4 (before any val/COCO/DAVIS result of them): v3:heads (keep-if-wins head edits from head_loop.py on SPair
TRAIN A/B) and v3:heads+pos16 a0.25. Whitening on DINOv3 failed COCO (>= rule) -> out.
DINOv2: PICK6+white b0.5 e1.0 passes (SPair .5672, J .6404, COCO .5403 >= stock) and beats PICK6 on mean dev gain.
RESULT so far: v3:fisher b0.25/0.5/1.0 DAVIS J .6943/.6947/.6896 < .6955 -> fail strict rule (SPair +3-4, COCO +.1-.4).
DINOv2: PICK6+fisher b0.5 eligible (SPair .5875, COCO .5454, J .6394) and best mean dev gain -> DINOv2 candidate.
Addendum 5 (before results): v3:heads+fisher b0.5; DINOv2 head loop restarted on PICK6+fisher b0.5.
RESULT (dev): v3:heads J .6925, v3:heads+pos16 .6921, v3:heads+fisher .6856 -> all fail DAVIS >= .6955.
DINOv3 strict pick = v3:pos16 a0.25 (only eligible). Reported as NON-picks (keypoint-specialised): v3:fisher b0.5,
v3:heads+fisher b0.5. DINOv2 pick = PICK6+fisher b0.5 (head loop on it kept 1 edit, B +0.3: not added).
FRESH6 (once): SPair test_fresh4 (seed 4 pairs, never used); ADE fresh3 + DAVIS val + look test6 REUSED (reported
before, never used for selection).
Addendum 5 (11-12 late, before any result): LABEL-FREE Fisher (no SPair keypoints): within-match scatter from
(v) mutual-NN token matches between DAVIS-train frames t, t+2 (48 fit videos = train minus the 12 dev videos) or
(a) crop/flip/colour-jitter views of the same DAVIS-train frames (known geometry). Variants: v3:vfisher b0.5,
v3:afisher b0.5, PICK6+vfisher b0.5, PICK6+afisher b0.5. Same dev tasks and same eligibility rule; compared with the
SPair-label Fisher on every dev task. Fresh check if any is eligible: ADE fresh3, SPair test_fresh4, DAVIS val (all
already reported once -> labelled REUSED), plus look test6 for DINOv2.
