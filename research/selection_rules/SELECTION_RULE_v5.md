# Round-6 rule (written 2026-11-18 BEFORE any result of these variants). Goal: close the DAVIS gap.
Candidates (DINOv2, all on TD2 + PICK6 side channel, closed-form):
  PICK6+dfisher b{0.25,0.5}         Fisher from DAVIS-2017 TRAIN object masks (48 fit videos = train minus the 12 dev
                                    videos): within-instance vs within-video total scatter of tokens (purity >= .75).
  PICK6+sp+d{0.5,1} fisher b0.5     pooled LDA: SPair-TRAIN keypoint problem + w * DAVIS-mask problem (each / tr(cov_t)).
Reference = current pick PICK6+fisher b0.5 (dev SPair-val .5875, COCO-dev .5454, DAVIS-train(12) J .6394).
Eligible: COCO-dev >= stock .5248, SPair-val >= stock .5574, DAVIS J >= stock .6360 (same as round 5).
Pick = eligible candidate with the largest MEAN dev gain over stock across the 3 tasks; it replaces the reference only
if that mean gain beats the reference's. Otherwise report the reference as still best.
FRESH TEST (once, never used for selection): DAVIS-2017 val J&F (standard benchmark; reported before, never selected on),
SPair TEST pairs seed 5 (new, never used), ADE fresh3 (reported before).
Addendum 1 (after the 4 candidates' SPair-val only; before ANY result of these): also "TTR PICK6+fisher b0.5" and
"TTR PICK6+sp+d0.5 fisher b0.5" (programmed test-time register, TTR PICK6 dev J .6450 / SPair .5508, + Fisher). Same rule.
Addendum 2 (after DAVIS-val-free dev results of the 4 original candidates: linear DAVIS-mask Fisher gives at most
+.006 J; before ANY result of these): hypothesis (unverified) = DAVIS propagation inside a 7-patch window benefits
from MORE positional signal (v3 has 16.8 % positional variance vs v2 12 %; removing it did not help DAVIS).
Candidates "PICK6+fisher b0.5+pos16 a-0.5" / "a-1.0" (positional subspace AMPLIFIED, projector in the last 2k MLP
units of the appended block so it does not overwrite the Fisher map). Same rule.
Addendum 3: pos-amplified variants stopped after DAVIS dev (no gain: .6394/.6398) - not eligible to win; SPair/COCO not run.
Addendum 4 (11:58, before ANY result of these): diagnostic layer sweep, stock DINOv2, DAVIS-train(12) dev J by block
read out: L13 .634, L16 .644, L19 .651, L21 .652, L22 .650, final .636 -> the last blocks cost propagation.
Candidates (weight edit = LayerScale damping of original blocks before the side channel / Fisher):
"PICK6+fisher b0.5+D23x0", "PICK6+fisher b0.5+D22-23x0.5", "PICK6+fisher b0.5+D22-23x0". Same eligibility and pick rule.
Addendum 5 (12:35, after addendum-4 dev results: D23x0 J .6482 but COCO .5208 FAIL; D22-23x0.5 eligible, mean gain
.0078 < ref .0180; D22-23x0 fails): candidates "PICK6+fisher b0.5+D23attn0" / "+D23mlp0" (only block 23's attention
or only its MLP turned off, attn_dominant gains). Same rule.
RESULT addendum 5 (dev): D23mlp0 SPair .5950 / COCO .5479 / J .6408 -> eligible, mean gain .0218 > ref .0180 => PICK9 =
"PICK6+fisher b0.5+D23mlp0". D23attn0 fails COCO (.4958).
Addendum 6 (12:58, before ANY result of these): extend the MLP removal one / two blocks: "+D22-23mlp0", "+D21-23mlp0".
Same eligibility; one of them replaces PICK9 only if its mean dev gain beats .0218.
RESULT addendum 6: D22-23mlp0 (COCO .5335, J .6361, SPair .5646) and D21-23mlp0 (.4892/.6361/.5059) both worse -> PICK9 = PICK6+fisher b0.5+D23mlp0. Fresh test launched 13:2x (run_fresh9.sh).
Addendum 7 (13:15, before ANY result of these; numbers goal, base PICK9): label-free head scores (global_heads.py,
attention-weighted query->key distance on 32 COCO-dev images; uniform ~13.9): block 23 most global heads 12,8,3,1
(19.9-18.6), block 22: 1,3,0 (18.5-17.3). Rationale: D23attn0 gave DAVIS J +.007 over PICK9 but killed COCO ->
remove only the global heads. Candidates "P9-g23h4" (zero block-23 heads 12,8,3,1) and "P9-g23h4+g22h3" (+ block-22
heads 1,3,0). Same eligibility; replaces PICK9 only if mean dev gain > .0218.
RESULT addendum 7 (dev): P9-g23h4 COCO .5385 / SPair .5923 / J .6404 (mean gain .0177); +g22h3 .5330/.5644/.6443 (.0078) -> neither beats PICK9 (.0218). PICK9 final. Look SELECT2 .9136/.9160 vs PICK9 .9142.
