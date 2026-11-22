# DINOv3 round 2: gain edits in the late blocks (written 2026-11-20, before any result of this round)
Goal: an edited DINOv3 ViT-L/16 that is better on all three tasks, not only keypoints (round 1: the Fisher edit gave
SPair +3-4 but DAVIS J -0.001..-0.002 and failed the strict rule).

Stage 1, practice search (scan_v3.py, no dev / test data):
  candidates = one LayerScale gain (attention branch or MLP branch; 0, 0.5, 0.75, 1.25, 1.5) or one attention
  sharpness (query rows x0.8 / x1.25) in blocks 12-23 of stock DINOv3.
  practice sets, each split in halves A / B: SPair-71k TRAIN pairs (seed 30), DAVIS-2017 train videos outside the
  12-video dev subset (first 16, 30 frames each, J), ADE20K training images used in no earlier draw (ridge probe).
  singles on A; greedy over singles whose three A scores are all >= stock - 0.002 and whose mean relative gain is > 0,
  best first; an edit is kept only if, against the running model, the mean relative gain is > 0 on A and on B and no
  score drops by more than 0.002 on A or on B. One edit per (block, part).
Stage 2, dev (same dev splits as before: COCO-dev probe, SPair val 30/cat, DAVIS train(12) J):
  variants v3:gains, v3:gains+fisher b0.5 (and the known v3:fisher b0.5 as reference).
  eligible = every dev task >= stock DINOv3. Pick = eligible variant with the largest mean relative dev gain.
  If none is eligible: report that, and report the variants as non-picks.
Stage 3, fresh test, once: SPair test_fresh6 (seed-6 pairs, never used), DAVIS val and ADE fresh3 (both reported
  before, never used for selection -> labelled REUSED).
RESULT (11-20). Stage 1 kept (22 attn x0.5), (22 temp x0.8), (16 attn x1.5), (13 temp x1.25); practice SPair A/B
.580/.561 -> .595/.583, ADE ridge .307/.141 -> .311/.146, DAVIS J .764/.859 -> .764/.861.
Stage 2 dev (stock / gains / fisher b0.5 / gains+fisher): SPair val .5935 / .6093 / .6292 / .6324; DAVIS J .6955 /
.6949 / .6947 / .6937; COCO mIoU .5971 / .5947 / .5990 / .5971. No variant is >= stock on all three -> NO PICK;
stage 3 not run (SPair test_fresh6 still unused). The practice segmentation gain did not transfer to the COCO probe.
Addendum (11-20, written after the SPair-val / DAVIS-dev results of these variants, BEFORE their COCO-dev results and
before any test): new candidates v3:fisher+dpair w{0.3,1.0,3.0} b0.5 = one programmed discriminant pooled from the
SPair TRAIN keypoint scatter and a DAVIS TRAIN local-pair scatter (wsp_v3.dpair_scatter: token pairs within +-7
tokens across frames of the 48 train videos outside the dev subset; same label near a boundary vs different label).
Same eligibility (every dev task >= stock DINOv3) and same pick (largest mean relative dev gain). Dev so far
(SPair val / DAVIS J): stock .5935/.6955; w0.3 .6282/.6960; w1.0 .6249/.6977; w3.0 .6179/.7000; DAVIS-only
dfisher b0.5 .5747/.7013 (fails SPair). Feature-level diagnostics (v3:emu *, not weight edits): earlier layers / CLS /
smoothing do not help; sharpen 0.5: SPair .6008, J .6951.
Fresh test for the pick, once: SPair test_fresh6 (never used), DAVIS val + ADE fresh3 (REUSED, never used for selection).
RESULT dev COCO: w0.3 .5988, w1.0 .5993, w3.0 .5999 (stock .5971) -> all three eligible. Mean relative dev gain:
w0.3 +2.07 %, w1.0 +1.99 %, w3.0 +1.74 % -> PICK = v3:fisher+dpair w0.3 b0.5. Fresh test now, once.

FRESH (once, pick = v3:fisher+dpair w0.3 b0.5), stock -> pick:
- SPair test_fresh6 (new pairs, 7975 kp): PCK@0.1 .6015 -> .6390, PCK@0.05 .4324 -> .4685
- DAVIS val (REUSED split): J .7502 -> .7536, F .8032 -> .8131, J&F .7767 -> .7834
- ADE fresh3 (REUSED split): first attempt did not start (RAM preflight); rerun one
  model per process as fresh10_ade_*.json (same pick, no reselection).
- ADE fresh3 (REUSED split; probe RAM cut, stock reproduces .4493 -> .4492): mIoU .4492 -> .4512,
  pixel acc .7958 -> .7962. Pick is above stock on all three fresh tests. Seed check of the
  probe (SEED=1,2) run afterwards as fresh10_ade_s*.json.
- Diagnostic after the pick: similarity-weighted smoothing on top (v3:pick10+nlm ...) lowers
  dev DAVIS J (.6960 -> .688-.695) and dev SPair (.628 -> .621-.626). Dropped.
- ADE probe seeds 0/1/2, stock -> pick: .4492/.4485/.4492 -> .4512/.4516/.4506 (pick above
  stock on every seed; mean +0.22 mIoU, seed spread about 0.05-0.1).

ROUND 11 (11-21, same rule: every dev task >= stock, pick = largest mean relative dev gain, then fresh once).
Dev stock SPair .5935 / DAVIS J .6955 / COCO .5971; round-10 pick .6282 / .6960 / .5988 (+2.07 %).
- v3:nfisher s0.5/1/2 (nonlinear lift): SPair .6224/.6141/.5958, J .6924/.6881/.6872 -> not eligible.
- v3:rope0.5/0.7/1.4/2.0: SPair .5797/.5852/.5963/.5933, J .6901/.6939/.6934/.6937,
  COCO .5929/.5954/.5979/.5956 -> not eligible (video).
- v3:fisher mix3 d0.3 s0.3/1.0/3.0 (adds ADE-unused class scatter): SPair .6344/.6247/.6164,
  J .6964/.6983/.6999, COCO .5999/.5994/.6004 -> all eligible, mean rel gain +2.50/+2.02/+1.68 %.
Round 12 (weight grid of mix3, RoPE + mix3) is defined before its results; the pick of rounds 11+12 is
the eligible candidate with the largest mean relative dev gain; fresh test once after round 12.
ROUND 12 dev (SPair / J / COCO): mix3 d1.0 s0.3 .6231/.7001/.5999; d1.0 s1.0 .6221/.7012/.6006;
d3.0 s1.0 .6171/.7018/.6013; d0.3 s0.5 .6322/.6968/.6000; d3.0 s3.0 .6126/.7024/.6006;
rope1.4+mix3s0.3 .6339/.6982/.6008; rope2.0+mix3s0.3 .6299/.6971/.5995. All eligible.
Ranking over rounds 11+12 (mean relative dev gain): rope1.4+mix3s0.3 +2.61 %, mix3 d0.3 s0.3 +2.49 %,
d0.3 s0.5 +2.40 %, ... round-10 pick +2.07 %.
PICK (rounds 11+12) = v3:rope1.4+mix3s0.3 = RoPE coordinates x1.4, then the pooled Fisher block
(SPair keypoints + 0.3 DAVIS-train pairs + 0.3 ADE-unused classes, beta 0.5).
Fresh test once: SPair test_fresh7 (new pairs), ADE fresh4 (new 3000 images, disjoint from every draw and from
the seg_scatter fit images; probe seeds 0/1/2), DAVIS val (REUSED). Stock and the round-10 pick run alongside.
FRESH (once, rounds 11+12 pick v3:rope1.4+mix3s0.3), stock / round-10 pick / new pick:
- SPair test_fresh7 (new pairs, 7784 kp): PCK@0.1 .5911 / .6289 / .6328; PCK@0.05 .4174 / .4552 / .4611
- ADE fresh4 (new 3000 images), probe seeds 0/1/2: stock .4371/.4391/.4374, round-10 .4389/.4401/.4384,
  new .4392/.4402/.4394 (mean .4379 / .4391 / .4396)
- DAVIS val (REUSED): J&F .7767 / .7835 / .7831; J .7502 / .7537 / .7529; F .8032 / .8134 / .8132
New pick > stock on all three fresh tests; vs the round-10 pick: SPair +0.4, ADE +0.05, DAVIS -0.04 (tie).
