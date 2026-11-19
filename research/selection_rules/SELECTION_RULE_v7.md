# Round-18 rule (written 2026-11-18 ~13:45 BEFORE any result). Idea from the DINOv3 paper (Gram anchoring, arXiv
2508.10104 Sec 4): match the PATCH GRAM of the final features to a cleaner earlier "Gram teacher"; HRef variant = teacher
features computed at 2x input resolution and bicubic-downsampled. Weight-edit analogue (no gradients): a closed-form linear
map A on the final normalised features, A = M^(1/2), M = C^-1 R C^-1 (ridge), C = sum X^T X, R = sum (X^T Y)(X^T Y)^T,
X = L2-normalised final patch features of PICK9, Y = L2-normalised block-21 features of the SAME model (teacher layer
chosen from the round-16 DAVIS layer sweep: stock L21 .652 vs final .636). Fit images = COCO-dev pos_images (256).
Installed as a fixed linear layer after the final LayerNorm.
Candidates: "P9+gram L21" (1x teacher), "P9+gram L21 href" (2x teacher, downsampled), "P9+gram L21 href a0.5" (A blended 50 % with I).
Eligible: dev floors (SPair-val >= .5574, COCO-dev >= .5248, DAVIS-train(12) J >= .6360). Pick = largest mean dev gain over
stock; replaces PICK9 only if > .0218. Fresh (once): DAVIS val, SPair test_fresh5, ADE fresh3 (all reported before, never
used for selection).
RESULT v1 (dev DAVIS J .6403/.6405/.6396 ~ PICK9 .6408) is INVALID as a test of the idea: diagnostic on 16 unseen dev images
shows the map moves the output Gram AWAY from the teacher (Gram corr .820 -> .740). Cause: the pooled approximation
sum_i C_i M C_i ~ n C M C is wrong (per-image covariances differ). Addendum 1 (before any result of these): replace by a
ridge-regression readout W = argmin ||X W - Y||^2 + lam ||W||^2 on raw final-LN outputs (X) and the teacher's LN outputs (Y),
installed the same way; gate = Gram corr to teacher must RISE on unseen dev images before any downstream run.
Candidates "P9+gramreg L21", "P9+gramreg L21 href". Same eligibility / pick rule.
Gate (Addendum 1): Gram corr to teacher on 16 unseen dev images: gramreg L21 .877->.975, gramreg L21 href .819->.940 -> passes.
RESULT addendum 1 (dev): gramreg L21 J .6541 / SPair .3831 / COCO .5395; href J .6524 / SPair .3015 / COCO .5402 ->
both FAIL SPair (block-21 Gram has lost cross-instance semantic correspondence). Addendum 2 (before any result):
partial anchoring out = X (I + alpha (W - I)), alpha .25 / .5 ("P9+gramreg L21 a0.25", "a0.5"). Same rule.
RESULT addendum 2 (dev): a0.25 J .6455 / SPair .5501, a0.5 J .6492 / SPair .4928 -> both fail the SPair floor (COCO not run). Gram anchoring to block 21 trades keypoints for propagation along one curve; PICK9 stays.
