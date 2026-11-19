# Round-3 rule (written 2026-11-12 before any result of the side-channel bilateral variants)
Base TD = texture damping mid0.5/late0.5 r>0.9 (round-2 pick). Candidates = TD + side_channel(A, beta, gamma) grid, and TD.
Stage 1: SPair-val (30/cat) >= stock (0.5574). Stage 2 (survivors): COCO-dev, DAVIS-train(12) J, look SELECT2.
Eligible = every dev task >= stock dev (COCO 0.5248, SPair 0.5574, DAVIS J 0.6360).
Pick = eligible variant with the highest SELECT2 (look); tie-break by min-gain.
FRESH TEST (once): ADE20K probe on 2000 train + 1000 val images disjoint from the round-2 ADE images;
SPair test pairs disjoint from seeds 0 and 1 (seed 2, 60/cat); DAVIS-17 val (30 videos; used once in round-1 test,
never for selection); DINOv3 look on TEST4 scans at different frames (offset 0.5).
