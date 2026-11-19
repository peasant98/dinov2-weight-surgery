# Round-2 rule (written 2026-11-12, BEFORE any result of the per-block damping grid; old test splits already seen)
Candidates: variants_ds.py round 7 (per-block texture damping, mid blocks 7-16 keep a, late 17-23 keep b, ratio thr r).
Stage 1: SPair-val (30/cat) for all; keep those >= stock (0.5574).
Stage 2: COCO-dev mIoU, DAVIS-train(12) J, look SELECT2 for stage-1 survivors (+ round-1 pick stock+tex0.3 noMS).
Pick = argmax over survivors with SELECT2 >= 0.86 of  min_task (score - stock_dev_score)
       [stock dev: COCO 0.5248, SPair 0.5574, DAVIS J 0.6360].
FRESH TEST (run once, pick vs stock vs round-1 pick vs DINOv3):
  ADE20K linear probe (train 2000 / val 1000 images), SPair test pairs disjoint from the 1080 used (seed 1, 60/cat),
  DAVIS-17 train videos not in the 12 dev videos (48 videos, J&F).
