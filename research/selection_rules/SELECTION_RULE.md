# Pre-committed selection rule (written 2026-11-12, before any dev-split result of the spectral variants)
Candidates: variants_ds.py (stock + spectral-block variants). DINOv3 is never used for selection.
Eligible = on the DEV splits: COCO-dev linear-probe mIoU >= stock - 0.005, SPair-val PCK@0.1 >= stock,
DAVIS-train J >= stock.
Pick = eligible variant with the highest LOOK score = SELECT2 from search.py on the dev scans
(mean of centred SAM-segment AUC and crop consistency). If no variant is eligible, report failure.
Then run the pick ONCE on the test splits (COCO test 2000/1000, SPair test, DAVIS-17 val) + the test3 DINOv3 look test.
