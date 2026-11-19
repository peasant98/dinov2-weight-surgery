"""Downstream 2: zero-shot semantic keypoint correspondence on SPair-71k test (no training).
60 random test pairs per category (18 cats). Each image resized to a 40x40 token grid; a source keypoint's feature is
bilinearly sampled, matched by cosine argmax over target patches; PCK@0.1 / @0.05 of max(target bbox side), per keypoint.
usage: spair.py <out.json> [test|val] [per_cat]   (MODELS env = comma list)"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

ROOT = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "benchmarks/spair/SPair-71k"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
G, PER_CAT = 40, int(sys.argv[3]) if len(sys.argv) > 3 else 60
FRESH = SPLIT in ("test_fresh", "test_fresh2", "test_fresh3", "test_fresh4", "test_fresh5")              # pairs disjoint from the 60/cat (seed 0) already used
files = sorted((ROOT / "PairAnnotation" / ("test" if FRESH else SPLIT)).glob("*.json"))
bycat = {}
for f in files:
    bycat.setdefault(f.stem.split(":")[1], []).append(f)
rng = np.random.default_rng(0)
chosen = [f for c in sorted(bycat) for f in rng.choice(bycat[c], min(PER_CAT, len(bycat[c])), replace=False)]
if FRESH:
    used = set(chosen); rng = np.random.default_rng(1)
    chosen = [f for c in sorted(bycat) for f in rng.choice([x for x in bycat[c] if x not in used], PER_CAT, replace=False)]
    for seed in {"test_fresh2": [2], "test_fresh3": [2, 3], "test_fresh4": [2, 3, 4], "test_fresh5": [2, 3, 4, 5]}.get(SPLIT, []):   # disjoint from all earlier draws
        used |= set(chosen); rng = np.random.default_rng(seed)
        chosen = [f for c in sorted(bycat) for f in rng.choice([x for x in bycat[c] if x not in used], PER_CAT, replace=False)]
pairs = [json.load(open(f)) for f in chosen]
names = sorted({(p["category"], p[k]) for p in pairs for k in ("src_imname", "trg_imname")})
idx = {n: i for i, n in enumerate(names)}
imgs = [Image.open(ROOT / "JPEGImages" / c / n) for c, n in names]
sizes = [im.size for im in imgs]                                            # (W, H)
print(len(pairs), "pairs,", len(names), "images", flush=True)

res = {}
for name in C.MODELS:
    m = C.load(name)
    feats = F.normalize(C.features(m, name, imgs, (G, G)), dim=-1)          # (N, G, G, C)
    del m; torch.cuda.empty_cache()
    hit10, hit05, percat = [], [], {}
    for p in pairs:
        s, t = idx[(p["category"], p["src_imname"])], idx[(p["category"], p["trg_imname"])]
        (sw, sh), (tw, th) = sizes[s], sizes[t]
        kp = torch.tensor(p["src_kps"], dtype=torch.float32); tk = torch.tensor(p["trg_kps"], dtype=torch.float32)
        grid = torch.stack([kp[:, 0] / sw * 2 - 1, kp[:, 1] / sh * 2 - 1], -1).view(1, -1, 1, 2)
        q = F.grid_sample(feats[s].permute(2, 0, 1)[None], grid, align_corners=False)[0, :, :, 0].T    # (K, C)
        sim = F.normalize(q, dim=-1) @ feats[t].reshape(-1, feats.shape[-1]).T                         # (K, G*G)
        a = sim.argmax(1); py, px = a // G, a % G
        pred = torch.stack([(px + 0.5) / G * tw, (py + 0.5) / G * th], -1)
        bb = p["trg_bndbox"]; thr = max(bb[2] - bb[0], bb[3] - bb[1])
        d = (pred - tk).norm(dim=-1)
        hit10 += (d <= 0.1 * thr).tolist(); hit05 += (d <= 0.05 * thr).tolist()
        percat.setdefault(p["category"], []).extend((d <= 0.1 * thr).tolist())
    res[name] = dict(pck10=float(np.mean(hit10)), pck05=float(np.mean(hit05)),
                     per_cat={c: float(np.mean(v)) for c, v in percat.items()})
    print(f"{name:8s} PCK@0.1 {res[name]['pck10']:.4f}  PCK@0.05 {res[name]['pck05']:.4f}  ({len(hit10)} keypoints)", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
