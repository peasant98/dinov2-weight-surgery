"""FRESH TEST: ADE20K linear-probe semantic segmentation (150 classes, 'other' ignored). 2000 random training images
for the probe, 1000 random validation images for mIoU; square centre crop, 32x32 tokens, 128x128 labels.
Split "fresh2" draws 2000 train / 1000 val images disjoint from the default draw (round-3 fresh test).
Split "fresh3" (round 4): 3000 unused TRAINING images (all val images were used by then).
usage: ade_probe.py <out.json> [default|fresh2|fresh3]   (MODELS env = comma list)"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

ROOT = C.ADE_ROOT
G, R, NTR, NTE, K = 32, 128, 2000, 1000, 150
files = C.ade_files(sys.argv[2] if len(sys.argv) > 2 else "default", NTR, NTE)

def crop(im):
    s = min(im.size); x, y = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((x, y, x + s, y + s))


imgs, labs = [], []
for f in tqdm(files, desc="load"):
    imgs.append(crop(Image.open(f).convert("RGB")))
    a = f.parent.parent.parent / "annotations" / f.parent.name / (f.stem + ".png")
    l = np.array(crop(Image.open(a)).resize((R, R), Image.NEAREST)).astype(np.int64) - 1    # 0 'other' -> -1
    l[l < 0] = 255; labs.append(torch.from_numpy(l))
labs = torch.stack(labs)
res = json.load(open(sys.argv[1])) if Path(sys.argv[1]).exists() else {}
for name in C.MODELS:
    m = C.load(name)
    f = C.features(m, name, imgs, (G, G), dtype=torch.float16)
    del m; torch.cuda.empty_cache()
    miou, acc = C.linear_probe(f, labs, NTR, K)
    del f
    res[name] = dict(mIoU=miou, pixel_acc=acc)
    print(f"{name:8s} mIoU {miou:.4f}  pixel acc {acc:.4f}", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
