"""Downstream 1: linear-probe semantic segmentation on COCO val2017 (80 thing classes + background).
Square centre crop, 32x32 tokens; train a linear layer on 2000 images (patch-level soft labels), test mIoU on 1000
other images at 128x128 (logits upsampled). Identical protocol for every backbone.
Split "dev" (selection) uses 1500 train / 500 eval images disjoint from the "test" 2000/1000.
usage: coco_probe.py <out.json> [test|dev]   (MODELS env = comma list)"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

ROOT = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "coco"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
G, R = 32, 128
NTR, NTE = (2000, 1000) if SPLIT == "test" else (1500, 500)
coco = COCO(str(ROOT / "annotations/instances_val2017.json"))
cat2idx = {c: i + 1 for i, c in enumerate(sorted(coco.getCatIds()))}
K = len(cat2idx) + 1
ids = sorted(coco.getImgIds())
rng = np.random.default_rng(0); rng.shuffle(ids)
ids = ids[:3000] if SPLIT == "test" else ids[3000:5000]


def crop(a):
    h, w = a.shape[:2]; s = min(h, w); y, x = (h - s) // 2, (w - s) // 2
    return a[y:y + s, x:x + s]


imgs, labs = [], []
for i in tqdm(ids, desc="load"):
    info = coco.loadImgs(i)[0]
    im = Image.fromarray(crop(np.asarray(Image.open(ROOT / "val2017" / info["file_name"]).convert("RGB"))))
    sem = np.zeros((info["height"], info["width"]), np.uint8)
    for a in sorted(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=None)), key=lambda a: -a["area"]):
        sem[coco.annToMask(a) > 0] = cat2idx[a["category_id"]]
    imgs.append(im)
    labs.append(torch.from_numpy(np.asarray(Image.fromarray(crop(sem)).resize((R, R), Image.NEAREST))).long())
labs = torch.stack(labs)                                                          # (N, R, R)
soft = torch.cat([F.avg_pool2d(F.one_hot(labs[i:i + 100], K).permute(0, 3, 1, 2).half(), R // G)
                  for i in range(0, len(labs), 100)])                            # (N, K, G, G) patch label shares, fp16

res = {}
for name in C.MODELS:
    m = C.load(name)
    f = C.features(m, name, imgs, (G, G)).half()                                   # (N, G, G, C)
    del m; torch.cuda.empty_cache()
    Xtr = f[:NTR].reshape(-1, f.shape[-1]); Ytr = soft[:NTR].permute(0, 2, 3, 1).reshape(-1, K)
    mu, sd = Xtr.float().mean(0), Xtr.float().std(0) + 1e-4
    lin = torch.nn.Linear(f.shape[-1], K).cuda()
    opt = torch.optim.AdamW(lin.parameters(), 1e-3, weight_decay=1e-4)
    steps, bs = 3000, 8192
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        j = torch.randint(0, len(Xtr), (bs,), generator=g)
        x = ((Xtr[j].float() - mu) / sd).cuda(); y = Ytr[j].cuda()
        loss = -(y * F.log_softmax(lin(x), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    conf = torch.zeros(K, K, dtype=torch.long)
    with torch.no_grad():
        for i in range(NTR, NTR + NTE, 50):
            x = ((f[i:i + 50].float() - mu) / sd).cuda()
            lg = lin(x).permute(0, 3, 1, 2)
            pred = F.interpolate(lg, (R, R), mode="bilinear", align_corners=False).argmax(1).cpu()
            conf += torch.bincount(labs[i:i + 50].flatten() * K + pred.flatten(), minlength=K * K).reshape(K, K)
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)
    present = conf.sum(1) > 0
    res[name] = dict(mIoU=float(iou[present].mean()), pixel_acc=float(conf.diag().sum() / conf.sum()), final_loss=float(loss))
    print(f"{name:8s} mIoU {res[name]['mIoU']:.4f}  pixel acc {res[name]['pixel_acc']:.4f}  (train loss {float(loss):.3f})", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
