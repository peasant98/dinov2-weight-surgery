"""Downstream 3: semi-supervised video object segmentation by label propagation on DAVIS-2017 val (30 videos), DINO
protocol (no training): context = first frame + last 7 frames, cosine affinity / 0.1, top-5 per target patch, spatial
neighbourhood of 7 patches (~DINO's 12 patches at 8 px), same 34x61 token grid for every model. J = region IoU,
F = boundary F-measure (DAVIS-style: 0.008*diag tolerance). First frame excluded from scores.
usage: davis.py <out.json> [n_videos] [val|train] [noF]   (MODELS env = comma list; train = selection split)"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

ROOT = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "benchmarks/davis/DAVIS"
GH, GW, NLAST, TOPK, TEMP, RAD = 34, 61, 7, 5, 0.1, 7
SET = sys.argv[3] if len(sys.argv) > 3 else "val"
NOF = len(sys.argv) > 4 and sys.argv[4] == "noF"
vids = (ROOT / f"ImageSets/2017/{SET.replace('_fresh', '')}.txt").read_text().split()
if SET == "train_fresh":                                     # train videos NOT in the 12-video dev subset
    dev = set(vids[::len(vids) // 12][:12]); vids = [v for v in vids if v not in dev]
elif len(sys.argv) > 2 and int(sys.argv[2]) > 0:
    vids = vids[::max(1, len(vids) // int(sys.argv[2]))][:int(sys.argv[2])]
yy, xx = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
pos = torch.stack([yy.flatten(), xx.flatten()], -1).float()
NB = ((pos[:, None] - pos[None]).abs().max(-1).values <= RAD).cuda()          # (P, P) neighbourhood mask


def boundary(m):
    return m ^ binary_erosion(m, iterations=1, border_value=0)


def f_measure(pred, gt):
    bp, bg = boundary(pred), boundary(gt)
    r = max(1, int(np.ceil(0.008 * np.hypot(*gt.shape))))
    dp, dg = binary_dilation(bp, disk(r)), binary_dilation(bg, disk(r))
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    prec = (bp & dg).sum() / max(bp.sum(), 1); rec = (bg & dp).sum() / max(bg.sum(), 1)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


@torch.no_grad()
def propagate(feats, lab0, K):
    """feats (T, P, C) unit; lab0 (P, K) soft first-frame labels -> list of (P, K) per frame."""
    segs = [lab0]
    for t in range(1, len(feats)):
        ctx = [0] + list(range(max(1, t - NLAST), t))
        src = feats[ctx]                                                          # (S, P, C)
        aff = torch.einsum("pc,sqc->psq", feats[t], src) / TEMP                   # (P, S, P)
        aff = aff.masked_fill(~NB[:, None, :], -1e4).reshape(len(feats[t]), -1)
        v, i = aff.topk(TOPK, dim=1)
        w = torch.softmax(v, 1)
        lab = torch.stack([segs[c] for c in ctx]).reshape(-1, K)                  # (S*P, K)
        segs.append((w[..., None] * lab[i]).sum(1))
    return segs


res = {}
for name in C.MODELS:
    m = C.load(name); J, Fm, per = [], [], {}
    for v in vids:
        fr = sorted((ROOT / "JPEGImages/480p" / v).glob("*.jpg"))
        an = sorted((ROOT / "Annotations/480p" / v).glob("*.png"))
        imgs = [Image.open(f) for f in fr]
        f = F.normalize(C.features(m, name, imgs, (GH, GW)), dim=-1).reshape(len(imgs), GH * GW, -1).cuda()
        g0 = np.array(Image.open(an[0])); objs = [o for o in np.unique(g0) if o != 0]; K = len(objs) + 1
        oh = torch.from_numpy(np.stack([g0 == o for o in [0] + objs])).float()[None]
        lab0 = F.adaptive_avg_pool2d(oh, (GH, GW))[0].reshape(K, -1).T.cuda()
        segs = propagate(f, lab0, K)
        H, W = g0.shape; jv, fv = [], []
        for t in range(1, len(imgs)):
            up = F.interpolate(segs[t].T.reshape(1, K, GH, GW), (H, W), mode="bilinear", align_corners=False)[0]
            pred = up.argmax(0).cpu().numpy(); gt = np.array(Image.open(an[t]))
            for k, o in enumerate(objs, 1):
                p, g = pred == k, gt == o
                jv.append(1.0 if (p | g).sum() == 0 else (p & g).sum() / (p | g).sum())
                fv.append(np.nan if NOF else f_measure(p, g))
        per[v] = (float(np.mean(jv)), float(np.mean(fv))); J += jv; Fm += fv
    del m; torch.cuda.empty_cache()
    j, fm = float(np.mean([p[0] for p in per.values()])), float(np.mean([p[1] for p in per.values()]))
    res[name] = dict(J=j, F=fm, JF=(j + fm) / 2, per_video=per)
    print(f"{name:8s} J&F {(j + fm) / 2:.4f}  J {j:.4f}  F {fm:.4f}  ({len(vids)} videos)", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
