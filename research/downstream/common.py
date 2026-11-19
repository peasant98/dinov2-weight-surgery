"""Shared for downstream tasks: the four backbones and a dense feature extractor on a common token grid."""
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WSP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WSP))
import dinocmp as dc  # noqa: E402
import wsp_edits as we  # noqa: E402

VARIANTS = runpy.run_path(str(Path(__file__).parent / "variants_ds.py"))["VARIANTS"]      # name -> edit spec on stock
MODELS = os.environ.get("MODELS", "stock,edits12,texture,dinov3").split(",")
PATCH = {"dinov3": 16}


def patch(name):
    return 16 if name.startswith("v3:") else PATCH.get(name, 14)


def load(name):
    if name == "dinov3":
        return dc.load_dinov3()
    if name.startswith("v3:"):                                                       # weight-edited DINOv3
        import wsp_v3
        return wsp_v3.build_v3(dc.load_dinov3(), wsp_v3.VARIANTS_V3[name])
    if name == "wsp":                                                                # the released edited weights
        sys.path.insert(0, str(WSP.parent))
        import dinov2_wsp
        w = WSP.parent / "weights/dinov2_vitl14_wsp.pth"
        return dinov2_wsp.dinov2_vitl14_wsp(weights=w if w.exists() else None).cuda()
    if name in VARIANTS:
        x_fit = dc.to_batch(dc.load_images(8), 448)[:24].cuda()                    # dev scan frames (closed-form fits)
        return we.build_variant(dc.load_dinov2("dinov2_vitl14"), VARIANTS[name], x_fit, verbose=False).eval()
    return dc.load_dinov2("dinov2_vitl14")


def to_tensor(images, hw):
    """PIL list -> normalised (N,3,H,W) at exactly hw=(H,W) pixels."""
    x = torch.stack([torch.from_numpy(np.asarray(im.convert("RGB").resize((hw[1], hw[0]), Image.BICUBIC))).permute(2, 0, 1)
                     for im in images])
    return ((x.float() / 255) - dc.MEAN) / dc.STD


@torch.no_grad()
def features(model, name, images, grid, bs=8, dtype=torch.float32):
    """(N, gh, gw, C) final-layer (normed) patch features, every model on the same gh x gw token grid."""
    gh, gw = grid; p = patch(name); out = []
    for i in range(0, len(images), bs):
        x = to_tensor(images[i:i + bs], (gh * p, gw * p)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L = os.environ.get("FEAT_LAYER")                                     # diagnostic only: an earlier block
            o = model.get_intermediate_layers(x, n=[int(L)] if L else 1, norm=True)[0]
        out.append(o.float().reshape(len(x), gh, gw, -1).to(dtype).cpu())
    return torch.cat(out)


def linear_probe(f, labs, ntr, K, ignore=255, steps=3000, bs=8192, seed=0, return_probe=False):
    """Train a linear layer on standardised patch features f (N,G,G,C) of the first ntr images against patch-level soft
    labels from labs (N,R,R; ignore pixels skipped); return (mIoU over present classes, pixel acc) on the rest."""
    import torch.nn.functional as F
    N, G = f.shape[0], f.shape[1]; R = labs.shape[-1]
    soft = []
    for i in range(0, N, 100):
        l = labs[i:i + 100].clone(); valid = l != ignore; l[~valid] = 0
        oh = F.one_hot(l, K).half() * valid[..., None]
        soft.append(F.avg_pool2d(oh.permute(0, 3, 1, 2), R // G))
    soft = torch.cat(soft)                                                        # (N, K, G, G)
    X = f[:ntr].reshape(-1, f.shape[-1]); Y = soft[:ntr].permute(0, 2, 3, 1).reshape(-1, K)
    keep = Y.sum(1) > 0.5; X, Y = X[keep], Y[keep] / Y[keep].sum(1, keepdim=True)
    mu, sd = X.float().mean(0), X.float().std(0) + 1e-4
    lin = torch.nn.Linear(f.shape[-1], K).cuda()
    opt = torch.optim.AdamW(lin.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        j = torch.randint(0, len(X), (bs,), generator=g)
        loss = -(Y[j].float().cuda() * F.log_softmax(lin(((X[j].float() - mu) / sd).cuda()), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    conf = torch.zeros(K, K, dtype=torch.long)
    with torch.no_grad():
        for i in range(ntr, N, 50):
            lg = lin(((f[i:i + 50].float() - mu) / sd).cuda()).permute(0, 3, 1, 2)
            pred = F.interpolate(lg, (R, R), mode="bilinear", align_corners=False).argmax(1).cpu()
            gt = labs[i:i + 50]; v = gt != ignore
            conf += torch.bincount(gt[v] * K + pred[v], minlength=K * K).reshape(K, K)
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)
    out = float(iou[conf.sum(1) > 0].mean()), float(conf.diag().sum() / conf.sum())
    return out + (lambda x: lin(((x.float() - mu) / sd).cuda()),) if return_probe else out


ADE_ROOT = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "benchmarks/ade/ADEChallengeData2016"


def ade_files(split="default", ntr=2000, nte=1000):
    """ADE20K image draws, each disjoint from the earlier ones. split 'unused' = all training images in no draw."""
    import numpy as np
    rng = np.random.default_rng(0)
    tr = sorted((ADE_ROOT / "images/training").glob("*.jpg")); va = sorted((ADE_ROOT / "images/validation").glob("*.jpg"))
    files = list(rng.choice(tr, ntr, replace=False)) + list(rng.choice(va, nte, replace=False))
    used = set()
    for seed in {"fresh2": [1], "fresh3": [1, 2], "unused": [1, 2]}.get(split, []):
        used |= set(files); rng = np.random.default_rng(seed)
        if seed == 2:                 # all 2000 val images are used by now: probe-test on unused TRAINING images
            files = list(rng.choice([f for f in tr if f not in used], ntr + nte, replace=False))
        else:
            files = list(rng.choice([f for f in tr if f not in used], ntr, replace=False)) + \
                list(rng.choice([f for f in va if f not in used], nte, replace=False))
    if split == "unused":
        used |= set(files)
        return [f for f in tr if f not in used]
    return files
