"""Shared helpers: load DINOv2 / DINOv3, extract dense features on a common 32x32 grid, and score how close a
DINOv2 variant is to DINOv3 (CKA, patch-similarity agreement, k-means agreement, high-norm artifact tokens)."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

DATA = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads")))       # benchmarks and fit images live here
DINOV2_REPO = Path(os.environ.get("WSP_DINOV2_REPO", str(Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")))
DINOV3_REPO = Path(os.environ.get("WSP_DINOV3_REPO", str(DATA / "dinov3")))                 # only for the DINOv3 reference rows
DINOV3_W = Path(os.environ.get("WSP_DINOV3_WEIGHTS", str(DATA / "dinov3_vitl16_pretrain_lvd1689m.pth")))
GRID = 32
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# Fit images for the closed-form calibrations: folders DATA/<name>/images. We used frames of six of our own video scans
# (not released); any ~24 natural images work.
SCANS = [s for s in os.environ.get("WSP_FIT_SCANS", "").split(",") if s]


def load_images(n_per_scan=4, scans=SCANS, offset=0.0):
    """Square centre crops, evenly spaced frames from each scan (offset shifts the sampling for held-out sets)."""
    out = []
    for s in scans:
        files = sorted((DATA / s / "images").glob("*"))
        idx = ((np.arange(n_per_scan) + 0.5 + offset) / n_per_scan * len(files)).astype(int) % len(files)
        for i in idx:
            im = Image.open(files[i]).convert("RGB")
            m = min(im.size)
            out.append(im.crop(((im.width - m) // 2, (im.height - m) // 2, (im.width + m) // 2, (im.height + m) // 2)))
    return out


def to_batch(images, size):
    x = torch.stack([torch.from_numpy(np.asarray(im.resize((size, size), Image.BICUBIC))).permute(2, 0, 1) for im in images])
    return ((x.float() / 255) - MEAN) / STD


def load_dinov2(name="dinov2_vitl14"):
    return torch.hub.load(str(DINOV2_REPO), name, source="local").eval().cuda()


def load_dinov3():
    return torch.hub.load(str(DINOV3_REPO), "dinov3_vitl16", source="local", weights=str(DINOV3_W)).eval().cuda()


@torch.no_grad()
def dense(model, images, patch, raw=False, bs=8):
    """(N, GRID*GRID, C) patch features. raw=True returns last-block tokens before the final norm (for norms)."""
    feats = []
    for i in range(0, len(images), bs):
        x = to_batch(images[i:i + bs], GRID * patch).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.get_intermediate_layers(x, n=1, norm=not raw)[0]
        feats.append(o.float().cpu())
    return torch.cat(feats)


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def gram_agreement(X, Y):
    """Pearson correlation between the two patch-patch cosine similarity matrices."""
    a = F.normalize(X, dim=1); b = F.normalize(Y, dim=1)
    ga, gb = (a @ a.T).flatten(), (b @ b.T).flatten()
    return float(torch.corrcoef(torch.stack([ga, gb]))[0, 1])


def artifact_frac(raw):
    """Fraction of patch tokens whose norm exceeds 2x the image median (the DINOv2 'high-norm' outliers)."""
    n = raw.norm(dim=-1)
    return float((n > 2 * n.median(dim=1, keepdim=True).values).float().mean())


def kmeans_ari(X, Y, k=6):
    la = KMeans(k, n_init=4, random_state=0).fit_predict(F.normalize(X, dim=1).numpy())
    lb = KMeans(k, n_init=4, random_state=0).fit_predict(F.normalize(Y, dim=1).numpy())
    return adjusted_rand_score(la, lb)


def score(f2, f3, raw2):
    """Mean per-image closeness of DINOv2-variant features f2 to DINOv3 features f3."""
    r = {"cka": [], "gram": [], "ari": []}
    for a, b in zip(f2, f3):
        r["cka"].append(linear_cka(a, b)); r["gram"].append(gram_agreement(a, b)); r["ari"].append(kmeans_ari(a, b))
    out = {k: float(np.mean(v)) for k, v in r.items()}
    out["artifact_frac"] = artifact_frac(raw2)
    return out


def pca_rgb(f, ref_basis=None):
    """First 3 PCs of one image's features -> (GRID, GRID, 3) uint8."""
    x = f - f.mean(0)
    _, _, V = torch.pca_lowrank(x, q=3)
    p = x @ V
    p = (p - p.quantile(0.01, 0)) / (p.quantile(0.99, 0) - p.quantile(0.01, 0))
    return (p.clamp(0, 1).reshape(GRID, GRID, 3).numpy() * 255).astype(np.uint8)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))
