"""Weight-space programming on DINOv3-L/16 (no gradients): same toolkit as for DINOv2.
pos_projector: append a block whose attention branch is zero and whose MLP is an exact linear map
h -> -alpha * c * U U^T LN(h) (GELU(z) - GELU(-z) = z), U = top-k positional subspace of the stream (per-position mean
over generic COCO dev images). DINOv3 has no ballast channels, so the LayerNorm scale is per-token; c = median scale."""
import os
import copy
from pathlib import Path

import torch

import side_channel as sc

HERE = Path(__file__).parent


@torch.no_grad()
def pos_projector(m, k=16, alpha=1.0, n_img=512, size=512, bs=8, info=None):
    blk = copy.deepcopy(m.blocks[-1])
    blk.attn.proj.weight.zero_()
    if blk.attn.proj.bias is not None:
        blk.attn.proj.bias.zero_()
    for ln in (blk.norm1, blk.norm2):
        ln.weight.fill_(1.0); ln.bias.zero_()
    for lin in (blk.mlp.fc1, blk.mlp.fc2):
        lin.weight.zero_(); lin.bias.zero_()
    for ls in (blk.ls1, blk.ls2):
        if hasattr(ls, "gamma"):
            ls.gamma.fill_(1.0)
    m.blocks.append(blk)
    x = sc.pos_images(n_img, size=size, path=HERE / f"proxy/pos_imgs_{n_img}_{size}.pt")
    cur, acc, sig, cnt = {}, 0, [], 0
    h = blk.norm2.register_forward_hook(lambda mod, inp, out: cur.__setitem__("h", inp[0].float()))
    s0 = 1 + m.n_storage_tokens
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x[i:i + bs].cuda().float(), n=1)
        hh = cur["h"][:, s0:]
        acc = acc + hh.sum(0); cnt += len(hh)
        sig.append(hh.std(-1, unbiased=False).flatten().cpu())
    h.remove()
    D = acc / cnt
    D = D - D.mean(0, keepdim=True)
    D = D - D.mean(1, keepdim=True)                               # orthogonal to the all-ones vector
    _, Sv, V = torch.linalg.svd(D, full_matrices=False)
    U = V[:k].T
    sig = torch.cat(sig); c = float(sig.median())
    dt = blk.mlp.fc1.weight.dtype
    blk.mlp.fc1.weight[:k] = U.T.to(dt); blk.mlp.fc1.weight[k:2 * k] = -U.T.to(dt)
    blk.mlp.fc2.weight[:, :k] = (-alpha * c * U).to(dt); blk.mlp.fc2.weight[:, k:2 * k] = (alpha * c * U).to(dt)
    if info is not None:
        ev = Sv ** 2 / (Sv ** 2).sum()
        info.update(k=k, energy_frac=float(ev[:k].sum()), sigma=c, sigma_spread=float(sig.std() / sig.mean()))
    return m


def build_v3(base, spec):
    m = copy.deepcopy(base)
    for fn, kw in spec:
        m = globals()[fn](m, **kw)
    return m.eval()


VARIANTS_V3 = {f"v3:pos{k} a{a}": [("pos_projector", {"k": k, "alpha": a})] for k in (4, 16, 64) for a in (0.25, 0.5, 1.0)}


@torch.no_grad()
def _append_zero_block(m):
    blk = copy.deepcopy(m.blocks[-1])
    blk.attn.proj.weight.zero_()
    if blk.attn.proj.bias is not None:
        blk.attn.proj.bias.zero_()
    for ln in (blk.norm1, blk.norm2):
        ln.weight.fill_(1.0); ln.bias.zero_()
    for lin in (blk.mlp.fc1, blk.mlp.fc2):
        lin.weight.zero_(); lin.bias.zero_()
    for ls in (blk.ls1, blk.ls2):
        if hasattr(ls, "gamma"):
            ls.gamma.fill_(1.0)
    m.blocks.append(blk)
    return blk


@torch.no_grad()
def whiten(m, beta=0.5, eps=0.1, center=0.0, n_img=512, size=512):
    """Programmed partial whitening of the output stream (appended block, exact linear MLP)."""
    import linmap
    blk = _append_zero_block(m)
    x = sc.pos_images(n_img, size=size, path=HERE / f"proxy/pos_imgs_{n_img}_{size}.pt")
    keep = torch.ones(m.embed_dim, dtype=torch.bool, device=blk.mlp.fc1.weight.device)
    mu, cov, c = linmap.stream_stats(m, blk, x, 1 + m.n_storage_tokens, keep)
    B = linmap.whitening_B(cov, beta, eps)
    bias = -c * center * ((B + torch.eye(len(B), dtype=B.dtype, device=B.device)) @ mu) if center else None
    linmap.program(blk, B.float(), c, keep, bias=None if bias is None else bias.float())
    return m


for _b in (0.25, 0.5):
    for _e in (0.1, 1.0):
        VARIANTS_V3[f"v3:white b{_b} e{_e}"] = [("whiten", {"beta": _b, "eps": _e})]
VARIANTS_V3["v3:white b0.5 e0.1 c0.5"] = [("whiten", {"beta": 0.5, "eps": 0.1, "center": 0.5})]
VARIANTS_V3["v3:pos16 a0.25+white b0.25 e1.0"] = [("pos_projector", {"k": 16, "alpha": 0.25}), ("whiten", {"beta": 0.25, "eps": 1.0})]


@torch.no_grad()
def spair_scatter(m, blk, start, per_cat=60, seed=20, G=40, patch=16, bs=8):
    """Within-match scatter (keypoint q in source vs its match t in target) and total scatter of z = LN(h) at blk's
    MLP input, from SPair-71k TRAIN pairs (a practice set never used for evaluation)."""
    import json
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F
    sys_path = str(HERE / "downstream")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import common as C
    root = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "benchmarks/spair/SPair-71k"
    bycat = {}
    for f in sorted((root / "PairAnnotation/trn").glob("*.json")):
        bycat.setdefault(f.stem.split(":")[1], []).append(f)
    rng = np.random.default_rng(seed)
    pairs = [json.load(open(f)) for c in sorted(bycat) for f in rng.choice(bycat[c], per_cat, replace=False)]
    names = sorted({(p["category"], p[k]) for p in pairs for k in ("src_imname", "trg_imname")})
    idx = {n: i for i, n in enumerate(names)}
    cur, zmaps = {}, []
    hk = blk.norm2.register_forward_hook(lambda mod, inp, out: cur.__setitem__("h", inp[0].float()))
    sizes = []
    for i in range(0, len(names), bs):
        ims = [Image.open(root / "JPEGImages" / c / n).convert("RGB") for c, n in names[i:i + bs]]
        sizes += [im.size for im in ims]
        x = C.to_tensor(ims, (G * patch, G * patch)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x, n=1)
        h = cur["h"][:, start:]
        zmaps.append(F.layer_norm(h, (h.shape[-1],)).reshape(len(ims), G, G, -1).half().cpu())
    hk.remove()
    Z = torch.cat(zmaps)
    allz = Z.reshape(-1, Z.shape[-1])[torch.randperm(Z.shape[0] * G * G)[:200000]].cuda().double()
    mu = allz.mean(0); cov_t = (allz - mu).T @ (allz - mu) / len(allz)
    d = []
    for p in pairs:
        s, t = idx[(p["category"], p["src_imname"])], idx[(p["category"], p["trg_imname"])]
        def samp(j, kps):
            w, h = sizes[j]; k = torch.tensor(kps, dtype=torch.float32)
            g = torch.stack([k[:, 0] / w * 2 - 1, k[:, 1] / h * 2 - 1], -1).view(1, -1, 1, 2)
            return F.grid_sample(Z[j].float().permute(2, 0, 1)[None], g, align_corners=False)[0, :, :, 0].T
        d.append(samp(s, p["src_kps"]) - samp(t, p["trg_kps"]))
    d = torch.cat(d).cuda().double()
    cov_w = d.T @ d / len(d) / 2
    return cov_t, cov_w, len(d)


DAVIS = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "benchmarks/davis/DAVIS"


def _davis_fit_videos():
    """DAVIS-2017 TRAIN videos not in the 12-video dev subset (davis.py's rule); val is never touched."""
    vids = (DAVIS / "ImageSets/2017/train.txt").read_text().split()
    dev = set(vids[::len(vids) // 12][:12])
    return [v for v in vids if v not in dev]


@torch.no_grad()
def _zmaps(m, blk, start, ims, grid, patch, bs=8):
    """z = LN(h) at blk's MLP input for each PIL image on a (gh, gw) token grid -> (N, gh, gw, C) half, CPU."""
    import sys
    import torch.nn.functional as F
    if str(HERE / "downstream") not in sys.path:
        sys.path.insert(0, str(HERE / "downstream"))
    import common as C
    cur, out = {}, []
    hk = blk.norm2.register_forward_hook(lambda mod, inp, o: cur.__setitem__("h", inp[0].float()))
    for i in range(0, len(ims), bs):
        x = C.to_tensor(ims[i:i + bs], (grid[0] * patch, grid[1] * patch)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x, n=1)
        h = cur["h"][:, start:]
        out.append(F.layer_norm(h, (h.shape[-1],)).reshape(len(x), *grid, -1).half().cpu())
    hk.remove()
    return torch.cat(out)


def _cov_t(Z, n=200000):
    allz = Z.reshape(-1, Z.shape[-1])[torch.randperm(Z[..., 0].numel())[:n]].cuda().double()
    mu = allz.mean(0)
    return (allz - mu).T @ (allz - mu) / len(allz)


@torch.no_grad()
def video_scatter(m, blk, start, patch=16, grid=(27, 48), step=10, gap=2, rad=3, **_):
    """LABEL-FREE within-match scatter from video: tokens of frames t and t+gap (DAVIS train fit videos) that are mutual
    nearest neighbours (cosine of z) inside a +-rad token window are taken as the same physical point."""
    import torch.nn.functional as F
    from PIL import Image
    pairs = []
    for v in _davis_fit_videos():
        fr = sorted((DAVIS / "JPEGImages/480p" / v).glob("*.jpg"))
        pairs += [(fr[t], fr[t + gap]) for t in range(0, len(fr) - gap, step)]
    ims = [Image.open(f).convert("RGB") for p in pairs for f in p]
    Z = _zmaps(m, blk, start, ims, grid, patch)
    gh, gw = grid
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    pos = torch.stack([yy.flatten(), xx.flatten()], -1).float().cuda()
    far = (pos[:, None] - pos[None]).abs().max(-1).values > rad
    d = []
    for i in range(0, len(Z), 2):
        a, b = Z[i].reshape(-1, Z.shape[-1]).cuda().float(), Z[i + 1].reshape(-1, Z.shape[-1]).cuda().float()
        s = (F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T).masked_fill(far, -2)
        ab, ba = s.argmax(1), s.argmax(0)
        mutual = ba[ab] == torch.arange(len(a), device=a.device)
        d.append((a[mutual] - b[ab[mutual]]).double())
    d = torch.cat(d)
    return _cov_t(Z), d.T @ d / len(d) / 2, len(d)


@torch.no_grad()
def aug_scatter(m, blk, start, patch=16, grid=(27, 48), step=5, seed=0, **_):
    """LABEL-FREE within-match scatter from augmentations: view A = whole frame, view B = random crop (50-90 % per side)
    + flip + colour jitter of the same frame; token centres of B map to known points of A (sampled bilinearly)."""
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image, ImageEnhance
    rng = np.random.default_rng(seed)
    files = [f for v in _davis_fit_videos() for f in sorted((DAVIS / "JPEGImages/480p" / v).glob("*.jpg"))[::step]]
    A, B, maps = [], [], []
    for f in files:
        im = Image.open(f).convert("RGB"); W, H = im.size
        sw, sh = rng.uniform(0.5, 0.9, 2); x0, y0 = rng.uniform(0, 1 - sw), rng.uniform(0, 1 - sh)
        b = im.crop((int(x0 * W), int(y0 * H), int((x0 + sw) * W), int((y0 + sh) * H)))
        flip = rng.random() < 0.5
        if flip:
            b = b.transpose(Image.FLIP_LEFT_RIGHT)
        for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            b = enh(b).enhance(rng.uniform(0.7, 1.3))
        A.append(im); B.append(b); maps.append((x0, y0, sw, sh, flip))
    ZA, ZB = _zmaps(m, blk, start, A, grid, patch), _zmaps(m, blk, start, B, grid, patch)
    gh, gw = grid
    u = (torch.arange(gw) + 0.5) / gw; w_ = (torch.arange(gh) + 0.5) / gh                 # B token centres in [0,1]
    d = []
    for i, (x0, y0, sw, sh, flip) in enumerate(maps):
        ub = (1 - u) if flip else u
        gx = (x0 + ub * sw) * 2 - 1; gy = (y0 + w_ * sh) * 2 - 1                         # same points in A, [-1,1]
        g = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None].float()  # (1, gh, gw, 2) as (x, y)
        za = F.grid_sample(ZA[i].float().permute(2, 0, 1)[None], g, align_corners=False)[0].permute(1, 2, 0)
        d.append((ZB[i].float() - za).reshape(-1, ZA.shape[-1]).double())
    d = torch.cat(d).cuda()
    return _cov_t(ZA), d.T @ d / len(d) / 2, len(d)


@torch.no_grad()
def dmask_scatter(m, blk, start, patch=16, grid=(34, 61), step=5, purity=0.75, **_):
    """SUPERVISED instance scatter from DAVIS-2017 TRAIN object masks (fit videos only): cov_t = within-video total
    scatter, cov_w = within-instance scatter (object ids + background, per video) of z = LN(h). Tokens whose pooled
    label is < purity pure are skipped. LDA on this separates the objects of a video from each other and the background."""
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    n = m.embed_dim; St = torch.zeros(n, n, dtype=torch.float64, device="cuda"); Sw = torch.zeros_like(St); nt = 0
    for v in _davis_fit_videos():
        fr = sorted((DAVIS / "JPEGImages/480p" / v).glob("*.jpg"))[::step]
        an = [DAVIS / "Annotations/480p" / v / (f.stem + ".png") for f in fr]
        Z = _zmaps(m, blk, start, [Image.open(f).convert("RGB") for f in fr], grid, patch).cuda().double().reshape(-1, n)
        labs = torch.from_numpy(np.stack([np.array(Image.open(a)) for a in an])).long()
        labs[labs == 255] = 0                                                                    # void (if any) -> bg
        K = int(labs.max()) + 1
        frac = F.adaptive_avg_pool2d(F.one_hot(labs, K).permute(0, 3, 1, 2).float(), grid)       # (T, K, gh, gw)
        pv, lab = frac.max(1); keep = (pv >= purity).flatten().cuda(); lab = lab.flatten().cuda()
        z, lab = Z[keep], lab[keep]
        zc = z - z.mean(0); St += zc.T @ zc
        for k in lab.unique():
            zk = z[lab == k]; zk = zk - zk.mean(0); Sw += zk.T @ zk
        nt += len(z)
    return St / nt, Sw / nt, nt


def mix_scatter(m, blk, start, patch=16, w=1.0, **kw):
    """pooled LDA of the SPair-keypoint and DAVIS-mask problems, each normalised by tr(cov_t)."""
    a_t, a_w, na = spair_scatter(m, blk, start, patch=patch, **kw)
    b_t, b_w, nb = dmask_scatter(m, blk, start, patch=patch)
    ta, tb = a_t.trace(), b_t.trace()
    return a_t / ta + w * b_t / tb, a_w / ta + w * b_w / tb, na + nb


def scatter(src, m, blk, start, patch=16, **kw):
    if src.startswith("spair+dmask"):
        return mix_scatter(m, blk, start, patch=patch, w=float(src[len("spair+dmask"):] or 1.0), **kw)
    return {"spair": spair_scatter, "video": video_scatter, "aug": aug_scatter,
            "dmask": dmask_scatter}[src](m, blk, start, patch=patch, **kw)


@torch.no_grad()
def fisher(m, beta=0.5, eps=0.05, per_cat=60, info=None, src="spair"):
    """Programmed Fisher discriminant for correspondence (appended block, exact linear MLP)."""
    import linmap
    blk = _append_zero_block(m)
    cov_t, cov_w, n = scatter(src, m, blk, 1 + m.n_storage_tokens, per_cat=per_cat)
    B, lam = linmap.fisher_B(cov_t, cov_w, beta, eps)
    x = sc.pos_images(512, size=512, path=HERE / "proxy/pos_imgs_512_512.pt")
    keep = torch.ones(m.embed_dim, dtype=torch.bool, device=blk.mlp.fc1.weight.device)
    _, _, c = linmap.stream_stats(m, blk, x[:64], 1 + m.n_storage_tokens, keep)
    linmap.program(blk, B.float(), c, keep)
    if info is not None:
        info.update(n_kp=n, lam_min=float(lam.min()), lam_med=float(lam.median()), lam_max=float(lam.max()))
    return m


for _b in (0.25, 0.5, 1.0):
    VARIANTS_V3[f"v3:fisher b{_b}"] = [("fisher", {"beta": _b})]
for _s, _t in (("video", "v"), ("aug", "a")):                 # label-free Fisher (addendum 5)
    VARIANTS_V3[f"v3:{_t}fisher b0.5"] = [("fisher", {"beta": 0.5, "src": _s})]


@torch.no_grad()
def scale_heads(m, edits):
    """Keep-if-wins head edits from head_loop.py: [(block, head, scale), ...] on attn.proj input columns."""
    if isinstance(edits, str):
        edits = _kept(edits)
    H = m.blocks[0].attn.num_heads; D = m.embed_dim // H
    for b, h, s in edits:
        m.blocks[b].attn.proj.weight[:, h * D:(h + 1) * D] *= s
    return m


def _kept(path):
    import json
    return [tuple(e[:3]) for e in json.load(open(HERE / path))["kept"]]


VARIANTS_V3["v3:heads"] = [("scale_heads", {"edits": "downstream/headloop_v3.json"})]
VARIANTS_V3["v3:heads+pos16 a0.25"] = [("scale_heads", {"edits": "downstream/headloop_v3.json"}),
                                       ("pos_projector", {"k": 16, "alpha": 0.25})]
VARIANTS_V3["v3:heads+fisher b0.5"] = [("scale_heads", {"edits": "downstream/headloop_v3.json"}), ("fisher", {"beta": 0.5})]
