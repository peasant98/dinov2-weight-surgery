"""Weight-space programming on DINOv3-L/16 (no gradients): same toolkit as for DINOv2.
pos_projector: append a block whose attention branch is zero and whose MLP is an exact linear map
h -> -alpha * c * U U^T LN(h) (GELU(z) - GELU(-z) = z), U = top-k positional subspace of the stream (per-position mean
over generic COCO dev images). DINOv3 has no ballast channels, so the LayerNorm scale is per-token; c = median scale."""
import copy
import os
from pathlib import Path

import torch

import side_channel as sc

HERE = Path(__file__).parent
DATA = Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads")))   # benchmarks live here


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


_STAGE = ""                                                     # edits applied before the current one (cache key)


def build_v3(base, spec):
    global _STAGE
    m = copy.deepcopy(base)
    for i, (fn, kw) in enumerate(spec):
        _STAGE = repr(spec[:i])
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
def spair_scatter(m, blk, start, per_cat=60, seed=20, G=40, patch=16, bs=8, lift=None):
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
    root = DATA / "benchmarks/spair/SPair-71k"
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
    if lift is not None:
        allz = torch.cat([lift(allz[i:i + 20000]) for i in range(0, len(allz), 20000)])
    mu = allz.mean(0); cov_t = (allz - mu).T @ (allz - mu) / len(allz)
    d = []
    for p in pairs:
        s, t = idx[(p["category"], p["src_imname"])], idx[(p["category"], p["trg_imname"])]
        def samp(j, kps):
            w, h = sizes[j]; k = torch.tensor(kps, dtype=torch.float32)
            g = torch.stack([k[:, 0] / w * 2 - 1, k[:, 1] / h * 2 - 1], -1).view(1, -1, 1, 2)
            zj = Z[j].float() if lift is None else lift(Z[j].cuda().double()).float().cpu()
            return F.grid_sample(zj.permute(2, 0, 1)[None], g, align_corners=False)[0, :, :, 0].T
        d.append(samp(s, p["src_kps"]) - samp(t, p["trg_kps"]))
    d = torch.cat(d).cuda().double()
    cov_w = d.T @ d / len(d) / 2
    return cov_t, cov_w, len(d)


DAVIS = DATA / "benchmarks/davis/DAVIS"


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


@torch.no_grad()
def dpair_scatter(m, blk, start, patch=16, grid=(34, 61), step=3, offsets=(1, 2), rad=7, purity=0.75, lift=None, **_):
    """SUPERVISED local pair scatter for label propagation, from DAVIS-2017 TRAIN object masks (fit videos only; the
    12 dev videos and val are never touched). Pairs = (token p of a later frame, token q of an earlier frame within
    +-rad tokens), exactly the pairs davis.py's propagation compares. cov_w: pairs with the same label, p restricted to
    tokens that also have a different-label token in their window (near a boundary); cov_b: pairs with different
    labels. Returns (cov_b, cov_w, n): shrinking cov_w directions relative to cov_b sharpens the propagation metric
    without asking a whole object to collapse (unlike dmask_scatter)."""
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    gh, gw = grid; n0 = m.embed_dim; n = n0 if lift is None else lift.dim
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    pos = torch.stack([yy.flatten(), xx.flatten()], -1).float().cuda()
    NB = (pos[:, None] - pos[None]).abs().max(-1).values <= rad
    Sw = torch.zeros(n, n, dtype=torch.float64, device="cuda"); Sb = torch.zeros_like(Sw); nw = nb = 0

    def acc(S, M, a, b):
        Md = M.double()
        S += a.T @ (Md.sum(1)[:, None] * a) + b.T @ (Md.sum(0)[:, None] * b)
        C = a.T @ (Md @ b); S -= C + C.T

    for v in _davis_fit_videos():
        fr = sorted((DAVIS / "JPEGImages/480p" / v).glob("*.jpg"))[::step]
        Z = _zmaps(m, blk, start, [Image.open(f).convert("RGB") for f in fr], grid, patch).cuda().double().reshape(len(fr), -1, n0)
        if lift is not None:
            Z = torch.stack([lift(z) for z in Z])
        labs = torch.from_numpy(np.stack([np.array(Image.open(DAVIS / "Annotations/480p" / v / (f.stem + ".png")))
                                          for f in fr])).long()
        labs[labs == 255] = 0
        frac = F.adaptive_avg_pool2d(F.one_hot(labs, int(labs.max()) + 1).permute(0, 3, 1, 2).float(), grid)
        pv, lab = frac.max(1); ok = (pv >= purity).flatten(1).cuda(); lab = lab.flatten(1).cuda()
        for t in range(len(fr)):
            for o in offsets:
                if t + o >= len(fr):
                    continue
                valid = NB & ok[t + o][:, None] & ok[t][None]
                same = lab[t + o][:, None] == lab[t][None]
                diff = valid & ~same
                near = valid & same & (diff.sum(1) > 0)[:, None]
                acc(Sw, near, Z[t + o], Z[t]); acc(Sb, diff, Z[t + o], Z[t])
                nw += int(near.sum()); nb += int(diff.sum())
    return Sb / nb / 2, Sw / nw / 2, nw + nb


def mix_dpair(m, blk, start, patch=16, w=1.0, **kw):
    """pooled discriminant of the SPair-keypoint problem and the DAVIS local-pair problem, each normalised by its
    reference scatter's trace."""
    a_t, a_w, na = spair_scatter(m, blk, start, patch=patch, **kw)
    b_t, b_w, nb = dpair_scatter(m, blk, start, patch=patch, lift=kw.get("lift"))
    ta, tb = a_t.trace(), b_t.trace()
    return a_t / ta + w * b_t / tb, a_w / ta + w * b_w / tb, na + nb


def scatter(src, m, blk, start, patch=16, **kw):
    if src.startswith("mix3:"):                                   # mix3:<video weight>:<segmentation weight>
        _, wd, ws = src.split(":")
        return mix3(m, blk, start, patch=patch, wd=float(wd), ws=float(ws), **kw)
    if src.startswith("spair+dpair"):
        return mix_dpair(m, blk, start, patch=patch, w=float(src[len("spair+dpair"):] or 1.0), **kw)
    if src == "dpair":
        return dpair_scatter(m, blk, start, patch=patch)
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


@torch.no_grad()
def gains(m, edits):
    """Gain edits from scan_v3.py: [(block, part, gain), ...]. part "attn" / "mlp" scales the branch's LayerScale,
    "temp" scales the attention logits (query rows of qkv)."""
    if isinstance(edits, str):
        import json
        edits = json.load(open(HERE / edits))["kept"]
    for b, part, g in edits:
        blk = m.blocks[b]
        if part == "temp":
            d = m.embed_dim
            blk.attn.qkv.weight[:d] *= g
            if blk.attn.qkv.bias is not None:
                blk.attn.qkv.bias[:d] *= g
        else:
            (blk.ls1 if part == "attn" else blk.ls2).gamma.mul_(g)
    return m


if (HERE / "downstream/scan_v3.json").exists():
    VARIANTS_V3["v3:gains"] = [("gains", {"edits": "downstream/scan_v3.json"})]
    VARIANTS_V3["v3:gains+fisher b0.5"] = [("gains", {"edits": "downstream/scan_v3.json"}), ("fisher", {"beta": 0.5})]


class Emulated(torch.nn.Module):
    """DIAGNOSTIC ONLY (not a weight edit): final patch features with extra information concatenated, to measure
    whether it is worth programming into the weights. Each part is L2-normalised; the extra part is projected to
    `dim` dimensions with a fixed random orthogonal map and scaled by s.
    layer: an earlier block whose (final-normed) patch features are appended; cls: append the class token to every
    patch; mean: append the image's mean patch feature to every patch."""

    def __init__(self, m, layer=None, cls=False, mean=False, s=1.0, dim=256, blur=0, sharpen=0.0, nlm=None):
        """blur: append the (blur x blur) box average of the final features; sharpen: in place, final features
        f + sharpen * (f - 3x3 box average) with nothing appended (negative = smoothing); nlm=(tau, r): in place,
        f + s * (similarity-weighted average of f, softmax(tau * cos), over tokens within r; r=0 means all)."""
        super().__init__()
        self.nlm = nlm
        self.m, self.layer, self.cls, self.mean, self.s = m, layer, cls, mean, s
        self.blur, self.sharpen = blur, sharpen
        g = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.linalg.qr(torch.randn(m.embed_dim, m.embed_dim, generator=g))[0][:, :dim])

    def get_intermediate_layers(self, x, n=1, norm=True):
        import torch.nn.functional as F
        last = len(self.m.blocks) - 1
        outs = self.m.get_intermediate_layers(x, n=[last] if self.layer is None else [self.layer, last], norm=True,
                                              return_class_token=True)
        early, (final, cls) = outs[0][0], outs[-1]
        final = final.float()
        if self.nlm:
            tau, r = self.nlm
            gh, gw = x.shape[-2] // self.m.patch_size, x.shape[-1] // self.m.patch_size
            f = F.normalize(final, dim=-1)
            logit = tau * f @ f.transpose(1, 2)
            if r:
                yx = torch.stack(torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij"), -1).flatten(0, 1)
                far = (yx[:, None] - yx[None]).abs().amax(-1) > r
                logit = logit.masked_fill(far.to(logit.device), float("-inf"))
            return (f + self.s * logit.softmax(-1) @ f,)
        if self.blur or self.sharpen:
            gh, gw = x.shape[-2] // self.m.patch_size, x.shape[-1] // self.m.patch_size
            k = self.blur or 3
            grid = F.normalize(final, dim=-1).reshape(len(x), gh, gw, -1).permute(0, 3, 1, 2)
            avg = F.avg_pool2d(F.pad(grid, (k // 2,) * 4, mode="replicate"), k, 1).permute(0, 2, 3, 1).flatten(1, 2)
            if self.sharpen:
                return (F.normalize(final, dim=-1) * (1 + self.sharpen) - self.sharpen * avg,)
            return (torch.cat((F.normalize(final, dim=-1), self.s * F.normalize(avg @ self.proj.to(final.device), dim=-1)), -1),)
        extra = early.float() if self.layer is not None else (
            cls.float()[:, None].expand_as(final) if self.cls else final.mean(1, keepdim=True).expand_as(final))
        extra = F.normalize(extra, dim=-1) @ self.proj.to(final.device)
        return (torch.cat((F.normalize(final, dim=-1), self.s * F.normalize(extra, dim=-1)), -1),)


def emulate(m, **kw):
    return Emulated(m, **kw)


for _l in (11, 15, 17, 19, 21):
    VARIANTS_V3[f"v3:emu L{_l}"] = [("emulate", {"layer": _l})]
VARIANTS_V3["v3:emu L17 s0.5"] = [("emulate", {"layer": 17, "s": 0.5})]
for _s in (0.5, 1.0):
    VARIANTS_V3[f"v3:emu cls s{_s}"] = [("emulate", {"cls": True, "s": _s})]
VARIANTS_V3["v3:emu mean s0.5"] = [("emulate", {"mean": True, "s": 0.5})]
for _g in (-0.5, 0.5, 1.0, 2.0):
    VARIANTS_V3[f"v3:emu sharpen {_g}"] = [("emulate", {"sharpen": _g})]
for _k in (3, 7):
    VARIANTS_V3[f"v3:emu blur{_k}"] = [("emulate", {"blur": _k})]
for _b in (0.5, 1.0):
    VARIANTS_V3[f"v3:dfisher b{_b}"] = [("fisher", {"beta": _b, "src": "dpair"})]
for _w in (0.3, 1.0, 3.0):
    VARIANTS_V3[f"v3:fisher+dpair w{_w} b0.5"] = [("fisher", {"beta": 0.5, "src": f"spair+dpair{_w}"})]
PICK10 = ("fisher", {"beta": 0.5, "src": "spair+dpair0.3"})
for _t, _r, _s in ((20, 3, 1.0), (40, 3, 1.0), (40, 3, 3.0), (40, 0, 1.0), (80, 3, 3.0), (20, 1, 1.0)):
    VARIANTS_V3[f"v3:pick10+nlm t{_t} r{_r} s{_s}"] = [PICK10, ("emulate", {"nlm": (_t, _r), "s": _s})]


class Lift:
    """Nonlinear feature map z -> [z, s * (GELU(z W^T + b) - mean)] with fixed random hidden units (nothing is
    trained). Directions are random unit vectors in the whitened top-k principal subspace of z, so every unit sees a
    pre-activation of standard deviation `gain`."""

    def __init__(self, zs, hidden=2048, k=256, gain=1.5, s=1.0, seed=0):
        """zs: (N, C) double sample of z = LN(h) on the GPU."""
        g = torch.Generator().manual_seed(seed)
        mu = zs.mean(0)
        lam, V = torch.linalg.eigh((zs - mu).T @ (zs - mu) / len(zs))
        lam, V = lam[-k:], V[:, -k:]
        r = torch.nn.functional.normalize(torch.randn(hidden, k, generator=g, dtype=torch.float64), dim=-1).to(zs.device)
        self.W = gain * (r / lam.sqrt()) @ V.T                                   # (hidden, C)
        self.b = 0.5 * torch.randn(hidden, generator=g, dtype=torch.float64).to(zs.device) - self.W @ mu
        self.s, self.mean = s, 0.0
        self.key = (hidden, k, gain, s, seed)
        self.mean = self.phi(zs).mean(0)
        self.dim = zs.shape[1] + hidden

    def phi(self, z):
        return torch.nn.functional.gelu(z @ self.W.T + self.b) - self.mean

    def __call__(self, z):
        return torch.cat((z, self.s * self.phi(z)), -1)


@torch.no_grad()
def _z_sample(m, blk, x, start, bs=8):
    """(N, C) double z = LN(h) of the patch tokens entering blk's MLP, and the median LN scale."""
    cur, out, sig = {}, [], []
    h = blk.norm2.register_forward_hook(lambda mod, inp, o: cur.__setitem__("h", inp[0].float()))
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x[i:i + bs].cuda().float(), n=1)
        hh = cur["h"][:, start:].reshape(-1, cur["h"].shape[-1])
        sig.append(hh.std(-1, unbiased=False))
        out.append(torch.nn.functional.layer_norm(hh, (hh.shape[-1],)).double())
    h.remove()
    return torch.cat(out), float(torch.cat(sig).median())


@torch.no_grad()
def nfisher(m, s=1.0, beta=0.5, eps=0.05, w=0.3, hidden=2048, gain=1.5, per_cat=60, info=None, src=None):
    """Programmed NONLINEAR discriminant (appended block, closed form, no gradients). The pooled keypoint + video
    Fisher metric is computed on lifted features [z, s*phi(z)] (Lift), the result is compressed back to the stream
    width with its top principal directions and rotated (Procrustes) to stay aligned with the original coordinates:
    h -> c * (M_z z + M_phi phi(z)). Hidden units [0, 2C) carry the linear part exactly, the next `hidden` carry phi."""
    import linmap
    blk = _append_zero_block(m)
    start, C = 1 + m.n_storage_tokens, m.embed_dim
    x = sc.pos_images(512, size=512, path=HERE / "proxy/pos_imgs_512_512.pt")
    zs, c = _z_sample(m, blk, x[:64], start)
    lift = Lift(zs, hidden=hidden, gain=gain, s=1.0)              # statistics at s=1 (cached), rescaled exactly below
    wd, ws = (w, 0.0) if src is None else map(float, src.split(":")[1:])
    d = torch.ones(lift.dim, dtype=torch.float64, device="cuda"); d[C:] = s
    cov_t, cov_w, n = mix3(m, blk, start, wd=wd, ws=ws, d=d, per_cat=per_cat, lift=lift)
    lift.s = s
    B, lam = linmap.fisher_B(cov_t, cov_w, beta, eps)
    A = B + torch.eye(len(B), dtype=B.dtype, device=B.device)
    u = lift(zs) @ A.T
    ev, U = torch.linalg.eigh(u.T @ u / len(u))
    U = U[:, -C:]
    P, _, Qh = torch.linalg.svd((u @ U).T @ zs)
    M = (A.T @ U @ P @ Qh).T                                                    # (C, C + hidden)
    keep = torch.ones(C, dtype=torch.bool, device=blk.mlp.fc1.weight.device)
    linmap.program(blk, (M[:, :C] - torch.eye(C, dtype=M.dtype, device=M.device)).float(), c, keep)
    W1, W2 = blk.mlp.fc1.weight, blk.mlp.fc2.weight
    Mp = c * s * M[:, C:]
    W1[2 * C:2 * C + hidden] = lift.W.to(W1.dtype)
    blk.mlp.fc1.bias[2 * C:2 * C + hidden] = lift.b.to(W1.dtype)
    W2[:, 2 * C:2 * C + hidden] = Mp.to(W2.dtype)
    blk.mlp.fc2.bias += (-(Mp @ lift.mean)).to(W2.dtype)   # phi is centred; note lift.mean was set before self.mean
    if info is not None:
        info.update(c=c, n=n, kept_var=float(ev[-C:].sum() / ev.sum()), lam_min=float(lam.min()), lam_max=float(lam.max()),
                    target=(zs @ M[:, :C].T + s * lift.phi(zs) @ M[:, C:].T)[:2048].float() * c, zs=zs[:2048])
    return m


for _s in (0.5, 1.0, 2.0):
    VARIANTS_V3[f"v3:nfisher s{_s}"] = [("nfisher", {"s": _s})]


@torch.no_grad()
def seg_scatter(m, blk, start, patch=16, n_img=1000, G=32, purity=0.75, lift=None, seed=30, **_):
    """Class discriminant scatter for semantic segmentation from ADE20K TRAINING images that are in no evaluation draw
    (common.ade_files("unused"); the fresh3 test images are excluded). Tokens whose 16x16 patch is >= purity one class.
    Returns (total scatter, within-class scatter, number of tokens)."""
    import sys
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    if str(HERE / "downstream") not in sys.path:
        sys.path.insert(0, str(HERE / "downstream"))
    import common as C
    files = list(np.random.default_rng(seed).choice(C.ade_files("unused"), n_img, replace=False))

    def crop(im):
        s = min(im.size); x, y = (im.width - s) // 2, (im.height - s) // 2
        return im.crop((x, y, x + s, y + s))

    n = m.embed_dim if lift is None else lift.dim
    s1 = torch.zeros(151, n, dtype=torch.float64, device="cuda"); cnt = torch.zeros(151, dtype=torch.float64, device="cuda")
    s2 = torch.zeros(n, n, dtype=torch.float64, device="cuda")
    for i in range(0, n_img, 50):
        chunk = files[i:i + 50]
        ims = [crop(Image.open(f).convert("RGB")) for f in chunk]
        Z = _zmaps(m, blk, start, ims, (G, G), patch).cuda().double().reshape(-1, m.embed_dim)
        if lift is not None:
            Z = lift(Z)
        labs = []
        for f in chunk:
            a = f.parent.parent.parent / "annotations" / f.parent.name / (f.stem + ".png")
            labs.append(torch.from_numpy(np.array(crop(Image.open(a)).resize((G * 4, G * 4), Image.NEAREST))).long())
        frac = F.avg_pool2d(F.one_hot(torch.stack(labs), 151).permute(0, 3, 1, 2).float(), 4)
        pv, lab = frac.max(1); pv, lab = pv.flatten().cuda(), lab.flatten().cuda()
        ok = (pv >= purity) & (lab > 0)                                           # 0 = 'other', ignored
        Z, lab = Z[ok], lab[ok]
        s1.index_add_(0, lab, Z); cnt += torch.bincount(lab, minlength=151).double(); s2 += Z.T @ Z
    N = cnt.sum(); mu = s1.sum(0) / N
    cov_t = s2 / N - mu[:, None] * mu[None]
    nz = cnt > 0
    between = (s1[nz] / cnt[nz, None]).T @ (s1[nz] / cnt[nz, None] * cnt[nz, None]) / N - mu[:, None] * mu[None]
    return cov_t, cov_t - between, int(N)


def mix3(m, blk, start, patch=16, wd=0.3, ws=1.0, d=None, **kw):
    """Pooled discriminant: SPair keypoints + wd * DAVIS local pairs + ws * ADE semantic classes, each normalised by the
    trace of its total scatter."""
    out_t, out_w, n = 0, 0, 0
    parts = ((1.0, spair_scatter, kw), (wd, dpair_scatter, {"lift": kw.get("lift")}), (ws, seg_scatter, {"lift": kw.get("lift")}))
    for w, fn, extra in parts:
        if not w:
            continue
        lift = extra.get("lift")
        key = f"{_STAGE}|{fn.__name__}|{None if lift is None else lift.key}"
        path = HERE / "proxy/scatter_cache" / (__import__("hashlib").md5(key.encode()).hexdigest() + ".pt")
        if path.exists():
            a_t, a_w, na = torch.load(path)
            a_t, a_w = a_t.cuda(), a_w.cuda()
        else:
            a_t, a_w, na = fn(m, blk, start, patch=patch, **extra)
            path.parent.mkdir(exist_ok=True)
            torch.save((a_t.cpu(), a_w.cpu(), na), path)
        if d is not None:                                         # rescaled coordinates: cov -> D cov D
            a_t, a_w = d[:, None] * a_t * d[None], d[:, None] * a_w * d[None]
        tr = a_t.trace()
        out_t, out_w, n = out_t + w * a_t / tr, out_w + w * a_w / tr, n + na
    return out_t, out_w, n


@torch.no_grad()
def rope_scale(m, k):
    """Scales the RoPE coordinates by k (periods / k). DINOv3 ViT-L was trained with random coordinate rescaling
    in [1/2, 2], so 0.5 <= k <= 2 stays inside the training distribution."""
    m.rope_embed.periods.div_(k)
    return m


for _k in (0.5, 0.7, 1.4, 2.0):
    VARIANTS_V3[f"v3:rope{_k}"] = [("rope_scale", {"k": _k})]
for _ws in (0.3, 1.0, 3.0):
    VARIANTS_V3[f"v3:fisher mix3 d0.3 s{_ws}"] = [("fisher", {"beta": 0.5, "src": f"mix3:0.3:{_ws}"})]
for _k in (1.4, 2.0, 2.5, 3.0):
    VARIANTS_V3[f"v3:rope{_k}+pick10"] = [("rope_scale", {"k": _k}), PICK10]
for _k in (2.5, 3.0):
    VARIANTS_V3[f"v3:rope{_k}"] = [("rope_scale", {"k": _k})]
MIX3S03 = ("fisher", {"beta": 0.5, "src": "mix3:0.3:0.3"})
for _k in (1.4, 2.0, 2.5):
    VARIANTS_V3[f"v3:rope{_k}+mix3s0.3"] = [("rope_scale", {"k": _k}), MIX3S03]
for _wd, _ws in ((1.0, 0.3), (1.0, 1.0), (3.0, 1.0), (0.3, 0.5), (3.0, 3.0)):
    VARIANTS_V3[f"v3:fisher mix3 d{_wd} s{_ws}"] = [("fisher", {"beta": 0.5, "src": f"mix3:{_wd}:{_ws}"})]
