"""Weight-space program: a protected positional side channel in DINOv2 + a hand-written local bilateral block.

1. fold_affine: fold every LayerNorm affine into the following linear layer and every LayerScale into the preceding
   output projection (exact; afterwards a channel's reads are the columns of qkv/fc1, its writes the rows of proj/fc2).
2. reserve: pick channels the network barely uses and cut every read from / write to them (incl. patch embed, cls,
   pos table, final norm output). They now carry only what we put there.
3. install_code: write an exact Fourier position code (cos/sin of k*w*u, u = row/col in 32-grid units) into the
   reserved code channels through the position table, a zero reference channel, and a constant "ballast" written by
   the last block's fc2 bias, so the appended block's LayerNorm has the same scale for every token.
4. append_bilateral: attention logit = A * sum_k cos(k w (u_i - u_j)) [local kernel, via the code] +
   beta * g_i.g_j / nu [feature similarity, PCA guide of the stream]; value = stream, output blended with weight gamma;
   its MLP removes the ballast again.
"""
import os
import copy

import numpy as np
import torch
import torch.nn.functional as F

C_CODE_FREQS = (1, 2, 3, 4)
OMEGA = np.pi / 32                     # base frequency in 32-grid units: period 64 > image span, so no wrap-around
G0 = 32


@torch.no_grad()
def fold_affine(m):
    for blk in m.blocks:
        for ln, lin in ((blk.norm1, blk.attn.qkv), (blk.norm2, blk.mlp.fc1)):
            w, b = ln.weight.clone(), ln.bias.clone()
            if lin.bias is None:
                lin.bias = torch.nn.Parameter(torch.zeros(lin.out_features, device=w.device, dtype=w.dtype))
            lin.bias += lin.weight @ b
            lin.weight *= w[None]
            ln.weight.fill_(1.0); ln.bias.zero_()
        for ls, lin in ((blk.ls1, blk.attn.proj), (blk.ls2, blk.mlp.fc2)):
            g = ls.gamma.clone()
            lin.weight *= g[:, None]; lin.bias *= g
            ls.gamma.fill_(1.0)
    return m


@torch.no_grad()
def channel_usage(m, x, bs=8):
    """Per-channel mean square of the residual stream over all block outputs (dev images)."""
    e = 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = m.get_intermediate_layers(x[i:i + bs].cuda(), n=list(range(len(m.blocks))), norm=False)
        e = e + sum(o.float().pow(2).mean((0, 1)) for o in outs)
    return e / len(outs)


@torch.no_grad()
def reserve(m, S):
    S = torch.as_tensor(S, device=m.pos_embed.device)
    m.patch_embed.proj.weight[S] = 0; m.patch_embed.proj.bias[S] = 0
    m.cls_token[..., S] = 0; m.pos_embed[..., S] = 0
    for blk in m.blocks:
        blk.attn.qkv.weight[:, S] = 0; blk.mlp.fc1.weight[:, S] = 0
        blk.attn.proj.weight[S] = 0; blk.attn.proj.bias[S] = 0
        blk.mlp.fc2.weight[S] = 0; blk.mlp.fc2.bias[S] = 0
    m.norm.weight[S] = 0; m.norm.bias[S] = 0
    return m


def code_table(M=37, freqs=C_CODE_FREQS, scale=1.0):
    """(M, M, 4*len(freqs)) Fourier code of (row, col) in 32-grid units, for the M x M position table."""
    u = torch.arange(M, dtype=torch.float64) * (G0 / (M - 1))
    cols = []
    for axis in (0, 1):
        for k in freqs:
            a = k * OMEGA * u
            for f in (torch.cos, torch.sin):
                v = f(a)
                cols.append(v[:, None].expand(M, M) if axis == 0 else v[None, :].expand(M, M))
    return (torch.stack(cols, -1) * scale).float()


@torch.no_grad()
def install_code(m, S_code, s_ref, S_bal, ballast, scale=1.0):
    pe = m.pos_embed
    M = int((pe.shape[1] - 1) ** 0.5)
    pe[0, 1:, torch.as_tensor(S_code, device=pe.device)] = code_table(M, scale=scale).reshape(M * M, -1).to(pe)
    last = m.blocks[-1].mlp.fc2
    last.bias[torch.as_tensor(S_bal, device=pe.device)] = ballast      # constant ballast, written once at the end
    return m


@torch.no_grad()
def append_bilateral(m, S_code, s_ref, S_bal, ballast, guide_U, nu, sigma_b, A, beta, gamma, scale=1.0,
                     remove_ballast=True, extra_reserved=()):
    """guide_U: (C, kg) orthonormal guide directions over non-reserved channels of z = LN(stream); nu = E|g|^2;
    sigma_b = the (near-constant) LayerNorm std of the ballasted stream."""
    C = m.embed_dim; H = m.blocks[-1].attn.num_heads; D = C // H
    blk = copy.deepcopy(m.blocks[-1])
    dev, dt = blk.attn.qkv.weight.device, blk.attn.qkv.weight.dtype
    nc = len(S_code); kg = guide_U.shape[1]
    assert nc + kg <= D
    Wq = torch.zeros(D, C, device=dev, dtype=torch.float64); Wk = torch.zeros_like(Wq)
    s = sigma_b / scale                                     # z_code - z_ref = scale*code/sigma_b
    for i, c in enumerate(S_code):                          # q.k = A * sum code_i code_j  (times SDPA 1/sqrt(D))
        Wq[i, c] = s * A ** 0.5 * D ** 0.25; Wq[i, s_ref] = -s * A ** 0.5 * D ** 0.25
        Wk[i, c] = s * A ** 0.5 * D ** 0.25; Wk[i, s_ref] = -s * A ** 0.5 * D ** 0.25
    g = guide_U.double().T * (beta / nu) ** 0.5 * D ** 0.25  # (kg, C)
    Wq[nc:nc + kg] = g; Wk[nc:nc + kg] = g
    keep = torch.ones(C, dtype=torch.bool, device=dev)
    keep[torch.as_tensor(list(S_code) + [s_ref] + list(S_bal) + list(extra_reserved), device=dev)] = False
    Wv = torch.diag(keep.double())                          # value = z on ordinary channels, nothing on reserved
    qkv = torch.zeros(3 * C, C, device=dev, dtype=torch.float64)
    for h in range(H):                                      # every head: same bilateral kernel, its own value slice
        qkv[h * D:(h + 1) * D] = Wq; qkv[C + h * D:C + (h + 1) * D] = Wk
    qkv[2 * C:] = Wv
    blk.norm1.weight.fill_(1.0); blk.norm1.bias.zero_()
    blk.attn.qkv.weight.copy_(qkv.to(dt)); blk.attn.qkv.bias.zero_()
    blk.attn.proj.weight.copy_((torch.eye(C, device=dev) * sigma_b * gamma).to(dt)); blk.attn.proj.bias.zero_()
    blk.ls1.gamma.fill_(1.0)
    blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_(); blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
    if remove_ballast:                                                   # only the last appended block
        blk.mlp.fc2.bias[torch.as_tensor(S_bal, device=dev)] = -ballast  # GELU(0)=0 -> MLP output = bias: ballast off
    blk.ls2.gamma.fill_(1.0)
    m.blocks.append(blk); m.n_blocks = len(m.blocks)
    return m


@torch.no_grad()
def _unit_stats(m, x, units, bs=8):
    """Per (block, unit): fc1 pre-activation at the register token (mean), max over patch tokens (mean over images
    and its overall max), and the register token's LayerNorm-2 std. Token layout: [cls, register, patches]."""
    acc = {u: [] for u in units}
    hooks, cur = [], {}
    for b in sorted({b for b, _ in units}):
        blk = m.blocks[b]
        hooks.append(blk.norm2.register_forward_hook(
            lambda mod, inp, out, b=b: cur.__setitem__(("sig", b), inp[0].float().std(-1, unbiased=False))))
        hooks.append(blk.mlp.fc1.register_forward_hook(lambda mod, inp, out, b=b: cur.__setitem__(("pre", b), out.float())))
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x[i:i + bs].cuda(), n=1)
        for b, u in units:
            pre, sig = cur[("pre", b)][..., u], cur[("sig", b)]
            acc[(b, u)].append(torch.stack([pre[:, 1], pre[:, 2:].max(1).values, sig[:, 1]], 1).cpu())
    for h in hooks:
        h.remove()
    return {u: torch.cat(v) for u, v in acc.items()}


@torch.no_grad()
def install_test_time_register(m, x, units, s_flag, s_ref, flag=10.0, margin=8.0, iters=3):
    """Test-time register (Jiang et al. 2025) as a weight program: one extra token whose reserved flag channel is
    `flag`; each register neuron (b, u) gets fc1 bias -M (silent on every ordinary token) and a read of
    (flag - ref)/sigma with weight w, calibrated so that at the register its pre-activation equals the mean per-image
    max patch pre-activation of the unedited neuron (what TTR copies there)."""
    C = m.embed_dim
    st0 = _unit_stats(m, x, units)                                  # before: no register yet -> use patch maxima
    target = {u: float(v[:, 1].mean()) for u, v in st0.items()}
    M = {u: float(v[:, 1].max()) + margin for u, v in st0.items()}
    m.register_tokens = torch.nn.Parameter(torch.zeros(1, 1, C, device=m.cls_token.device, dtype=m.cls_token.dtype))
    m.register_tokens[0, 0, s_flag] = flag
    m.num_register_tokens = 1
    for (b, u) in units:
        m.blocks[b].mlp.fc1.bias[u] -= M[(b, u)]
    w = {u: 0.0 for u in units}
    for _ in range(iters):                                          # sigma at the register depends on earlier units
        st = _unit_stats(m, x, units)
        for (b, u), v in st.items():
            pre_wo = float(v[:, 0].mean()) - w[(b, u)] * float((flag / v[:, 2]).mean())   # pre without our read
            dw = (target[(b, u)] - pre_wo) / float((flag / v[:, 2]).mean()) - w[(b, u)]
            fc1 = m.blocks[b].mlp.fc1
            fc1.weight[u, s_flag] += dw; fc1.weight[u, s_ref] -= dw
            w[(b, u)] += dw
    st = _unit_stats(m, x, units)
    return {f"{b}:{u}": dict(target=target[(b, u)], reg=float(v[:, 0].mean()), reg_sd=float(v[:, 0].std()),
                             patch_max=float(v[:, 1].max())) for (b, u), v in st.items()}


@torch.no_grad()
def final_stream(m, x, bs=8, block=-1):
    out = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m.get_intermediate_layers(x[i:i + bs].cuda(), n=[len(m.blocks) + block if block < 0 else block],
                                                 norm=False)[0].float())
    return torch.cat(out)


@torch.no_grad()
def guide_basis(z, keep, kg=48):
    """Top-kg PCA directions of z (ordinary channels only) and nu = mean |projection|^2."""
    zz = z.reshape(-1, z.shape[-1])[:, keep]
    zz = zz - zz.mean(0)
    _, _, V = torch.pca_lowrank(zz, q=kg, center=False)
    U = torch.zeros(z.shape[-1], kg, device=z.device); U[keep] = V
    nu = float((zz @ V).pow(2).sum(1).mean())
    return U, nu


@torch.no_grad()
def channel_impact(m, x, bs=8):
    """Per-channel cost of reserving it: 1 - mean cosine of final features (ordinary channels) with vs without the cut."""
    def feats(mm):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.cat([mm.get_intermediate_layers(x[i:i + bs].cuda(), n=1, norm=True)[0].float()
                              for i in range(0, len(x), bs)])
    f0 = feats(m); C = f0.shape[-1]; cost = torch.zeros(C)
    for c in range(C):
        mm = reserve(copy.deepcopy(m), [c])
        keep = torch.ones(C, dtype=torch.bool, device=f0.device); keep[c] = False
        cost[c] = 1 - F.cosine_similarity(feats(mm)[..., keep], f0[..., keep], dim=-1).mean()
    return cost


def build_side_channel(m, x_fit, A, beta, gamma, n_res=19, ballast_mult=10.0, kg=48, scale=0.1, verbose=False, info=None,
                       impact_path=None, more=(), ttr_units=None, ttr_flag=10.0, pos_k=0, pos_alpha=1.0, pos_tail=False,
                       white_beta=0.0, white_eps=1.0, fisher_beta=0.0, fisher_src="spair"):
    """Full program on a copy of m. Returns the edited model. more = [(A, beta, gamma), ...]: further bilateral blocks,
    each with its edge guide read from the (already filtered) stream entering it (rolling-guidance filter)."""
    m = fold_affine(copy.deepcopy(m))
    use = channel_usage(m, x_fit)
    from pathlib import Path
    impact_path = Path(impact_path or Path(__file__).parent / "proxy/channel_impact.pt")
    if not impact_path.exists():
        torch.save(channel_impact(m, x_fit[:8]), impact_path)
    S = torch.argsort(torch.load(impact_path))[:n_res + bool(ttr_units)].tolist()  # channels whose removal costs least
    S_code, s_ref, S_bal = S[:4 * len(C_CODE_FREQS)], S[4 * len(C_CODE_FREQS)], S[4 * len(C_CODE_FREQS) + 1:n_res]
    reserve(m, S)
    if ttr_units:
        rep = install_test_time_register(m, x_fit, [tuple(u) for u in ttr_units], S[n_res], s_ref, flag=ttr_flag)
        if verbose:
            print("   TTR", {k: {a: round(b, 2) for a, b in v.items()} for k, v in list(rep.items())[:4]}, flush=True)
        if info is not None:
            info["ttr"] = rep
    xs = final_stream(m, x_fit)
    ballast = ballast_mult * float(xs[:, 1:].norm(dim=-1).quantile(0.99)) / len(S_bal) ** 0.5
    install_code(m, S_code, s_ref, S_bal, ballast, scale=scale)
    xs = final_stream(m, x_fit)                               # now ballasted
    z = F.layer_norm(xs, (xs.shape[-1],))
    sig = xs.float().std(-1, unbiased=False)
    sigma_b = float(sig[:, 1:].median())
    keep = torch.ones(xs.shape[-1], dtype=torch.bool, device=xs.device); keep[torch.as_tensor(S, device=xs.device)] = False
    U, nu = guide_basis(z[:, 1:], keep, kg)
    if info is not None:
        info.update(S=S, ballast=ballast, sigma_b=sigma_b, sigma_spread=float(sig[:, 1:].std() / sig[:, 1:].mean()),
                    usage_frac=float(use[S].sum() / use.sum()))
    if verbose:
        print(f"   reserved {len(S)} ch; ballast {ballast:.0f}; "
              f"LN std {sigma_b:.1f} (spread {float(sig[:, 1:].std() / sig[:, 1:].mean()):.4f})", flush=True)
    stages = [(A, beta, gamma)] + list(more)
    for t, (a, b, g) in enumerate(stages):
        if t > 0:                                             # re-read guide + LN scale from the filtered stream
            xs = final_stream(m, x_fit)
            z = F.layer_norm(xs, (xs.shape[-1],))
            sigma_b = float(xs.float().std(-1, unbiased=False)[:, 1:].median())
            U, nu = guide_basis(z[:, 1:], keep, kg)
        m = append_bilateral(m, S_code, s_ref, S_bal, ballast, U, nu, sigma_b, a, b, g, scale=scale,
                             remove_ballast=t == len(stages) - 1, extra_reserved=S[n_res:])
    if white_beta:                                            # programmed partial whitening (linmap.py)
        import linmap
        blk = m.blocks[-1]
        keepw = torch.ones(m.embed_dim, dtype=torch.bool, device=blk.mlp.fc1.weight.device)
        keepw[torch.as_tensor(S, device=keepw.device)] = False
        _, cov, cw = linmap.stream_stats(m, blk, pos_images(512), 1 + m.num_register_tokens, keepw)
        linmap.program(blk, linmap.whitening_B(cov, white_beta, white_eps).float(), cw, keepw)
    if fisher_beta:                                           # programmed Fisher discriminant (SPair TRAIN keypoints or label-free)
        import linmap
        import wsp_v3
        blk = m.blocks[-1]
        keepf = torch.ones(m.embed_dim, dtype=torch.bool, device=blk.mlp.fc1.weight.device)
        keepf[torch.as_tensor(S, device=keepf.device)] = False
        cov_t, cov_w, _ = wsp_v3.scatter(fisher_src, m, blk, 1 + m.num_register_tokens, patch=14)
        kk = torch.nonzero(keepf).flatten()
        B, _ = linmap.fisher_B(cov_t[kk][:, kk], cov_w[kk][:, kk], fisher_beta)
        _, _, cf = linmap.stream_stats(m, blk, pos_images(512)[:64], 1 + m.num_register_tokens, keepf)
        linmap.program(blk, B.float(), cf, keepf)
    if pos_k:
        rep = install_position_projector(m, pos_k, pos_alpha, exclude=S, tail=pos_tail)
        if verbose:
            print("   position projector", rep, flush=True)
        if info is not None:
            info["pos"] = rep
    return m


@torch.no_grad()
def pos_images(n=512, size=448, path=None):
    """Generic images for estimating the positional subspace: COCO val2017 sorted ids [3000:3000+n] (dev split)."""
    from pathlib import Path
    from PIL import Image
    import dinocmp as dc
    path = Path(path or Path(__file__).parent / f"proxy/pos_imgs_{n}.pt")              # (448 px default)
    if path.exists():
        return torch.load(path)
    files = sorted((Path(os.environ.get("WSP_DATA", str(Path.home() / "Downloads"))) / "coco/val2017").glob("*.jpg"))[3000:3000 + n]
    ims = []
    for f in files:
        im = Image.open(f).convert("RGB"); s = min(im.size)
        ims.append(im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)))
    x = dc.to_batch(ims, size).half()
    torch.save(x, path)
    return x


@torch.no_grad()
def install_position_projector(m, k, alpha=1.0, exclude=(), n_img=512, bs=8, tail=False):
    """Blog-style direction removal: estimate the k-dim positional subspace U of the stream entering the last
    appended block's MLP (top PCs of the per-position mean over n_img images, orthogonal to the all-ones vector and
    to reserved channels) and program that MLP as an exact linear map h -> -alpha U U^T h
    (GELU(z) - GELU(-z) = z; the LayerNorm scale is constant thanks to the ballast). alpha < 0 AMPLIFIES position.
    tail=True writes the LAST 2k hidden units (so it can share the MLP with a Fisher / whitening map in units [0, 2K))."""
    blk = m.blocks[-1]
    x = pos_images(n_img)
    acc, sig, cnt = 0, [], 0
    cur = {}
    h = blk.norm2.register_forward_hook(lambda mod, inp, out: cur.__setitem__("h", inp[0].float()))
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x[i:i + bs].cuda().float(), n=1)
        hh = cur["h"][:, 1 + m.num_register_tokens:]
        acc = acc + hh.sum(0); cnt += len(hh)
        sig.append(hh.std(-1, unbiased=False).flatten().cpu())
    h.remove()
    mean_map = acc / cnt                                         # (P, C) per-position mean
    C = mean_map.shape[1]
    keep = torch.ones(C, dtype=torch.bool, device=mean_map.device)
    if len(exclude):
        keep[torch.as_tensor(list(exclude), device=keep.device)] = False
    D = mean_map - mean_map.mean(0, keepdim=True)
    D[:, ~keep] = 0
    D = D - D.mean(1, keepdim=True) * keep.float()[None] * (C / keep.sum())   # remove the all-ones component
    D[:, ~keep] = 0
    _, Sv, V = torch.linalg.svd(D, full_matrices=False)
    U = V[:k].T                                                 # (C, k)
    sigma = float(torch.cat(sig).median())
    dt = blk.mlp.fc1.weight.dtype; o = blk.mlp.fc1.weight.shape[0] - 2 * k if tail else 0
    blk.mlp.fc1.weight[o:o + k] = U.T.to(dt); blk.mlp.fc1.weight[o + k:o + 2 * k] = -U.T.to(dt)
    blk.mlp.fc2.weight[:, o:o + k] = (-alpha * sigma * U).to(dt); blk.mlp.fc2.weight[:, o + k:o + 2 * k] = (alpha * sigma * U).to(dt)
    if tail:
        assert blk.mlp.fc1.weight[o:o + 2 * k].count_nonzero() > 0 and blk.mlp.fc1.bias[o:o + 2 * k].abs().max() == 0
    ev = (Sv ** 2) / (Sv ** 2).sum()
    return dict(k=k, sigma=sigma, energy_frac=float(ev[:k].sum()), sig_spread=float(torch.cat(sig).std() / torch.cat(sig).mean()))
