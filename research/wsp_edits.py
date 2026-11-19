"""Weight-space edits of DINOv2-L (no gradients, no DINOv3). Each takes the model and returns it edited in place.
Plus the DINOv3-free proxy: SAM2-segment contrast AUC on DINO patch features."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

REG7 = [(17, 884), (17, 2373), (17, 133), (17, 2436), (18, 208), (18, 530), (18, 2011)]


# ---------------------------------------------------------------- edits
def zero_units(m, units=REG7):
    """Silence MLP hidden units (zero their output column): removes the high-norm artifact writers."""
    with torch.no_grad():
        for b, u in units:
            m.blocks[b].mlp.fc2.weight[:, u] = 0
    return m


def pos_lowpass(m, sigma):
    """Gaussian low-pass of the learned 37x37 position table."""
    if sigma <= 0:
        return m
    with torch.no_grad():
        pe = m.pos_embed[:, 1:]
        M = int(pe.shape[1] ** 0.5); C = pe.shape[-1]
        g = pe.reshape(1, M, M, C).permute(0, 3, 1, 2)
        r = int(np.ceil(3 * sigma)); x = torch.arange(-r, r + 1, device=g.device, dtype=g.dtype)
        k = torch.exp(-x ** 2 / (2 * sigma ** 2)); k = k / k.sum()
        g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
        g = F.conv2d(F.pad(g, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
        m.pos_embed[:, 1:] = g.permute(0, 2, 3, 1).reshape(1, M * M, C)
    return m


def pos_dct_lowpass(m, k):
    """Keep only the lowest k x k 2D-DCT frequencies of the 37x37 position table (k >= 37 = unchanged)."""
    with torch.no_grad():
        pe = m.pos_embed[:, 1:].float()
        M = int(pe.shape[1] ** 0.5); C = pe.shape[-1]
        n = torch.arange(M, device=pe.device, dtype=pe.dtype)
        D = torch.cos(np.pi * (n[None] + 0.5) * n[:, None] / M) * torch.sqrt(torch.tensor(2.0 / M))
        D[0] /= np.sqrt(2.0)                                                        # orthonormal DCT-II (M, M)
        g = pe.reshape(M, M, C)
        coef = torch.einsum("ui,ijc,vj->uvc", D, g, D)
        coef[k:] = 0; coef[:, k:] = 0
        m.pos_embed[:, 1:] = torch.einsum("ui,uvc,vj->ijc", D, coef, D).reshape(1, M * M, C).to(m.pos_embed.dtype)
    return m


def pos_scale(m, beta):
    """Shrink the patch position table toward its mean by beta (0 = no positional signal beyond the mean)."""
    with torch.no_grad():
        pe = m.pos_embed[:, 1:]
        mu = pe.mean(1, keepdim=True)
        m.pos_embed[:, 1:] = mu + beta * (pe - mu)
    return m


def diag_antialias(m, on=True):
    """DIAGNOSTIC ONLY (not a weight edit): switch the pos-embed resize to antialiased bicubic."""
    m.interpolate_antialias = on
    return m


def damp_late(m, blocks, alpha):
    """Scale LayerScale gammas of late blocks (alpha=0 turns a block into identity)."""
    with torch.no_grad():
        for b in blocks:
            m.blocks[b].ls1.gamma.mul_(alpha); m.blocks[b].ls2.gamma.mul_(alpha)
    return m


def attn_temp(m, blocks, tau):
    """Multiply attention logits by tau (<1 = softer, more averaging) via the query rows of qkv."""
    with torch.no_grad():
        for b in blocks:
            qkv = m.blocks[b].attn.qkv; C = qkv.weight.shape[1]
            qkv.weight[:C].mul_(tau)
            if qkv.bias is not None:
                qkv.bias[:C].mul_(tau)
    return m


def append_nlm_blocks(m, basis, nu, n=1, temp=8.0, gamma=1.0):
    """Hand-written denoising blocks appended after the last block.
    Every head gets the same query = key = sqrt(temp * 64) * (top-64 PCA projection of LN(x)) / |.|-scale, so all
    heads share one attention map that weighs patches by feature similarity (non-local means); logits are
    temp * <p_i, p_j> / nu with nu = mean |p|^2, i.e. ~temp * cosine. Each head's value is
    its own 64-dim slice of LN(x) and the output projection is identity, so the block adds gamma * (similarity-
    weighted average of LN(x)) to the residual. MLP is switched off. basis: (1024, 64) PCA directions of LN(x)."""
    C = m.embed_dim; H = m.blocks[-1].attn.num_heads; d = C // H
    for _ in range(n):
        blk = copy.deepcopy(m.blocks[-1])
        with torch.no_grad():
            blk.norm1.weight.fill_(1.0); blk.norm1.bias.zero_()
            Wq = (basis.T * np.sqrt(temp * np.sqrt(d) / nu)).to(blk.attn.qkv.weight)   # (64, C); logits ~ temp*cos
            blk.attn.qkv.weight.zero_(); blk.attn.qkv.bias.zero_()
            for h in range(H):
                blk.attn.qkv.weight[h * d:(h + 1) * d] = Wq                               # q
                blk.attn.qkv.weight[C + h * d:C + (h + 1) * d] = Wq                       # k
            blk.attn.qkv.weight[2 * C:] = torch.eye(C, device=Wq.device, dtype=Wq.dtype)  # v = LN(x)
            blk.attn.proj.weight.copy_(torch.eye(C)); blk.attn.proj.bias.zero_()
            blk.ls1.gamma.fill_(gamma); blk.ls2.gamma.zero_()
        m.blocks.append(blk)
    m.n_blocks = len(m.blocks)
    return m


def append_mean_subtract_block(m, gamma):
    """Hand-written block: query = key = 0 gives uniform attention, so each head returns the image mean of its LN(x)
    slice; value = identity, output projection = -identity, scale gamma -> x - gamma * mean_tokens(LN(x)). MLP off."""
    C = m.embed_dim
    blk = copy.deepcopy(m.blocks[-1])
    with torch.no_grad():
        blk.norm1.weight.fill_(1.0); blk.norm1.bias.zero_()
        blk.attn.qkv.weight.zero_(); blk.attn.qkv.bias.zero_()
        blk.attn.qkv.weight[2 * C:] = torch.eye(C, device=blk.attn.qkv.weight.device, dtype=blk.attn.qkv.weight.dtype)
        blk.attn.proj.weight.copy_(-torch.eye(C)); blk.attn.proj.bias.zero_()
        blk.ls1.gamma.fill_(gamma); blk.ls2.gamma.zero_()
    m.blocks.append(blk)
    m.n_blocks = len(m.blocks)
    return m


@torch.no_grad()
def guide_readout(m, x, src_block=13, k=64, ridge=1e-1):
    """Closed-form (ridge) linear readout W (C, k) such that LN(final stream) @ W ~ top-k PCA of LN(block src_block).
    Only DINOv2's own activations are used. Returns W, nu (mean |p|^2) and the readout R^2."""
    outs = []
    for i in range(0, len(x), 8):
        a, b = m.get_intermediate_layers(x[i:i + 8], n=[src_block, len(m.blocks) - 1], norm=False)
        outs.append((F.layer_norm(a.float(), a.shape[-1:]).reshape(-1, a.shape[-1]),
                     F.layer_norm(b.float(), b.shape[-1:]).reshape(-1, b.shape[-1])))
    Z13 = torch.cat([o[0] for o in outs]); Z = torch.cat([o[1] for o in outs])
    mu13 = Z13.mean(0)
    _, _, V = torch.pca_lowrank(Z13 - mu13, q=k, center=False)
    T = (Z13 - mu13) @ V                                                            # (N, k) mid-layer structure
    A = Z.T @ Z + ridge * len(Z) * torch.eye(Z.shape[1], device=Z.device)
    W = torch.linalg.solve(A, Z.T @ T)                                              # (C, k)
    P = Z @ W
    r2 = float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())
    return W, float((P ** 2).sum(-1).mean()), r2


@torch.no_grad()
def ln_basis(m, x, k=64):
    """Top-k PCA directions of LN(last-block output) over the given images (DINOv2's own statistics)."""
    o = m.get_intermediate_layers(x, n=1, norm=False)[0].float()
    z = F.layer_norm(o, o.shape[-1:]).reshape(-1, o.shape[-1])
    z = z - z.mean(0)
    _, _, V = torch.pca_lowrank(z, q=k, center=False)
    nu = float(((z @ V) ** 2).sum(-1).mean())
    return V, nu                                                                           # (C, k), mean |p|^2


def build_variant(base, spec, x_fit, verbose=True):
    """Apply an edit chain to a copy of base. x_fit: DINOv2 dev images used only by the closed-form guide readout."""
    m = copy.deepcopy(base)
    for fn, kw in spec:
        if fn == "append_guided_block":
            W, nu, r2 = guide_readout(m, x_fit, src_block=kw.get("src", 13), k=64, ridge=kw.get("ridge", 0.1))
            if verbose:
                print(f"   guide readout R^2 {r2:.3f}", flush=True)
            m = append_nlm_blocks(m, W, nu, n=kw.get("n", 1), temp=kw["temp"], gamma=kw["gamma"])
        elif fn == "append_bilateral_block":
            Wg, nu_g, r2g = guide_readout(m, x_fit, src_block=kw.get("src", 13), k=48, ridge=kw.get("ridge", 0.1))
            Wp, r2p = pos_readout(m, x_fit)
            if verbose:
                print(f"   guide R^2 {r2g:.3f}  position-code R^2 {r2p:.3f}", flush=True)
            m = append_bilateral_block(m, Wg, nu_g, Wp, temp=kw["temp"], lam=kw["lam"], gamma=kw["gamma"])
        elif fn == "side_channel":
            import side_channel as sc
            m = sc.build_side_channel(m, x_fit, **kw)
        elif fn == "append_spectral_block":
            m = append_spectral_block(m, x_fit, **kw)
        elif fn == "append_nlm_blocks":
            V, nu = ln_basis(m, x_fit)
            m = append_nlm_blocks(m, V, nu, **kw)
        else:
            m = globals()[fn](m, **kw)
    return m


@torch.no_grad()
def pos_readout(m, x, grid=32, freqs=(1, 2, 3, 4), ridge=1e-1):
    """Closed-form linear readout of low-frequency Fourier codes of each patch's (row, col) from LN(final stream).
    Targets per axis and frequency k: cos/sin(2*pi*k*pos/(2*grid)). Their dot product between patches i, j is
    sum_k cos(w_k (y_i-y_j)) + cos(w_k (x_i-x_j)): a bump peaked at zero distance. Returns W (C, 4*len(freqs)), R^2."""
    Zs = []
    for i in range(0, len(x), 8):
        b = m.get_intermediate_layers(x[i:i + 8], n=1, norm=False)[0].float()
        Zs.append(F.layer_norm(b, b.shape[-1:]).reshape(-1, b.shape[-1]))
    Z = torch.cat(Zs)
    yy, xx = np.divmod(np.arange(grid * grid), grid)
    T = []
    for k in freqs:
        w = 2 * np.pi * k / (2 * grid)
        T += [np.cos(w * yy), np.sin(w * yy), np.cos(w * xx), np.sin(w * xx)]
    T = torch.tensor(np.stack(T, 1), dtype=Z.dtype, device=Z.device).repeat(len(Z) // (grid * grid), 1)
    A = Z.T @ Z + ridge * len(Z) * torch.eye(Z.shape[1], device=Z.device)
    W = torch.linalg.solve(A, Z.T @ T)
    P = Z @ W
    return W, float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


def append_bilateral_block(m, Wg, nu_g, Wp, temp, lam, gamma):
    """Hand-written bilateral anchoring block: like append_nlm_blocks, but query = key = [feature guide (48 dims),
    position code (16 dims)], so logits = temp * <g_i, g_j>/nu_g + lam * sum_k cos(w_k * dpos): patches attend to
    ones that look alike (mid-layer structure) AND are nearby."""
    C = m.embed_dim; H = m.blocks[-1].attn.num_heads; d = C // H
    kg, kp = Wg.shape[1], Wp.shape[1]
    assert kg + kp == d
    blk = copy.deepcopy(m.blocks[-1])
    with torch.no_grad():
        blk.norm1.weight.fill_(1.0); blk.norm1.bias.zero_()
        sc = np.sqrt(np.sqrt(d))                                             # undo the 1/sqrt(d) attention scale
        Wq = torch.cat([Wg.T * np.sqrt(temp / nu_g), Wp.T * np.sqrt(lam)], 0) * sc
        Wq = Wq.to(blk.attn.qkv.weight)
        blk.attn.qkv.weight.zero_(); blk.attn.qkv.bias.zero_()
        for h in range(H):
            blk.attn.qkv.weight[h * d:(h + 1) * d] = Wq
            blk.attn.qkv.weight[C + h * d:C + (h + 1) * d] = Wq
        blk.attn.qkv.weight[2 * C:] = torch.eye(C, device=Wq.device, dtype=Wq.dtype)
        blk.attn.proj.weight.copy_(torch.eye(C)); blk.attn.proj.bias.zero_()
        blk.ls1.gamma.fill_(gamma); blk.ls2.gamma.zero_()
    m.blocks.append(blk)
    m.n_blocks = len(m.blocks)
    return m


# ---------------------------------------------------------------- proxy
def retrieval_acc(feats, labels, group, per_img=64, seed=0):
    """Semantic guard: share of sampled patches whose nearest neighbour in OTHER images comes from the same scan."""
    g = torch.Generator().manual_seed(seed)
    P, owner = [], []
    for i, (f, l) in enumerate(zip(feats, labels)):
        idx = torch.nonzero(l >= 0).squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)[:per_img]]
        P.append(F.normalize(f[idx].float(), dim=-1)); owner += [i] * len(idx)
    P = torch.cat(P).cuda(); owner = torch.tensor(owner).cuda(); grp = torch.as_tensor(group).cuda()[owner]
    S = P @ P.T
    S[owner[:, None] == owner[None]] = -2                                            # exclude same image
    return float((grp[S.argmax(1)] == grp).float().mean())


def _auc(pos, neg):
    """P(pos > neg) via ranks (Mann-Whitney)."""
    s = torch.cat([pos, neg]); r = torch.empty_like(s); r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype, device=s.device)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def segment_auc(feats, labels, grid=32, near=3):
    """Mean over images of AUC(same-segment vs different-segment patch cosine), all pairs and pairs <= near patches apart."""
    yy, xx = np.divmod(np.arange(grid * grid), grid)
    D = torch.from_numpy(np.maximum(np.abs(yy[:, None] - yy[None]), np.abs(xx[:, None] - xx[None])))
    a_all, a_near = [], []
    for f, l in zip(feats, labels):
        keep = l >= 0
        f = F.normalize(f[keep].float().cuda(), dim=-1); l = l[keep].cuda(); d = D[keep][:, keep].cuda()
        S = f @ f.T
        iu = torch.triu_indices(len(l), len(l), 1, device=S.device)
        s, same, dist = S[iu[0], iu[1]], (l[iu[0]] == l[iu[1]]), d[iu[0], iu[1]]
        if same.any() and (~same).any():
            a_all.append(_auc(s[same], s[~same]))
        nm = dist <= near
        if (same & nm).any() and (~same & nm).any():
            a_near.append(_auc(s[same & nm], s[~same & nm]))
    return float(np.mean(a_all)), float(np.mean(a_near))


def neighbour_roughness(feats, labels, grid=32):
    """DINOv3-free smoothness proxy on per-image centred unit features over 4-neighbour patch pairs:
    (inside = mean 1-cos for pairs in the same SAM segment [speckle, lower = smoother],
     across = mean 1-cos for pairs straddling a segment boundary [edge contrast, should stay high])."""
    ins, acr = [], []
    for f, l in zip(feats, labels):
        f = F.normalize(f.float() - f.float().mean(0), dim=-1).reshape(grid, grid, -1)
        l = l.reshape(grid, grid)
        for fa, fb, la, lb in ((f[:, :-1], f[:, 1:], l[:, :-1], l[:, 1:]), (f[:-1], f[1:], l[:-1], l[1:])):
            d = 1 - (fa * fb).sum(-1); ok = (la >= 0) & (lb >= 0)
            ins.append(d[ok & (la == lb)]); acr.append(d[ok & (la != lb)])
    return float(torch.cat(ins).mean()), float(torch.cat(acr).mean())


def mean_ablate_texture(m, blocks, k, ratio_min=0.0, path=None, use_mean=True, keep=0.0):
    """Replace the output of each block's top-k texture MLP units (texture_units.py ranking, unit must have
    within/across-segment activation-difference ratio > ratio_min) by its dev-set mean: fc2.bias += w_u * mean_u, w_u = 0."""
    from pathlib import Path
    T = torch.load(path or Path(__file__).parent / "proxy/texunits.pt")
    with torch.no_grad():
        for b in blocks:
            t = T[b]; o = t["order"][t["ratio"][t["order"]] > ratio_min][:k]
            fc2 = m.blocks[b].mlp.fc2
            if use_mean:                                   # keep > 0: damp (w -> keep*w) instead of removing
                fc2.bias += (1 - keep) * (fc2.weight[:, o] * t["mean"][o].to(fc2.weight)).sum(1)
            fc2.weight[:, o] *= keep
    return m


def zero_heads(m, heads):
    """Silence attention heads (zero their columns of the output projection)."""
    with torch.no_grad():
        for b, h in heads:
            D = m.blocks[b].attn.proj.weight.shape[1] // m.blocks[b].attn.num_heads
            m.blocks[b].attn.proj.weight[:, h * D:(h + 1) * D] = 0
    return m


@torch.no_grad()
def spectral_basis(m, x, labels, bs=8):
    """PCA of the (affine-free) layer-normed final stream over dev images, and each component's speckle ratio:
    r_k = within-SAM-segment / across-boundary mean squared neighbour difference of the component's spatial map."""
    xs = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xs.append(m.get_intermediate_layers(x[i:i + bs].cuda(), n=1, norm=False)[0].float())
    z = F.layer_norm(torch.cat(xs), (xs[0].shape[-1],))                           # (N, P, C)
    mu = z.reshape(-1, z.shape[-1]).mean(0)
    zc = z - mu
    cov = zc.reshape(-1, zc.shape[-1]).T @ zc.reshape(-1, zc.shape[-1]) / (zc.shape[0] * zc.shape[1])
    var, U = torch.linalg.eigh(cov.double()); var, U = var.flip(0).float(), U.flip(1).float()
    p = (zc @ U).cpu()                                                             # (N, P, C) component maps
    G = int(p.shape[1] ** 0.5); ein = eacr = 0; nin = nacr = 0
    for k in range(len(p)):
        a = p[k].reshape(G, G, -1); l = labels[k].reshape(G, G)
        for fa, fb, la, lb in ((a[:, :-1], a[:, 1:], l[:, :-1], l[:, 1:]), (a[:-1], a[1:], l[:-1], l[1:])):
            d = (fa - fb) ** 2; ok = (la >= 0) & (lb >= 0)
            ein = ein + d[ok & (la == lb)].sum(0); nin += int((ok & (la == lb)).sum())
            eacr = eacr + d[ok & (la != lb)].sum(0); nacr += int((ok & (la != lb)).sum())
    return U, mu, var, (ein / nin) / (eacr / nacr)


def append_linear_block(m, M, bias, gamma=1000.0):
    """Hand-written block computing an EXACT per-token linear map of LN(x): the MLP uses GELU(z) - GELU(-z) = z with
    fc1 = [A; -A], fc2 = [B, -B] (M = B A), attention off, LayerScale gamma >> 1 so the final norm sees ~ M LN(x) + bias.
    M invertible => no information is lost (a linear probe can undo it); only the cosine geometry changes."""
    C = m.embed_dim
    blk = copy.deepcopy(m.blocks[-1])
    with torch.no_grad():
        dev, dt = blk.mlp.fc1.weight.device, blk.mlp.fc1.weight.dtype
        blk.ls1.gamma.zero_()
        blk.norm2.weight.fill_(1.0); blk.norm2.bias.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_(); blk.mlp.fc2.weight.zero_()
        I = torch.eye(C, device=dev, dtype=dt)
        blk.mlp.fc1.weight[:C] = I; blk.mlp.fc1.weight[C:2 * C] = -I
        blk.mlp.fc2.weight[:, :C] = M.to(dev, dt); blk.mlp.fc2.weight[:, C:2 * C] = -M.to(dev, dt)
        blk.mlp.fc2.bias.copy_(bias.to(dev, dt))
        blk.ls2.gamma.fill_(gamma)
    m.blocks.append(blk)
    m.n_blocks = len(m.blocks)
    return m


def spectral_gains(r, beta, smin=0.1, smax=4.0, r0=None):
    """Per-component gain: speckled components (high within/across ratio r) turned down, smooth ones up."""
    r0 = r.median() if r0 is None else r0
    return ((r0 / r) ** beta).clamp(smin, smax)


def append_spectral_block(m, x_fit, beta, smin=0.1, smax=4.0, gamma=1000.0, labels_path=None):
    from pathlib import Path
    labels = torch.load(labels_path or Path(__file__).parent / "proxy/labels_dev.pt")[:len(x_fit)]
    U, mu, var, r = spectral_basis(m, x_fit, labels)
    s = spectral_gains(r, beta, smin, smax).to(U)
    M = U @ torch.diag(s) @ U.T
    return append_linear_block(m, M, mu - M @ mu, gamma)


@torch.no_grad()
def kk_attention(m, blocks, lam=1.0):
    """NACLIP/SCLIP-style key-key attention as a weight edit: W_q <- (1-lam) W_q + lam W_k (and bias), so a token
    attends to tokens whose keys resemble its own key. Stock architecture."""
    C = m.embed_dim
    for b in blocks:
        qkv = m.blocks[b].attn.qkv
        qkv.weight[:C] = (1 - lam) * qkv.weight[:C] + lam * qkv.weight[C:2 * C]
        if qkv.bias is not None:
            qkv.bias[:C] = (1 - lam) * qkv.bias[:C] + lam * qkv.bias[C:2 * C]
    return m


@torch.no_grad()
def attn_dominant(m, blocks=(-1,), attn_gain=1.0, mlp_gain=1.0):
    """ClearCLIP-style: scale a block's attention branch (LayerScale 1) and MLP branch (LayerScale 2). With the final
    LayerNorm, a large attn_gain ~ dropping the residual; mlp_gain 0 drops the FFN. Stock architecture."""
    for b in blocks:
        m.blocks[b].ls1.gamma *= attn_gain
        m.blocks[b].ls2.gamma *= mlp_gain
    return m


class NormThenLinear(torch.nn.Module):
    """final LayerNorm followed by a fixed (closed-form) linear map: x -> LN(x) @ A^T."""

    def __init__(self, norm, A):
        super().__init__()
        self.norm = norm
        self.register_buffer("A", A.float())

    def forward(self, x):
        return self.norm(x) @ self.A.T.to(x.dtype)


@torch.no_grad()
def gram_anchor(m, layer=21, hires=False, alpha=1.0, n_img=256, ridge=1e-3, bs=8):
    """DINOv3-style Gram anchoring as a closed-form weight edit: a linear map A after the final norm such that the patch
    Gram of the output ~ the Gram of block `layer` (the 'Gram teacher'; hires = teacher at 2x input, bicubic-downsampled).
    min_M sum_i ||X_i M X_i^T - Y_i Y_i^T||^2 with pooled C: M = C^-1 R C^-1, A = M^(1/2) (rows L2-normalised X, Y)."""
    import torch.nn.functional as F
    import side_channel as sc
    x = sc.pos_images(512)[:n_img]
    d = m.embed_dim; C = torch.zeros(d, d, dtype=torch.float64, device="cuda"); R = torch.zeros_like(C)
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            X = m.get_intermediate_layers(xb, n=1, norm=True, reshape=True)[0].float()             # (B, d, g, g)
            xt = F.interpolate(xb, scale_factor=2, mode="bicubic", align_corners=False) if hires else xb
            Y = m.get_intermediate_layers(xt, n=[layer], norm=True, reshape=True)[0].float()
        if hires:
            Y = F.interpolate(Y, size=X.shape[-2:], mode="bicubic", align_corners=False)
        X = F.normalize(X.flatten(2).transpose(1, 2), dim=-1).double(); Y = F.normalize(Y.flatten(2).transpose(1, 2), dim=-1).double()
        C += torch.einsum("bpi,bpj->ij", X, X)
        S = torch.einsum("bpi,bpj->bij", X, Y)                                                       # (B, d, d)
        R += torch.einsum("bij,bkj->ik", S, S)
    n = len(x)
    C, R = C / n, R / n
    Ci = torch.linalg.inv(C + ridge * C.diagonal().mean() * torch.eye(d, dtype=C.dtype, device=C.device))
    M = Ci @ R @ Ci; M = (M + M.T) / 2
    lam, V = torch.linalg.eigh(M)
    A = V @ torch.diag(lam.clamp_min(0).sqrt()) @ V.T
    A = A / A.diagonal().mean()                                                                       # unit average gain
    A = (1 - alpha) * torch.eye(d, dtype=A.dtype, device=A.device) + alpha * A
    m.norm = NormThenLinear(m.norm, A)
    return m


@torch.no_grad()
def gram_readout(m, layer=21, hires=False, n_img=256, ridge=1e-2, bs=8, alpha=1.0):
    """Gram anchoring v2: ridge-regression readout of the Gram teacher (block `layer`, optional 2x-res teacher) from the
    final LN output, installed after the final norm: out = LN(x) @ W. If XW ~ Y then the output Gram ~ teacher Gram."""
    import torch.nn.functional as F
    import side_channel as sc
    x = sc.pos_images(512)[:n_img]
    d = m.embed_dim; XX = torch.zeros(d, d, dtype=torch.float64, device="cuda"); XY = torch.zeros_like(XX)
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            X = m.get_intermediate_layers(xb, n=1, norm=True, reshape=True)[0].float()
            xt = F.interpolate(xb, scale_factor=2, mode="bicubic", align_corners=False) if hires else xb
            Y = m.get_intermediate_layers(xt, n=[layer], norm=True, reshape=True)[0].float()
        if hires:
            Y = F.interpolate(Y, size=X.shape[-2:], mode="bicubic", align_corners=False)
        X = X.flatten(2).transpose(1, 2).reshape(-1, d).double(); Y = Y.flatten(2).transpose(1, 2).reshape(-1, d).double()
        XX += X.T @ X; XY += X.T @ Y
    W = torch.linalg.solve(XX + ridge * XX.diagonal().mean() * torch.eye(d, dtype=XX.dtype, device=XX.device), XY)
    W = (1 - alpha) * torch.eye(d, dtype=W.dtype, device=W.device) + alpha * W                    # partial anchoring
    m.norm = NormThenLinear(m.norm, W.T)
    return m
