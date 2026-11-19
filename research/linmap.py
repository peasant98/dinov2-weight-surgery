"""Exact linear maps programmed into an MLP (no gradients): GELU(z) - GELU(-z) = z, so with fc1 = [I; -I] (on the
chosen channels) and fc2 = [c B, -c B] the MLP outputs c * B z for z = LN(h). Used for the positional projector
(B = -alpha U U^T) and for partial whitening (B = M - I, M = V diag(((lam + eps*mean)/mean)^(-beta/2)) V^T)."""
import torch


@torch.no_grad()
def stream_stats(m, blk, x, start, keep, bs=8):
    """Mean / covariance of z = LN(h) (no affine) over patch tokens entering blk's MLP, and the median LN scale."""
    cur, s1, s2, n, sig = {}, 0, 0, 0, []
    h = blk.norm2.register_forward_hook(lambda mod, inp, out: cur.__setitem__("h", inp[0].float()))
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x[i:i + bs].cuda().float(), n=1)
        hh = cur["h"][:, start:].reshape(-1, cur["h"].shape[-1])
        sig.append(hh.std(-1, unbiased=False).cpu())
        z = torch.nn.functional.layer_norm(hh, (hh.shape[-1],))[:, keep].double()
        s1 = s1 + z.sum(0); s2 = s2 + z.T @ z; n += len(z)
    h.remove()
    mu = s1 / n
    return mu, s2 / n - mu[:, None] * mu[None], float(torch.cat(sig).median())


@torch.no_grad()
def whitening_B(cov, beta, eps):
    lam, V = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0); mean = lam.mean()
    g = ((lam + eps * mean) / ((1 + eps) * mean)) ** (-beta / 2)       # 1 at the mean eigenvalue
    return V @ torch.diag(g) @ V.T - torch.eye(len(lam), dtype=cov.dtype, device=cov.device)


@torch.no_grad()
def program(blk, B, c, keep, bias=None):
    """Write h -> h + c * B LN(h)[keep] (+ bias) into blk's MLP using hidden units [0, 2K)."""
    W1, W2 = blk.mlp.fc1.weight, blk.mlp.fc2.weight
    idx = torch.nonzero(keep).flatten(); K = len(idx)
    assert 2 * K <= W1.shape[0]
    W1[:2 * K] = 0
    W1[torch.arange(K), idx] = 1.0; W1[K + torch.arange(K), idx] = -1.0
    W2[:, :2 * K] = 0
    Bc = (c * B).to(W2.dtype)
    W2[idx[:, None], torch.arange(K)[None].to(idx)] = Bc
    W2[idx[:, None], (K + torch.arange(K))[None].to(idx)] = -Bc
    if bias is not None:
        blk.mlp.fc2.bias[idx] += bias.to(W2.dtype)


@torch.no_grad()
def fisher_B(cov_t, cov_w, beta=0.5, eps=0.05, gmin=0.2, gmax=3.0):
    """Closed-form metric learning: generalized eigenvectors of (within-match scatter cov_w, total scatter cov_t).
    Directions where two views of the same keypoint differ as much as random patches (lambda ~ 1) are nuisance and get
    shrunk; directions with small lambda (consistent across views, varied across patches) get boosted.
    Returns B = A - I with A = I + S V diag(g - 1) V^T, V^T S V = I (S = cov_t regularised), g = (lam/mean)^(-beta/2)."""
    n = len(cov_t)
    S = cov_t + eps * cov_t.diagonal().mean() * torch.eye(n, dtype=cov_t.dtype, device=cov_t.device)
    L = torch.linalg.cholesky(S)
    Li = torch.linalg.inv(L)
    lam, Q = torch.linalg.eigh(Li @ cov_w @ Li.T)
    V = Li.T @ Q
    lam = lam.clamp_min(1e-6)
    g = (lam / lam.mean()).pow(-beta / 2).clamp(gmin, gmax)
    return S @ V @ torch.diag(g - 1) @ V.T, lam
