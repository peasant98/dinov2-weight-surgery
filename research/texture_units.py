"""Score every MLP hidden unit of edits1+2 DINOv2-L (blocks 4-23) as a "texture" unit (dev frames, DINOv3 never loaded):
  e_in  = mean squared activation difference between 4-neighbour patches in the SAME SAM segment (speckle it writes)
  e_x   = same across a segment boundary (useful edge signal)
  score = e_in * ||fc2 column||^2 , kept only if e_in / e_x > ratio_min (unit ignores object boundaries)
Saves per block: units sorted by score, their mean activation (for mean-ablation) -> proxy/texunits.pt"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import dinocmp as dc  # noqa: E402
import wsp_edits as we  # noqa: E402

here = Path(__file__).parent
U14 = [tuple(u) for u in json.load(open(here / "proxy/regunits.json"))]
labels = torch.load(here / "proxy/labels_dev.pt")
x = dc.to_batch(dc.load_images(8), 448)
SPEC = [("pos_lowpass", {"sigma": 1.5}), ("zero_units", {"units": U14})]
if len(sys.argv) > 2:                                       # optional: score on top of an already texture-ablated model
    SPEC += [("mean_ablate_texture", {"blocks": list(range(7, 24)), "k": 4096, "ratio_min": float(sys.argv[2]), "path": sys.argv[3]})]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "proxy/texunits.pt"
m = we.build_variant(dc.load_dinov2("dinov2_vitl14"), SPEC, x[:24].cuda(), verbose=False)
BLOCKS, G = list(range(4, 24)), 32


def pairs(l):
    l = l.reshape(G, G); idx = torch.arange(G * G).reshape(G, G)
    a = torch.cat([idx[:, :-1].flatten(), idx[:-1].flatten()]); b = torch.cat([idx[:, 1:].flatten(), idx[1:].flatten()])
    la, lb = l.flatten()[a], l.flatten()[b]; ok = (la >= 0) & (lb >= 0)
    return a[ok & (la == lb)], b[ok & (la == lb)], a[ok & (la != lb)], b[ok & (la != lb)]


acc = {b: dict(ein=0, nin=0, ex=0, nx=0, s=0, n=0) for b in BLOCKS}
cur = {}


def hook(b):
    def f(mod, inp, out):
        a = out[:, 1 + 4:].float() if out.shape[1] == 1 + 4 + G * G else out[:, 1:].float()   # drop cls (+registers)
        st = acc[b]
        for k in range(len(a)):
            ia, ib, xa, xb = cur["pairs"][k]
            st["ein"] += ((a[k, ia] - a[k, ib]) ** 2).sum(0); st["nin"] += len(ia)
            st["ex"] += ((a[k, xa] - a[k, xb]) ** 2).sum(0); st["nx"] += len(xa)
        st["s"] += a.sum((0, 1)); st["n"] += a.shape[0] * a.shape[1]
    return f


hs = [m.blocks[b].mlp.act.register_forward_hook(hook(b)) for b in BLOCKS]
with torch.no_grad():
    for i in range(0, len(x), 8):
        cur["pairs"] = [tuple(t.cuda() for t in pairs(labels[j])) for j in range(i, i + 8)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.get_intermediate_layers(x[i:i + 8].cuda(), n=1)
for h in hs:
    h.remove()
out = {}
for b in BLOCKS:
    st = acc[b]
    ein, ex = st["ein"] / st["nin"], st["ex"] / st["nx"]
    w2 = m.blocks[b].mlp.fc2.weight.float().pow(2).sum(0) * m.blocks[b].ls2.gamma.float().pow(2).mean()
    score = ein * w2
    order = torch.argsort(score, descending=True)
    out[b] = dict(order=order.cpu(), score=score.cpu(), ratio=(ein / ex).cpu(), mean=(st["s"] / st["n"]).cpu())
    top = order[:256]
    print(f"block {b:2d}: share of within-seg MLP energy in top-256 units {float(score[top].sum() / score.sum()):.2f}  "
          f"median in/x ratio top-256 {float((ein / ex)[top].median()):.2f} vs all {float((ein / ex).median()):.2f}", flush=True)
torch.save(out, OUT)
