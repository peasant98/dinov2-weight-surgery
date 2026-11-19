"""Rebuilds the released model from stock DINOv2 ViT-L/14 with the closed-form edits and compares it to the release.

Needs: WSP_FIT_SCANS (comma list of folders under WSP_DATA, each with an images/ subfolder; ~24 natural images are
enough), COCO val2017 under WSP_DATA/coco and SPair-71k under WSP_DATA/benchmarks/spair (TRAIN pairs, for the Fisher
map). The fit images we used are not released, so a rebuild matches the release closely but not bit for bit.

usage: python build.py [out.pth]
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "downstream"))
import common as C  # noqa: E402

NAME = "PICK6+fisher b0.5+D23mlp0"                       # the released pick; its edit chain is C.VARIANTS[NAME]

if __name__ == "__main__":
    print("edit chain:", *C.VARIANTS[NAME], sep="\n  ")
    model = C.load(NAME)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rebuilt.pth"
    torch.save({k: v.float().cpu() for k, v in model.state_dict().items()}, out)
    ref = C.load("wsp")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 448, 448).cuda()
    with torch.no_grad():
        a = model.get_intermediate_layers(x, n=1, norm=True)[0]
        b = ref.get_intermediate_layers(x, n=1, norm=True)[0]
    print(f"saved {out} | blocks {len(model.blocks)} | patch-feature cosine to the release "
          f"{float(F.cosine_similarity(a, b, dim=-1).mean()):.6f} (max abs diff {float((a - b).abs().max()):.4g})")
