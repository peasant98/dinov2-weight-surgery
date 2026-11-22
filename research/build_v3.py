"""Rebuilds the DINOv3 ViT-L/16 edit from stock DINOv3 and exports it.

The edit is "v3:rope1.4+mix3s0.3" (selection_rules/SELECTION_RULE_v8.md):
RoPE coordinates scaled by 1.4, then one appended block whose MLP holds a
pooled Fisher map (wsp_v3.py). The exported file holds the RoPE scale and the
appended block without its query/key/value weights (unused: the block's
attention output projection is zero). The script loads the file back through
`dinov2_wsp.dinov3_vitl16_wsp` and compares patch features with the built model
and with the release.

Needs WSP_DATA with SPair-71k, DAVIS-2017 and ADE20K under benchmarks/ and
COCO val2017 under coco/, plus the DINOv3 code and stock weights
(WSP_DINOV3_REPO, WSP_DINOV3_WEIGHTS).

Usage:
    python build_v3.py [out.pth]
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "downstream"))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dinov2_wsp  # noqa: E402
import wsp_v3  # noqa: E402

NAME = "v3:rope1.4+mix3s0.3"  # the released pick


def export_edit(model):
    """Collects the edit from a built model.

    Args:
        model: DINOv3 ViT-L/16 built with `wsp_v3.build_v3` for NAME.

    Returns:
        Dict with "rope_scale", "rope_periods" (the edited RoPE periods) and
        "block" (appended block state dict, float32, on CPU, without
        attn.qkv.*).
    """
    (fn, kw), _ = wsp_v3.VARIANTS_V3[NAME]
    assert fn == "rope_scale"
    block = {
        k: v.float().cpu().clone()
        for k, v in model.blocks[-1].state_dict().items()
        if not k.startswith("attn.qkv.")
    }
    periods = model.rope_embed.periods.float().cpu().clone()
    return {"rope_scale": float(kw["k"]), "rope_periods": periods,
            "block": block}


@torch.no_grad()
def compare(a_model, b_model):
    """Patch-feature cosine and max abs difference on a fixed random input.

    Args:
        a_model: First model.
        b_model: Second model.

    Returns:
        (mean cosine, max abs difference), both floats.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 448, 448).cuda()
    a = a_model.get_intermediate_layers(x, n=1, norm=True)[0]
    b = b_model.get_intermediate_layers(x, n=1, norm=True)[0]
    cos = float(F.cosine_similarity(a, b, dim=-1).mean())
    return cos, float((a - b).abs().max())


def main():
    print("edit chain:", *wsp_v3.VARIANTS_V3[NAME], sep="\n  ")
    model = C.load(NAME)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rebuilt_v3.pth"
    torch.save(export_edit(model), out)
    loaded = dinov2_wsp.dinov3_vitl16_wsp(
        str(C.dc.DINOV3_W), edit=out, dinov3_repo=str(C.dc.DINOV3_REPO),
        dinov3_source="local").cuda()
    cos, diff = compare(model, loaded)
    print(f"saved {out} | reloaded vs built: cosine {cos:.6f}, "
          f"max abs diff {diff:.4g}")
    del loaded
    cos, diff = compare(model, C.load("wsp3"))
    print(f"built vs release: cosine {cos:.6f}, max abs diff {diff:.4g}")


if __name__ == "__main__":
    main()
