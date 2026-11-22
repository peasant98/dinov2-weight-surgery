"""Loaders for the weight-edited DINOv2 ViT-L/14 and DINOv3 ViT-L/16.

Each edited model is the stock model with one extra transformer block appended
(25 blocks instead of 24); the DINOv3 edit also rescales the RoPE coordinates.
Every weight was written in closed form; nothing was trained.
"""

import copy

import torch

WEIGHTS_URL = (
    "https://github.com/peasant98/dinov2-weight-surgery/releases/download/"
    "v1.0/dinov2_vitl14_wsp.pth"
)
DINOV3_EDIT_URL = (
    "https://github.com/peasant98/dinov2-weight-surgery/releases/download/"
    "v1.1/dinov3_vitl16_wsp_edit.pth"
)
NUM_APPENDED_BLOCKS = 1


def dinov2_vitl14_wsp(weights=None, pretrained=True):
    """Builds the weight-edited DINOv2 ViT-L/14.

    Args:
        weights: Path to a local `dinov2_vitl14_wsp.pth`. If None, the file is
            downloaded from the GitHub release.
        pretrained: If False, returns the 25-block architecture with stock
            random initialization (no download).

    Returns:
        A `DinoVisionTransformer` in eval mode. Use it like stock DINOv2, for
        example `model.get_intermediate_layers(x, n=1, norm=True)` for dense
        patch features (input sides must be multiples of 14).
    """
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14", pretrained=False
    )
    for _ in range(NUM_APPENDED_BLOCKS):
        model.blocks.append(copy.deepcopy(model.blocks[-1]))
    model.n_blocks = len(model.blocks)
    if pretrained:
        if weights is None:
            state = torch.hub.load_state_dict_from_url(
                WEIGHTS_URL, map_location="cpu", weights_only=True
            )
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.eval()


def dinov3_vitl16_wsp(
    dinov3_weights,
    edit=None,
    dinov3_repo="facebookresearch/dinov3",
    dinov3_source="github",
):
    """Builds the weight-edited DINOv3 ViT-L/16.

    Only the edit is distributed here. It is applied on top of the stock
    DINOv3 ViT-L/16 (LVD-1689M) weights, which Meta provides after you accept
    the DINOv3 License.

    Args:
        dinov3_weights: Path or URL of the stock
            `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
        edit: Path to a local `dinov3_vitl16_wsp_edit.pth`. If None, the file
            is downloaded from the GitHub release.
        dinov3_repo: DINOv3 code, passed to `torch.hub.load` (a GitHub repo
            or, with `dinov3_source="local"`, a local clone).
        dinov3_source: "github" or "local".

    Returns:
        A DINOv3 `DinoVisionTransformer` in eval mode. Use it like stock
        DINOv3, for example `model.get_intermediate_layers(x, n=1, norm=True)`
        for dense patch features (input sides must be multiples of 16).
    """
    model = torch.hub.load(
        dinov3_repo,
        "dinov3_vitl16",
        source=dinov3_source,
        weights=dinov3_weights,
    )
    if edit is None:
        state = torch.hub.load_state_dict_from_url(
            DINOV3_EDIT_URL, map_location="cpu", weights_only=True
        )
    else:
        state = torch.load(edit, map_location="cpu", weights_only=True)
    return apply_dinov3_edit(model, state)


def apply_dinov3_edit(model, state):
    """Applies the edit to a stock DINOv3 ViT-L/16 in place.

    Args:
        model: Stock DINOv3 ViT-L/16 `DinoVisionTransformer`.
        state: Dict loaded from `dinov3_vitl16_wsp_edit.pth`: "rope_scale"
            (float), "rope_periods" (the stock RoPE periods divided by
            rope_scale, stored so that results do not depend on the device
            that does the division) and "block" (state dict of the appended
            block without its query/key/value weights, which are unused
            because the block's attention output projection is zero).

    Returns:
        The edited model in eval mode.
    """
    periods = model.rope_embed.periods
    new = state["rope_periods"].to(periods)
    if not torch.allclose(new * state["rope_scale"], periods, rtol=1e-5):
        raise ValueError("Stock RoPE periods differ: not DINOv3 ViT-L/16?")
    with torch.no_grad():
        periods.copy_(new)
    block = copy.deepcopy(model.blocks[-1])
    missing, unexpected = block.load_state_dict(state["block"], strict=False)
    if unexpected or any(not k.startswith("attn.qkv.") for k in missing):
        raise ValueError(
            f"Edit does not match the block: missing {missing}, "
            f"unexpected {unexpected}."
        )
    model.blocks.append(block)
    model.n_blocks = len(model.blocks)
    return model.eval()
