"""Loader for the weight-edited DINOv2 ViT-L/14.

The edited model is a standard DINOv2 ViT-L/14 with one extra transformer block
appended (25 blocks instead of 24). Every weight was written in closed form;
nothing was trained.
"""

import copy

import torch

WEIGHTS_URL = (
    "https://github.com/peasant98/dinov2-weight-surgery/releases/download/"
    "v1.0/dinov2_vitl14_wsp.pth"
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
