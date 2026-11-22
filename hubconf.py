"""torch.hub entry point.

Examples:
    model = torch.hub.load("peasant98/dinov2-weight-surgery", "dinov2_vitl14_wsp")
    model = torch.hub.load("peasant98/dinov2-weight-surgery", "dinov3_vitl16_wsp",
                           dinov3_weights="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
"""

dependencies = ["torch"]

from dinov2_wsp import dinov2_vitl14_wsp, dinov3_vitl16_wsp  # noqa: E402,F401
