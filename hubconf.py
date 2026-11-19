"""torch.hub entry point.

Example:
    model = torch.hub.load("peasant98/dinov2-weight-surgery", "dinov2_vitl14_wsp")
"""

dependencies = ["torch"]

from dinov2_wsp import dinov2_vitl14_wsp  # noqa: E402,F401
