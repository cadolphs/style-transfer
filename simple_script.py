# Starting with a clean slate

from modal import Stub, Image

ARTFUSION_GITHUB_PATH = "https://github.com/cadolphs/ArtFusion.git"
ARTFUSION_PATH = "/git/artfusion/"

stub = Stub("art-fusion")

image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install(
        "opencv-python",
        "torch",
        "torchvision",
        "pytorch-lightning",
        "omegaconf",
        "einops",
        "transformers",
        "imageio",
        "imageio-ffmpeg",
        "albumentations",
    )
    .apt_install("git")
    .run_commands(
        f"cd / && mkdir -p {ARTFUSION_PATH} && cd {ARTFUSION_PATH} && git clone --depth 1 {ARTFUSION_GITHUB_PATH} .",
        force_build=False,
    )
)


@stub.function(image=image)
def test_imports():
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pytorch_lightning import seed_everything
    from PIL import ImageDraw, ImageFont, Image

    from einops import rearrange
    from omegaconf import OmegaConf
    import albumentations

    import sys

    sys.path.append(ARTFUSION_PATH)
    from main import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler

    return True
