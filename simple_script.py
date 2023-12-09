# Starting with a clean slate

from modal import Stub, Image

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

    #    from ldm.models.diffusion.ddim import DDIMSampler

    return True
