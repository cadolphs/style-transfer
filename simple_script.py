# Starting with a clean slate

from modal import Secret, Stub, Image

ARTFUSION_GITHUB_PATH = "https://github.com/cadolphs/ArtFusion.git"
ARTFUSION_PATH = "/git/artfusion/"
MODEL_DIR = "/models"
MODEL_NAME = "artfusion_r12_step=317673.ckpt"
BASE_MODEL = "lagerbaer/artfusion"

stub = Stub("art-fusion")


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


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
        "hf-transfer~=0.1",
    )
    .apt_install("git")
    .run_commands(
        f"cd / && mkdir -p {ARTFUSION_PATH} && cd {ARTFUSION_PATH} && git clone --depth 1 {ARTFUSION_GITHUB_PATH} .",
        force_build=False,
    )
    .run_commands(f"cd / && mkdir -p {MODEL_DIR}")
    .run_function(
        download_model_to_folder, secret=Secret.from_name("my-huggingface-secret")
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

    import os
    import pathlib

    assert pathlib.Path(MODEL_DIR, MODEL_NAME).exists()
    return True
