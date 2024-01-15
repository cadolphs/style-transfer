# Starting with a clean slate

from modal import Secret, Stub, Image, gpu, method, web_endpoint

ARTFUSION_GITHUB_PATH = "https://github.com/cadolphs/ArtFusion.git"
ARTFUSION_PATH = "/git/artfusion/"
MODEL_DIR = "/models"
EXAMPLE_STYLE_DIR = "/example_styles"
MODEL_NAME = "artfusion_r12_step=317673.ckpt"
BASE_MODEL = "lagerbaer/artfusion"
CKPT_PATH = f"{MODEL_DIR}/{MODEL_NAME}"
CFG_PATH = f"{ARTFUSION_PATH}/configs/kl16_content12.yaml"
DEVICE = "cuda"

stub = Stub("art-fusion")

from fastapi import Form
from pydantic import BaseModel


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


def instantiate_model():
    """Useful to pre-load the required weights"""
    import sys

    sys.path.append(ARTFUSION_PATH)
    from omegaconf import OmegaConf
    from main import instantiate_from_config

    config = OmegaConf.load(CFG_PATH)
    config.model.params.ckpt_path = CKPT_PATH
    config.model.params.first_stage_config.params.ckpt_path = None
    model = instantiate_from_config(config.model)


image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install(
        "torch",
        "torchvision",
        "pytorch-lightning",
        "omegaconf",
        "einops",
        "transformers",
        "imageio",
        "imageio-ffmpeg",
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
    .run_function(instantiate_model)
    .run_commands(f"cd / && mkdir -p {EXAMPLE_STYLE_DIR}")
    .copy_local_file("scream.jpg", f"{EXAMPLE_STYLE_DIR}/scream.jpg")
)


def preprocess_image(image, size=(256, 256), max_size=256):
    import numpy as np
    import torch
    from einops import rearrange

    if not image.mode == "RGB":
        image = image.convert("RGB")
    im_max_size = max(image.size)
    max_size = min(max_size, im_max_size)

    image.thumbnail((max_size, max_size))
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = rearrange(image, "h w c -> c h w")
    return torch.from_numpy(image)[None, :].to(DEVICE)


def tensor_to_rgb(x):
    import torch

    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)


def convert_samples(samples):
    from einops import rearrange
    from PIL import Image
    import numpy as np

    if isinstance(samples, (list, tuple)):
        samples = torch.cat(samples, dim=0)

    samples = rearrange(samples[0, :], "c h w -> h w c").cpu().numpy() * 255.0
    samples = Image.fromarray(samples.astype(np.uint8))
    return samples


@stub.cls(image=image, gpu=gpu.T4(), container_idle_timeout=180)
class StyleTransfer:
    def __enter__(self):
        from omegaconf import OmegaConf
        from pytorch_lightning import seed_everything
        import sys

        sys.path.append(ARTFUSION_PATH)
        from main import instantiate_from_config

        seed_everything(42)

        config = OmegaConf.load(CFG_PATH)
        config.model.params.ckpt_path = CKPT_PATH
        config.model.params.first_stage_config.params.ckpt_path = None
        model = instantiate_from_config(config.model)

        self.model = model.eval().to(DEVICE)

    def get_content_style_features(
        self, content_image, style_image, max_size=256, style_size=256
    ):
        import torch

        model = self.model
        style_image = preprocess_image(style_image, max_size=style_size)
        content_image = preprocess_image(content_image, max_size=max_size)

        with torch.no_grad(), model.ema_scope("Plotting"):
            vgg_features = model.vgg(model.vgg_scaling_layer(style_image))
            c_style = model.get_style_features(vgg_features)
            null_style = c_style.clone()
            null_style[:] = model.null_style_vector.weight[0]

            content_encoder_posterior = model.encode_first_stage(content_image)
            content_encoder_posterior = model.get_first_stage_encoding(
                content_encoder_posterior
            )
            c_content = model.get_content_features(content_encoder_posterior)
            null_content = torch.zeros_like(c_content)

        c = {"c1": c_content, "c2": c_style}
        c_null_style = {"c1": c_content, "c2": null_style}
        c_null_content = {"c1": null_content, "c2": c_style}

        return c, c_null_style, c_null_content

    def style_transfer(
        self,
        content_image,
        style_image,
        content_s,
        style_s,
        ddim_steps,
        eta,
        max_size=None,
        style_size=None,
    ):
        import torch

        c, c_null_style, c_null_content = self.get_content_style_features(
            content_image, style_image, max_size=max_size, style_size=style_size
        )

        with torch.no_grad(), self.model.ema_scope("Plotting"):
            samples = self.model.sample_log(
                cond=c,
                batch_size=1,
                x_T=torch.rand_like(c["c1"]),
                ddim=True,
                ddim_steps=ddim_steps,
                eta=eta,
                unconditional_guidance_scale=content_s,
                unconditional_conditioning=c_null_content,
                unconditional_guidance_scale_2=style_s,
                unconditional_conditioning_2=c_null_style,
            )[0]

            x_samples = self.model.decode_first_stage(samples)
            x_samples = tensor_to_rgb(x_samples)

        return x_samples

    @method()
    def generate(
        self,
        content_bytes=None,
        style_bytes=None,
        content_strength=0.5,
        style_strength=1.0,
        max_size=1024,
        style_size=256,
        ddim_steps=10,
        eta=0,
    ):
        from PIL import Image

        import io

        content_image = Image.open(io.BytesIO(content_bytes))
        style_image = Image.open(io.BytesIO(style_bytes))

        x_samples = self.style_transfer(
            content_image,
            style_image,
            content_s=content_strength,
            style_s=style_strength,
            max_size=max_size,
            style_size=style_size,
            ddim_steps=ddim_steps,
            eta=eta,
        )

        x_samples = convert_samples(x_samples)

        img_bytes = io.BytesIO()
        x_samples.save(img_bytes, format="PNG")

        return img_bytes.getvalue()


@stub.local_entrypoint()
def main(
    content_file_name: str,
    style_file_name: str,
    content_strength: float = 0.5,
    style_strength: float = 1.0,
    max_size: int = 512,
    style_size: int = 256,
    eta: float = 0,
    ddim_steps: int = 10,
):
    import io

    with open(content_file_name, "rb") as f:
        content_bytes = io.BytesIO(f.read()).getvalue()
    with open(style_file_name, "rb") as f:
        style_bytes = io.BytesIO(f.read()).getvalue()

    image_bytes = StyleTransfer().generate.remote(
        content_bytes,
        style_bytes,
        content_strength=content_strength,
        style_strength=style_strength,
        max_size=max_size,
        style_size=style_size,
        eta=eta,
        ddim_steps=ddim_steps,
    )
    output_path = "output.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)


class Base64Image(BaseModel):
    image_base64: str


@stub.function(image=image, container_idle_timeout=180)
@web_endpoint(method="POST")
def screamify(image: Base64Image):
    from io import BytesIO
    import base64
    import re

    image_content = image.image_base64

    if "," in image_content:
        image_content = image_content.split(",")[1]

    print("Image content [truncated]: ")
    print(image_content[:200])

    content_bytes = BytesIO(base64.b64decode(image_content)).getvalue()

    with open(f"{EXAMPLE_STYLE_DIR}/scream.jpg", "rb") as f:
        style_bytes = BytesIO(f.read()).getvalue()

    print("Running style transfer")
    image_bytes = StyleTransfer().generate.remote(
        content_bytes,
        style_bytes,
        content_strength=1,
        style_strength=3,
        max_size=1024,
        style_size=256,
        eta=0,
        ddim_steps=10,
    )
    print("Done")

    img_str = base64.b64encode(image_bytes).decode("utf-8")
    img_str = "data:image/png;base64," + img_str

    return {"image_base64": img_str}
