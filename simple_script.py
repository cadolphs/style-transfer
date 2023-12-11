# Starting with a clean slate

from modal import Secret, Stub, Image

ARTFUSION_GITHUB_PATH = "https://github.com/cadolphs/ArtFusion.git"
ARTFUSION_PATH = "/git/artfusion/"
MODEL_DIR = "/models"
MODEL_NAME = "artfusion_r12_step=317673.ckpt"
BASE_MODEL = "lagerbaer/artfusion"
CKPT_PATH = f"{MODEL_DIR}/{MODEL_NAME}"

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


@stub.function(image=image, gpu="any")
def test_image_generation():
    CFG_PATH = f"{ARTFUSION_PATH}/configs/kl16_content12.yaml"
    H = 256
    W = 256
    DDIM_STEPS = 250
    ETA = 1.0
    SEED = 42
    DEVICE = "cuda"

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

    seed_everything(SEED)

    config = OmegaConf.load(CFG_PATH)
    config.model.params.ckpt_path = CKPT_PATH
    config.model.params.first_stage_config.params.ckpt_path = None
    model = instantiate_from_config(config.model)
    model = model.eval().to("cuda")

    def preprocess_image(image_path, size=(W, H)):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize(size)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = rearrange(image, "h w c -> c h w")
        return torch.from_numpy(image)

    def tensor_to_rgb(x):
        return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

    def get_content_style_features(content_image_path, style_image_path, h=H, w=W):
        style_image = preprocess_image(style_image_path)[None, :].to(DEVICE)
        content_image = preprocess_image(content_image_path, size=(w, h))[None, :].to(
            DEVICE
        )

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
        content_image_path,
        style_image_path,
        h=H,
        w=W,
        content_s=1.0,
        style_s=1.0,
        ddim_steps=DDIM_STEPS,
        eta=ETA,
    ):
        c, c_null_style, c_null_content = get_content_style_features(
            content_image_path, style_image_path, h, w
        )

        with torch.no_grad(), model.ema_scope("Plotting"):
            samples = model.sample_log(
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

            x_samples = model.decode_first_stage(samples)
            x_samples = tensor_to_rgb(x_samples)

        return x_samples

    def convert_samples(samples):
        if isinstance(samples, (list, tuple)):
            samples = torch.cat(samples, dim=0)

        samples = rearrange(samples[0, :], "c h w -> h w c").cpu().numpy() * 255.0
        samples = Image.fromarray(samples.astype(np.uint8))
        return samples

    content_image_path = f"{ARTFUSION_PATH}/data/contents/lofoton.jpg"
    style_image_path = (
        f"{ARTFUSION_PATH}/data/styles/d523d66a2f745aff1d3db21be993093fc.jpg"
    )

    x_samples = style_transfer(
        content_image_path, style_image_path, content_s=0.5, style_s=2.0
    )

    x_samples = convert_samples(x_samples)

    import io

    img_bytes = io.BytesIO()
    x_samples.save(img_bytes, format="PNG")

    return img_bytes.getvalue()


@stub.local_entrypoint()
def main():
    image_bytes = test_image_generation.remote()
    output_path = "output.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)
