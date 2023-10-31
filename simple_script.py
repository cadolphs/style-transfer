from modal import Image, Stub, Volume
from pathlib import Path
import os

MODEL_URL = "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
MODEL_PATH = "/root/models/sd/sd-v1-4.ckpt"
STYLE_MODEL_PATH = Path("/style_model")


def download_sd14():
    import httpx
    from tqdm import tqdm

    with open(MODEL_PATH, "wb") as download_file:
        with httpx.stream("GET", MODEL_URL, follow_redirects=True) as response:
            total = int(response.headers["Content-Length"])
            with tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(
                        response.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = response.num_bytes_downloaded


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("httpx", "tqdm")
    .run_commands("mkdir -p /root/models/sd")
    .run_function(download_sd14)
    .apt_install("git")
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/cadolphs/InST.git",
    )
)

# Persisted volume to use for our pretrained styles:
volume = Volume.persisted("inst-style-volume")

stub = Stub(name="simple_script", image=image)
stub.volume = volume


@stub.function()
def verify_image():
    assert os.path.exists("/root/InST"), "InST not found"
    assert os.path.exists(MODEL_PATH), "Model not found"


@stub.function(volumes={str(STYLE_MODEL_PATH): volume})
def put_something_into_volume():
    with open("/style_model/somefile.txt", "w") as f:
        f.write("hello world")
    stub.volume.commit()


@stub.function(volumes={str(STYLE_MODEL_PATH): volume})
def verify_wrote_into_volume():
    stub.volume.reload()
    assert os.path.exists("/style_model/somefile.txt"), "File not found"
    with open("/style_model/somefile.txt", "r") as f:
        assert f.read() == "hello world", "File contents incorrect"


@stub.local_entrypoint()
def main():
    verify_image.remote()
    put_something_into_volume.remote()
    verify_wrote_into_volume.remote()
