from modal import Image, Stub
import os

MODEL_URL = "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
MODEL_PATH = "/root/InST/models/sd/sd-v1-4.ckpt"


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
    .apt_install("git")
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/cadolphs/InST.git",
        "cd /root/InST && mkdir -p models/sd/",
    )
    .run_function(download_sd14)
)

stub = Stub(name="simple_script", image=image)


@stub.function()
def verify_image():
    assert os.path.exists("/root/InST"), "InST not found"
    assert os.path.exists(MODEL_PATH), "Model not found"


@stub.local_entrypoint()
def main():
    verify_image.remote()