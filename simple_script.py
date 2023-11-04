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
    Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
        "pip3 install torch torchvision numpy httpx tqdm albumentations opencv-python pudb imageio imageio-ffmpeg pytorch-lightning omegaconf test-tube streamlit setuptools pillow einops torch-fidelity transformers torchmetrics kornia git+https://github.com/cadolphs/taming-transformers.git@master#egg=taming-transformers git+https://github.com/openai/CLIP.git@main#egg=clip",
    )
    .run_commands("mkdir -p /root/models/sd")
    .run_function(download_sd14)
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/cadolphs/InST.git",
        # force_build=True,
    )
    .run_commands("pip3 install pytorch-lightning==1.6.5")
    .run_commands("cd /root/InST && pip3 install -e .")
    .run_commands("mkdir -p /root/images")
    .copy_local_file("large_blue_horses.jpg", "/root/images/large_blue_horses.jpg")
)

# Persisted volume to use for our pretrained styles:
volume = Volume.persisted("inst-style-volume")

stub = Stub(name="simple_script", image=image)
stub.volume = volume


# @stub.function()
# def verify_image():
#     assert os.path.exists("/root/InST"), "InST not found"
#     assert os.path.exists(MODEL_PATH), "Model not found"


# @stub.function(volumes={str(STYLE_MODEL_PATH): volume})
# def put_something_into_volume():
#     with open("/style_model/somefile.txt", "w") as f:
#         f.write("hello world")
#     stub.volume.commit()


# @stub.function(volumes={str(STYLE_MODEL_PATH): volume})
# def verify_wrote_into_volume():
#     stub.volume.reload()
#     assert os.path.exists("/style_model/somefile.txt"), "File not found"
#     with open("/style_model/somefile.txt", "r") as f:
#         assert f.read() == "hello world", "File contents incorrect"


# @stub.function()
# def check_import_of_inst_main():
#     import sys

#     sys.path.append("/root")
#     sys.path.append("/root/InST")
#     from InST.main import main as inst_main

#     # assert that inst_main is a function
#     assert callable(inst_main), "inst_main is not a function"


@stub.function(gpu="A10G", volumes={str(STYLE_MODEL_PATH): volume}, timeout=7200)
def run_main():
    import os

    # call main.py from InST repo
    os.system(
        f"cd /root/InST/ && python3 main.py --base configs/stable-diffusion/v1-finetune.yaml\
            -t \
            --actual_resume {MODEL_PATH}\
            -n horses \
            --log_dir {STYLE_MODEL_PATH} \
            --gpus 0, \
            --data_root /root/images \
            --no-test True\
            "
    )
    volume.commit()


@stub.function(volumes={str(STYLE_MODEL_PATH): volume})
def explore_fs():
    pass


@stub.local_entrypoint()
def main():
    # verify_image.remote()
    # put_something_into_volume.remote()
    # verify_wrote_into_volume.remote()
    # check_import_of_inst_main.remote()
    run_main.remote()
