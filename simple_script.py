from modal import Image, Stub
import os

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/cadolphs/InST.git"
    )
)

stub = Stub(name="simple_script", image=image)

@stub.function()
def check_that_inst_directory_exists():
    assert os.path.exists("/root/InST")

@stub.local_entrypoint()
def main():
    check_that_inst_directory_exists.remote()