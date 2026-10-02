import modal

app = modal.App("test-check")
image = modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")

@app.function(gpu="T4", image=image)
def check():
    import torch
    import subprocess
    nvcc_ver = subprocess.check_output(["nvcc", "--version"]).decode()
    return f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}\n{nvcc_ver}"

@app.local_entrypoint()
def main():
    print(check.remote())
