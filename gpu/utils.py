import ctypes
import os
import platform
import subprocess

import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

os.environ["HF_HOME"] = "/tmp/hf_cache"

MODELS = {
    "hf": ["bert-base-uncased", "gpt2", "tohoku-nlp/bert-base-japanese-v2"],
    "st": ["all-MiniLM-L6-v2", "intfloat/multilingual-e5-small"],
}

ITERATIONS = 50


def ctype_memory_release():
    if platform.system() == "Darwin":  # macOS
        libc = ctypes.CDLL("libc.dylib")
        libc.malloc_zone_pressure_relief(0, 0)
    elif platform.system() == "Linux":
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def cuda_device_reset():
    libcudart = ctypes.cdll.LoadLibrary("libcudart.so")

    # cudaDeviceResetシンボルを取得し、戻り値や引数型を設定
    reset_func = libcudart.cudaDeviceReset
    reset_func.restype = ctypes.c_int
    reset_func.argtypes = []

    reset_func()


def get_vram_from_torch():
    usage = torch.cuda.memory_usage()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    return {
        "reserved": reserved // 1024**2,
        "allocated": allocated // 1024**2,
        "usage": usage,
    }


def get_vram_usage_from_nvidia_smi():
    pid = os.getpid()
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            p, mem = map(int, line.split(", "))
            if p == pid:
                return mem
    return 0


def get_memory_usage(prefix: str) -> dict:
    usage_vram = get_vram_usage_from_nvidia_smi()
    res = {"vram": usage_vram}
    print(f"{prefix} vram smi: {usage_vram} MB")

    usage_torch = get_vram_from_torch()
    for k, v in usage_torch.items():
        print(f"{prefix} vram {k} torch: {v} MB")
        res[f"{k}_torch"] = v
    return res


def get_hf_models(model_name: str):
    model = AutoModel.from_pretrained(model_name)
    return model


def get_st_models(model_name: str):
    model = SentenceTransformer(model_name)
    return model


def plot_memory_usage(
    metrics: list[dict[str, float]], output: str, keys: list[str] = None
):
    if keys is None:
        keys = metrics[0].keys()
    plt.figure(figsize=(10, 6))
    for key in keys:
        plt.plot([metric[key] for metric in metrics], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Over Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_csv(metrics: list[dict[str, float]], output: str):
    with open(output, "w") as f:
        f.write("index,vram\n")
        for idx, metric in enumerate(metrics):
            f.write(f"{idx},{metric['vram']}\n")
