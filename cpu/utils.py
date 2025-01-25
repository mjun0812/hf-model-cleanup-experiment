import ctypes
import os
import platform

import matplotlib.pyplot as plt
import psutil
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

os.environ["HF_HOME"] = "/tmp/hf_cache"

MODELS = {
    "hf": ["bert-base-uncased", "gpt2", "tohoku-nlp/bert-base-japanese-v2"],
    "st": ["all-MiniLM-L6-v2", "intfloat/multilingual-e5-small"],
}

ITERATIONS = 50


def get_memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # RSSから共有メモリを引いて、実際に使用している物理メモリを計算
    private_memory = (mem_info.rss - mem_info.shared) / 1024 / 1024
    return private_memory


def get_memory_usage_psutil_virtual():
    vm = psutil.virtual_memory()
    # 総メモリから利用可能なメモリとバッファ/キャッシュを引く
    actual_used = (vm.total - vm.available - vm.buffers - vm.cached) / 1024 / 1024
    return actual_used


def get_memory_usage(prefix: str) -> dict[str, float]:
    usage_psutil = get_memory_usage_psutil()
    usage_psutil_virtual = get_memory_usage_psutil_virtual()
    print(f"{prefix} psutil: {usage_psutil} MB")
    print(f"{prefix} psutil_virtual: {usage_psutil_virtual} MB")
    return {
        "psutil": usage_psutil,
        "psutil_virtual": usage_psutil_virtual,
    }


def ctype_memory_release():
    if platform.system() == "Darwin":  # macOS
        libc = ctypes.CDLL("libc.dylib")
        libc.malloc_zone_pressure_relief(0, 0)
    elif platform.system() == "Linux":
        ctypes.CDLL("libc.so.6").malloc_trim(0)


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
        f.write("index,psutil,psutil_virtual\n")
        for idx, metric in enumerate(metrics):
            f.write(f"{idx},{metric['psutil']},{metric['psutil_virtual']}\n")
