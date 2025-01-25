import os
import subprocess

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

os.environ["HF_HOME"] = "/tmp/hf_cache"

MODELS = {
    "hf": ["bert-base-uncased", "gpt2", "tohoku-nlp/bert-base-japanese-v2"],
    "st": ["all-MiniLM-L6-v2", "intfloat/multilingual-e5-small"],
}

ITERATIONS = 50


def get_vram_usage():
    pid = os.getpgid()
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
            p, mem = map(int, line.split(","))
            if p == pid:
                return mem
    return 0


def get_memory_usage(prefix: str) -> dict[str, float]:
    usage_vram = get_vram_usage()
    print(f"{prefix} vram: {usage_vram} MB")
    return {"vram": usage_vram}


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
