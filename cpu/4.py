import gc
import os

import torch.nn as nn

from utils import (
    ITERATIONS,
    MODELS,
    ctype_memory_release,
    get_hf_models,
    get_memory_usage,
    plot_memory_usage,
    save_csv,
)


def clean_model(model: nn.Module):
    model.cpu()

    for param in model.parameters():
        if hasattr(param, "grad"):
            param.grad = None
        del param

    for buffer in model.buffers():
        del buffer

    for _, module in model.named_modules():
        del module

    # モデル自体を削除
    del model


def main():
    models = []
    metrics = []

    metrics.append(get_memory_usage("initial"))
    for iter_idx in range(ITERATIONS):
        prefix = f"[{iter_idx:2d}] "
        metrics.append(get_memory_usage(prefix + "before get models"))
        for i, model_name in enumerate(MODELS["hf"]):
            models.append(get_hf_models(model_name))
            metrics.append(get_memory_usage(prefix + f"after get models[{i}]"))
        metrics.append(get_memory_usage(prefix + "after get models"))

        for i in range(len(models)):
            clean_model(models[0])
            del models[0]
            gc.collect()
            ctype_memory_release()
            metrics.append(get_memory_usage(prefix + f"after del models[{i}]"))
        metrics.append(get_memory_usage(prefix + "after del all models"))

    metrics.append(get_memory_usage("final"))

    output = "figs/memory_usage_4.png"
    plot_memory_usage(metrics, output, keys=["psutil"])
    os.chmod(output, 0o777)

    save_csv(metrics, "csv/memory_usage_4.csv")


if __name__ == "__main__":
    main()
