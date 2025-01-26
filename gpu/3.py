import gc
import os

import torch

from utils import (
    ITERATIONS,
    MODELS,
    get_hf_models,
    get_memory_usage,
    plot_memory_usage,
    save_csv,
)


def main():
    models = []
    metrics = []

    metrics.append(get_memory_usage("initial"))
    for iter_idx in range(ITERATIONS):
        prefix = f"[{iter_idx:2d}] "
        metrics.append(get_memory_usage(prefix + "before get models"))
        for i, model_name in enumerate(MODELS["hf"]):
            model = get_hf_models(model_name)
            metrics.append(get_memory_usage(prefix + f"after get models[{i}]"))
            model = model.to("cuda:0")
            metrics.append(get_memory_usage(prefix + f"to cuda models[{i}]"))
            models.append(model)
        model = None
        metrics.append(get_memory_usage(prefix + "after get models"))

        for i in range(len(models)):
            del models[0]
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            metrics.append(get_memory_usage(prefix + f"after del models[{i}]"))
        metrics.append(get_memory_usage(prefix + "after del all models"))

    metrics.append(get_memory_usage("final"))

    output = "figs/memory_usage_3.png"
    plot_memory_usage(metrics, output, keys=["vram"])
    os.chmod(output, 0o777)

    save_csv(metrics, "csv/memory_usage_3.csv")


if __name__ == "__main__":
    main()
