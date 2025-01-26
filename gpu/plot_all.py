import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(file: str):
    results = []
    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 文字列を浮動小数点数に変換
            results.append({k: float(v) for k, v in row.items()})
    return results


def main():
    # CSVファイルのパス
    base_dir = Path(sys.argv[1])
    results_1 = load_csv(base_dir / "memory_usage_1.csv")
    results_2 = load_csv(base_dir / "memory_usage_2.csv")
    results_3 = load_csv(base_dir / "memory_usage_3.csv")
    results_4 = load_csv(base_dir / "memory_usage_4.csv")
    results_5 = load_csv(base_dir / "memory_usage_5.csv")
    results_6 = load_csv(base_dir / "memory_usage_6.csv")

    plt.figure(figsize=(10, 6))
    metrics = ["vram"]
    for key in metrics:
        plt.plot([metric[key] for metric in results_1], label=f"del model.{key}")
        plt.plot(
            [metric[key] for metric in results_2], label=f"del model+empty_cache.{key}"
        )
        plt.plot([metric[key] for metric in results_3], label=f"ipc_collect.{key}")
        plt.plot([metric[key] for metric in results_4], label=f"cublas.{key}")
        plt.plot([metric[key] for metric in results_5], label=f"cufft.{key}")
        plt.plot([metric[key] for metric in results_6], label=f"to('cpu').{key}")

    plt.xlabel("Iteration")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 出力ディレクトリの作成
    os.makedirs("figs", exist_ok=True)
    output = "figs/memory_usage_all.png"
    plt.savefig(output)
    os.chmod(output, 0o777)
    plt.close()


if __name__ == "__main__":
    main()
