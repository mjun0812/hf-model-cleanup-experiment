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
    base_dir = Path(sys.argv[1])  # e.g. csv/2048m
    results_1 = load_csv(base_dir / "memory_usage_1.csv")
    results_2 = load_csv(base_dir / "memory_usage_2.csv")
    results_3 = load_csv(base_dir / "memory_usage_3.csv")
    results_4 = load_csv(base_dir / "memory_usage_4.csv")

    plt.figure(figsize=(10, 6))
    metrics = ["psutil"]
    for key in metrics:
        plt.plot([metric[key] for metric in results_1], label=f"del model.{key}")
        plt.plot([metric[key] for metric in results_2], label=f"del param.{key}")
        plt.plot(
            [metric[key] for metric in results_3], label=f"del model + ctypes.{key}"
        )
        plt.plot(
            [metric[key] for metric in results_4],
            label=f"del params + ctypes.{key}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 出力ディレクトリの作成
    os.makedirs("figs", exist_ok=True)
    output = f"figs/{base_dir.name}/memory_usage_all.png"
    plt.savefig(output)
    os.chmod(output, 0o777)
    plt.close()


if __name__ == "__main__":
    main()
