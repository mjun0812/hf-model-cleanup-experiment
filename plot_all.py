import csv

import matplotlib.pyplot as plt


def load_csv(file: str):
    results = []
    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def main():
    results_1 = load_csv("csv/memory_usage_1.csv")
    results_2 = load_csv("csv/memory_usage_2.csv")
    results_3 = load_csv("csv/memory_usage_3.csv")
    results_4 = load_csv("csv/memory_usage_4.csv")

    plt.figure(figsize=(10, 6))
    for key in ["psutil"]:
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
    plt.tight_layout()
    plt.savefig("figs/memory_usage_all.png")
    plt.close()


if __name__ == "__main__":
    main()
