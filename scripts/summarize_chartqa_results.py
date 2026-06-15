import json
from pathlib import Path


PATHS = [
    ("base_val", "outputs/chartqa/metrics/base_val_full_metrics.json"),
    ("lora1k_val", "outputs/chartqa/metrics/lora1k_val_full_metrics.json"),
    ("lora5k_val", "outputs/chartqa/metrics/lora5k_val_full_metrics.json"),
    ("base_test", "outputs/chartqa/metrics/base_test_full_metrics.json"),
    ("lora5k_test", "outputs/chartqa/metrics/lora5k_test_full_metrics.json"),
]


def main():
    print("| Run | N | Exact Match | Relaxed Acc | Numeric Relaxed Acc | Nonempty |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, path_str in PATHS:
        path = Path(path_str)
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        numeric = metrics["numeric_relaxed_accuracy"]
        print(
            f"| {name} | {metrics['num_samples']} | "
            f"{metrics['exact_match']:.4f} | "
            f"{metrics['relaxed_accuracy']:.4f} | "
            f"{(numeric if numeric is not None else 0):.4f} | "
            f"{metrics['nonempty_rate']:.4f} |"
        )


if __name__ == "__main__":
    main()
