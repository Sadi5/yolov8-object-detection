"""Model evaluation helpers for YOLOv8."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def evaluate_model(model_path: str, data_yaml: str):
    """Evaluate a YOLOv8 model and print common detection metrics."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print("\n===== EVALUATION RESULTS =====")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print("==============================")
    return metrics


def compare_models(yolov8_path: str, baseline_path: str, data_yaml: str):
    """Compare two YOLO-format compatible models and save summary results."""
    yolo_metrics = evaluate_model(yolov8_path, data_yaml)
    baseline_metrics = evaluate_model(baseline_path, data_yaml)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    dataframe = pd.DataFrame(
        {
            "Model": ["YOLOv8", "Baseline"],
            "mAP@0.5": [yolo_metrics.box.map50, baseline_metrics.box.map50],
            "mAP@0.5:0.95": [yolo_metrics.box.map, baseline_metrics.box.map],
            "Precision": [yolo_metrics.box.mp, baseline_metrics.box.mp],
            "Recall": [yolo_metrics.box.mr, baseline_metrics.box.mr],
        }
    )
    output_path = results_dir / "comparison.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")
    print(dataframe)
    return dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 models.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model to evaluate.")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset YAML file.")
    parser.add_argument(
        "--baseline",
        default="",
        help="Optional second model path for side-by-side comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.baseline:
        compare_models(args.model, args.baseline, args.data)
    else:
        evaluate_model(args.model, args.data)
