"""Training script for YOLOv8 custom datasets."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def train_yolov8(
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    model_name: str = "yolov8n.pt",
):
    """Train a YOLOv8 model on a dataset config."""
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yolov8_custom",
        project="runs/train",
        patience=10,
        save=True,
        plots=True,
        exist_ok=True,
    )

    print("Training complete. Best weights are saved under runs/train/yolov8_custom/weights/")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset.")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_yolov8(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
    )
