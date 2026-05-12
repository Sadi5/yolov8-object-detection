"""YOLOv8 object detection entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def detect_image(model_path: str, image_path: str, conf: float = 0.5):
    """Detect objects in a single image and save annotated output."""
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        project="results",
        name="image_detection",
        exist_ok=True,
    )
    print("Detection complete. Results saved in results/image_detection/")
    return results


def detect_video(model_path: str, video_path: str, conf: float = 0.5):
    """Detect objects in a video file and save annotated output."""
    model = YOLO(model_path)
    results = model.predict(
        source=video_path,
        conf=conf,
        save=True,
        project="results",
        name="video_detection",
        exist_ok=True,
    )
    print("Video detection complete. Results saved in results/video_detection/")
    return results


def detect_webcam(model_path: str, conf: float = 0.5):
    """Run real-time detection using the default webcam."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    print("Press 'q' to quit webcam detection.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Detection - Press Q to Quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def validate_source(mode: str, source: str):
    """Ensure image/video modes receive a source path."""
    if mode in {"image", "video"} and not source:
        raise ValueError(f"--source is required when --mode is '{mode}'.")

    if source and not Path(source).exists():
        raise FileNotFoundError(f"Source file not found: {source}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "--mode",
        choices=["image", "video", "webcam"],
        default="webcam",
        help="Detection mode to run.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to a YOLOv8 model file or model name.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Input image or video path for non-webcam modes.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_source(args.mode, args.source)

    if args.mode == "webcam":
        detect_webcam(args.model, args.conf)
    elif args.mode == "image":
        detect_image(args.model, args.source, args.conf)
    else:
        detect_video(args.model, args.source, args.conf)
