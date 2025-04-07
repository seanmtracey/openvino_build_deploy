import argparse
import json
import logging as log
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
from openvino import runtime as ov
from ultralytics import YOLO
from ultralytics.utils import ops

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils  # assumes you have a `demo_utils.py` from the original project

CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

RESULTS = []

def convert(model_name: str, model_dir: Path) -> tuple[Path, Path]:
    model_path = model_dir / f"{model_name}.pt"
    yolo_model = YOLO(model_path)
    ov_model_path = model_dir / f"{model_name}_openvino_model"
    ov_int8_model_path = model_dir / f"{model_name}_int8_openvino_model"

    if not ov_model_path.exists():
        ov_model_path = yolo_model.export(format="openvino", dynamic=False, half=True)
    if not ov_int8_model_path.exists():
        ov_int8_model_path = yolo_model.export(format="openvino", dynamic=False, half=True, int8=True, data="coco128.yaml")

    return Path(ov_model_path) / f"{model_name}.xml", Path(ov_int8_model_path) / f"{model_name}.xml"


def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    shape = img.shape[1::-1]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape != new_unpad:
        img = cv2.resize(img, dsize=new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img, ratio, (int(dw), int(dh))


def preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    image, _, padding = letterbox(image, new_shape=input_size)
    image = image.astype(np.float32) / 255.0
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, padding


def postprocess(pred_boxes: np.ndarray, pred_masks: np.ndarray, input_size: Tuple[int, int], orig_img: np.ndarray,
                padding: Tuple[int, int], category_id: int, min_conf_threshold: float = 0.25,
                nms_iou_threshold: float = 0.75, agnostic_nms: bool = False, max_detections: int = 100):
    nms_kwargs = {"agnostic": agnostic_nms, "max_det": max_detections}
    pred = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)[0]
    if not len(pred):
        return []

    masks = None
    if pred_masks is not None:
        masks = np.array(ops.process_mask(torch.from_numpy(pred_masks[0]), pred[:, 6:], pred[:, :4], input_size, upsample=True))
        masks = np.array([cv2.resize(mask[padding[1]:-padding[1] - 1, padding[0]:-padding[0] - 1], orig_img.shape[:2][::-1], interpolation=cv2.INTER_AREA) for mask in masks])
        masks = masks.astype(np.bool_)

    pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], orig_img.shape).round()
    pred = np.array(pred)
    detections = []

    # for i in range(len(pred)):
    #     detections.append({
    #         "bbox": [int(x) for x in pred[i, :4]],
    #         "confidence": float(pred[i, 4]),
    #         "class_id": int(pred[i, 5])
    #     })

    frame_height, frame_width = orig_img.shape[:2]

    for i in range(len(pred)):
        x1, y1, x2, y2 = pred[i, :4]
        detections.append({
            "bbox": [
                round(float(x1) / frame_width, 6),
                round(float(y1) / frame_height, 6),
                round(float(x2) / frame_width, 6),
                round(float(y2) / frame_height, 6)
            ],
            "confidence": float(pred[i, 4]),
            "class_id": int(pred[i, 5])
        })

    return detections


def get_model(model_path: Path, device: str = "AUTO") -> ov.CompiledModel:
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device_name=device, config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "cache"})
    return compiled_model


def run_headless(
    video_path: str,
    model_paths: Tuple[Path, Path],
    model_name: str = "",
    category: str = "person",
    flip: bool = True,
    output_path: str = "detections_log.json"
) -> None:
    log.getLogger().setLevel(log.INFO)
    model_mapping = {"FP16": model_paths[0], "INT8": model_paths[1]}
    model_type = "INT8"
    device_type = "AUTO"

    core = ov.Core()
    core.set_property({"CACHE_DIR": "cache"})
    model = get_model(model_mapping[model_type], device_type)
    input_shape = tuple(model.inputs[0].shape)[:0:-1]

    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)

    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60, flip=flip)
    category_id = CATEGORIES.index(category)

    processing_times = deque(maxlen=100)
    frame_number = 0

    player.start()
    with open(output_path, "w") as output_file:
        while True:
            frame = player.next()
            if frame is None:
                break

            frame = np.array(frame)
            input_image, padding = preprocess(image=frame, input_size=input_shape[:2])
            start_time = time.time()
            results = model(input_image)
            processing_times.append(time.time() - start_time)

            boxes = results[model.outputs[0]]
            masks = results[model.outputs[1]] if len(model.outputs) > 1 else None

            detections = postprocess(
                pred_boxes=boxes,
                pred_masks=masks,
                input_size=input_shape[:2],
                orig_img=frame,
                padding=padding,
                category_id=category_id
            )

            # timestamp = time.time()
            # timestamp = frame_number / player.fps
            # timestamp = player.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp = player._VideoPlayer__cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


            for det in detections:
                entry = {
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "class": CATEGORIES[det["class_id"]],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"]
                }
                # output_file.write(json.dumps(entry) + "\n")
                RESULTS.append(entry)

            frame_number += 1

    player.stop()
    print(">>>>>>>>>>")
    print(json.dumps(RESULTS))
    with open(output_path, "w") as output_file:
        output_file.write(json.dumps(RESULTS))



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--stream', default="0", type=str, help="Path to video file or webcam index")
    # parser.add_argument('--model_name', type=str, default="yolo11n", help="YOLO model version")
    # parser.add_argument('--model_dir', type=str, default="model", help="Directory to store/load model files")
    # parser.add_argument('--category', type=str, default="person", choices=CATEGORIES, help="Category to detect")
    # parser.add_argument('--flip', type=bool, default=True, help="Mirror the input stream")
    # parser.add_argument('--output', type=str, default="detections_log.json", help="Output JSON file")

    stream = os.environ.get("STREAM", "0")
    model_name = os.environ.get("MODEL_NAME", "yolo11n")
    model_dir = Path(os.environ.get("MODEL_DIR", "model"))
    category = os.environ.get("CATEGORY", "person")
    flip = os.environ.get("FLIP", "true").lower() == "true"
    output = os.environ.get("OUTPUT", "detections_log.json")


    # args = parser.parse_args()
    # model_paths = convert(args.model_name, Path(args.model_dir))
    # run_headless(
    #     video_path=args.stream,
    #     model_paths=model_paths,
    #     model_name=args.model_name,
    #     category=args.category,
    #     flip=args.flip,
    #     output_path=args.output
    # )

    model_paths = convert(model_name, model_dir)
    run_headless(
        video_path=stream,
        model_paths=model_paths,
        model_name=model_name,
        category=category,
        flip=flip,
        output_path=output
    )
