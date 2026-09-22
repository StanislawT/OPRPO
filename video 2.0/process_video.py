#!/usr/bin/env python3

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from datetime import timedelta

import cv2
import torch
from PIL import Image

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)


def frame_to_pil(frame_bgr) -> Image.Image:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split a video into frames and run each frame through OmniParser, "
                    "writing a single JSON file with the results."
    )
    p.add_argument("video", help="Path to the input video file (inside the container).")
    p.add_argument(
        "-o", "--output",
        default="/data/output/result.json",
        help="Path to the output JSON file (default: /data/output/result.json).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="How many frames per second of the source video to sample and analyze. "
             "Use 0 to process every single frame of the video (default: 1.0).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard cap on the number of sampled frames to process (0 = no limit). "
             "Useful for quick tests on long videos.",
    )
    p.add_argument("--box-threshold", type=float, default=0.05, help="YOLO icon-detection confidence threshold.")
    p.add_argument("--iou-threshold", type=float, default=0.7, help="IOU threshold used for box de-duplication.")
    p.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also save the annotated (boxes-drawn) PNG for every processed frame "
             "next to the output JSON, under an 'annotated_frames' subfolder.",
    )
    p.add_argument(
        "--som-model-path",
        default="/app/OmniParserLocal/weights/icon_detect/model.pt",
        help="Path to the YOLO icon-detection weights.",
    )
    p.add_argument(
        "--caption-model-path",
        default="/app/OmniParserLocal/weights/icon_caption_florence",
        help="Path to the Florence-2 icon-captioning weights folder.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device. 'auto' uses CUDA if available, otherwise CPU (default: auto).",
    )
    p.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip loading/running the Florence-2 icon-captioning model. Detected icons will "
             "have content=null instead of a text description. MUCH faster on CPU-only machines "
             "(no GPU), since the captioning step is the slowest part of the pipeline there.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size in pixels. Lower values (e.g. 320-480) are noticeably "
             "faster on CPU at some cost to small-icon detection accuracy (default: 640).",
    )
    p.add_argument(
        "--use-paddleocr",
        action="store_true",
        help="Use PaddleOCR instead of EasyOCR for text detection (default: EasyOCR).",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: --device cuda requested but no CUDA GPU is visible; falling back to CPU.", file=sys.stderr)
            device = "cpu"

    print(f"Using device: {device}", flush=True)
    if device == "cpu":
        print(
            "NOTE: running on CPU. This will be considerably slower than on a GPU. "
            "Consider a low --fps, --imgsz 320-480 and/or --no-captions to speed things up.",
            flush=True,
        )

    print("Loading YOLO icon-detection model...", flush=True)
    t_load = time.time()
    yolo_model = get_yolo_model(model_path=args.som_model_path)

    caption_model_processor = None
    if not args.no_captions:
        print("Loading Florence-2 icon-captioning model...", flush=True)
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", model_name_or_path=args.caption_model_path, device=device
        )
    else:
        print("Skipping icon-captioning model (--no-captions): icons will have content=null.", flush=True)
    print(f"Models loaded in {time.time() - t_load:.1f}s", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video with OpenCV: {args.video}", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.fps and args.fps > 0:
        frame_interval = max(int(round(src_fps / args.fps)), 1)
    else:
        frame_interval = 1

    print(
        f"Source video: fps={src_fps:.2f}, total_frames={total_frames}, "
        f"sampling every {frame_interval} frame(s) (~{src_fps / frame_interval:.2f} fps analyzed)",
        flush=True,
    )

    annotated_dir = None
    if args.save_annotated:
        annotated_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", "annotated_frames")
        os.makedirs(annotated_dir, exist_ok=True)

    results = []
    frame_idx = 0
    processed_idx = 0
    run_started = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / src_fps
            pil_img = frame_to_pil(frame)

            t0 = time.time()
            error = None
            annotated_b64 = None
            parsed_content_list = []
            try:
                (ocr_text, ocr_bbox), _ = check_ocr_box(
                    pil_img,
                    display_img=False,
                    output_bb_format="xyxy",
                    easyocr_args={"paragraph": False, "text_threshold": 0.9},
                    use_paddleocr=args.use_paddleocr,
                )
                ocr_text = ocr_text or []
                ocr_bbox = ocr_bbox or []
                annotated_b64, _, parsed_content_list = get_som_labeled_img(
                    pil_img,
                    yolo_model,
                    BOX_TRESHOLD=args.box_threshold,
                    output_coord_in_ratio=True,
                    ocr_bbox=ocr_bbox,
                    caption_model_processor=caption_model_processor,
                    ocr_text=ocr_text,
                    use_local_semantics=not args.no_captions,
                    iou_threshold=args.iou_threshold,
                    imgsz=args.imgsz,
                    batch_size=32,
                )
            except Exception:
                error = traceback.format_exc(limit=3)
            elapsed = time.time() - t0

            entry = {
                "frame_index": frame_idx,
                "processed_index": processed_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "timestamp": str(timedelta(seconds=timestamp_sec)),
                "processing_time_sec": round(elapsed, 3),
                "elements": parsed_content_list,
                "element_count": len(parsed_content_list),
            }
            if error:
                entry["error"] = error
            results.append(entry)

            if annotated_dir and annotated_b64:
                out_path = os.path.join(annotated_dir, f"frame_{frame_idx:08d}.png")
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(annotated_b64))

            print(
                f"[{processed_idx + 1}] frame#{frame_idx} @ {timestamp_sec:.2f}s -> "
                f"{len(parsed_content_list)} elements ({elapsed:.2f}s)"
                + (f" ERROR: {error.splitlines()[-1]}" if error else ""),
                flush=True,
            )

            processed_idx += 1
            if args.max_frames and processed_idx >= args.max_frames:
                break

        frame_idx += 1

    cap.release()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    output_doc = {
        "video_file": os.path.basename(args.video),
        "source_fps": src_fps,
        "requested_sample_fps": args.fps,
        "total_source_frames": total_frames,
        "frames_processed": processed_idx,
        "box_threshold": args.box_threshold,
        "iou_threshold": args.iou_threshold,
        "total_processing_time_sec": round(time.time() - run_started, 2),
        "frames": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, ensure_ascii=False, indent=2, default=str)

    print(f"Done. Wrote {processed_idx} frame result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()