# /fish_project/ai/app/main.py
from pathlib import Path
from collections import defaultdict
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Fish YOLO Inference API")

MODEL_PATH = "/fish_project/ai/model/fish_yolo26m/weights/best.pt"
MEDIA_ROOT = Path("/fish_project/backend/media").resolve()

# 서버 시작 시 1회 로드
model = YOLO(MODEL_PATH)


class VideoInferRequest(BaseModel):
    video_path: str
    conf: float = 0.25
    imgsz: int = 640
    vid_stride: int = 1
    save_annotated: bool = False


def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "width": 0, "height": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def validate_media_path(video_path: str) -> Path:
    p = Path(video_path).resolve()

    # media 디렉토리 밖 접근 차단
    if MEDIA_ROOT not in p.parents and p != MEDIA_ROOT:
        raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다.")

    if not p.exists():
        raise HTTPException(status_code=404, detail="영상 파일이 존재하지 않습니다.")

    if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 비디오 형식입니다.")

    return p


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/infer/by-path")
def infer_by_path(req: VideoInferRequest):
    video_path = validate_media_path(req.video_path)
    meta = get_video_meta(str(video_path))

    class_summary = defaultdict(int)
    frame_results = []

    predict_kwargs = {
        "source": str(video_path),
        "conf": req.conf,
        "imgsz": req.imgsz,
        "stream": True,      # 긴 비디오 메모리 효율 처리
        "vid_stride": req.vid_stride,
        "verbose": False,
        "save": req.save_annotated,
    }

    results = model.predict(**predict_kwargs)

    fps = meta["fps"] if meta["fps"] > 0 else 0.0

    for idx, result in enumerate(results):
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            clses = result.boxes.cls.cpu().tolist()

            for box, score, cls_id in zip(boxes, confs, clses):
                cls_id = int(cls_id)
                class_name = result.names.get(cls_id, str(cls_id))
                class_summary[class_name] += 1

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(float(score), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in box],
                })

        frame_results.append({
            "frame_index": idx * req.vid_stride,
            "time_sec": round((idx * req.vid_stride) / fps, 3) if fps > 0 else None,
            "detections": detections,
        })

    return {
        "video_path": str(video_path),
        "model_path": MODEL_PATH,
        "video_meta": meta,
        "params": {
            "conf": req.conf,
            "imgsz": req.imgsz,
            "vid_stride": req.vid_stride,
            "save_annotated": req.save_annotated,
        },
        "class_summary": dict(class_summary),
        "num_frames_returned": len(frame_results),
        "frames": frame_results,
    }
