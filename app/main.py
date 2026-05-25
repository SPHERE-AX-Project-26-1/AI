# /fish_project/ai/app/main.py
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from ultralytics import YOLO

app = FastAPI(title="Fish YOLO Inference API")
router = APIRouter(prefix="/api")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL_PATH = ROOT_DIR/"ai"/"model"/"fish_yolo26m"/"weights"/"best.pt"
DEFAULT_MEDIA_PATH = ROOT_DIR/"backend"/"media"
DEFAULT_CLS_MODEL_PATH = ROOT_DIR/"ai"/"model"/"skygazor_cls_yolo26m"/"weights"/"best.pt"


# 환경변수가 있으면 받아오고 아니라면 위에 써둔 default path 사용
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
MEDIA_PATH = Path(os.getenv("MEDIA_ROOT", str(DEFAULT_MEDIA_PATH))).resolve()
CLS_MODEL_PATH = Path(os.getenv("CLS_MODEL_PATH", str(DEFAULT_CLS_MODEL_PATH))).resolve()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# 모델 로드
if not MODEL_PATH.exists():
    raise RuntimeError(f"YOLO 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

cls_model: Optional[YOLO] = None


def get_cls_model() -> YOLO:
    global cls_model

    if cls_model is None:
        if not CLS_MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"2차 분류 모델 파일을 찾을 수 없습니다: {CLS_MODEL_PATH}"
            )
        cls_model = YOLO(str(CLS_MODEL_PATH))

    return cls_model


# 추론 리퀘스트 형식 정의 클래스
class VideoInferRequest(BaseModel):
    # video_id: Optional[int] = Field(default=None, description="DB에 저장된 영상id값")
    video_path: str = Field(..., description="백엔드에서 전달받는 영상 저장 경로")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="신뢰도 값(이 점수 이상만 인식)")
    imgsz: int = Field(640, gt=0, description="각 프레임 처리할 때 이미지 사이즈")
    vid_stride: int = Field(1, ge=1, description="몇 프레임마다 추론할 지")
    include_frames: bool = Field(
        False,
        description="프레임별 상세 결과 포함 여부",
    )
    save_thumbnail: bool = Field(
        True,
        description="썸네일 생성 여부",
    )
    save_annotated: bool = Field(
        False,
        description="박스가 그려진 분석 결과 영상 저장 여부",
    )

# 트래킹 리퀘스트 형식 정의 클래스
class VideoTrackRequest(BaseModel):
    video_path: str = Field(..., description="백엔드에서 전달받는 영상 저장 경로")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="신뢰도 값(이 점수 이상만 인식)")
    imgsz: int = Field(640, gt=0, description="각 프레임 처리할 때 이미지 사이즈")
    vid_stride: int = Field(1, ge=1, description="몇 프레임마다 추론할 지")
    include_frames: bool = Field(False, description="프레임별 상세 결과 포함 여부")
    save_thumbnail: bool = Field(True, description="썸네일 생성 여부")
    save_annotated: bool = Field(False, description="박스가 그려진 분석 결과 영상 저장 여부")

    tracker: str = Field(
        "botsort.yaml",
        description="tracking 설정 파일(bytetrack.yaml로 바꿔도 됨)"
    )
    iou: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="tracking 시 같은 개체로 판단하는 기준치"
    )

# 2차 분류 리퀘스트 형식 정의 클래스
class VideoSpeciesTrackRequest(BaseModel):
    video_path: str = Field(..., description="백엔드에서 전달받는 영상 저장 경로")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="1차 detection/tracking 신뢰도")
    cls_conf: float = Field(0.0, ge=0.0, le=1.0, description="2차 classification 최소 신뢰도")
    imgsz: int = Field(640, gt=0, description="1차 detection/tracking 이미지 크기")
    cls_imgsz: int = Field(224, gt=0, description="2차 classification 이미지 크기")
    vid_stride: int = Field(1, ge=1, description="몇 프레임마다 추론할지")
    include_frames: bool = Field(False, description="프레임별 상세 결과 포함 여부")
    save_thumbnail: bool = Field(True, description="썸네일 생성 여부")
    tracker: str = Field("botsort.yaml", description="botsort.yaml 또는 bytetrack.yaml")
    iou: float = Field(0.7, ge=0.0, le=1.0, description="tracking IOU 기준")
    crop_margin: float = Field(0.15, ge=0.0, le=1.0, description="bbox crop 확장 비율")


# 영상 밸리데이션 함수
def validate_media_path(video_path: str) -> Path:
    try:
        p = Path(video_path).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="영상 파일이 존재하지 않습니다.")

    try:
        p.relative_to(MEDIA_PATH)
    except ValueError:
        raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다.")

    if not p.is_file():
        raise HTTPException(status_code=400, detail="파일이 아닙니다.")

    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 비디오 형식입니다.")

    return p


# 영상 meta 정보 추출 함수
def get_video_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="영상 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_sec = round(frame_count / fps, 3) if fps > 0 else None

    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def to_media_relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(MEDIA_PATH)).replace("\\", "/")

# 썸네일 생성 함수
def create_thumbnail(video_path: Path) -> Path:
    thumb_dir = ensure_dir(MEDIA_PATH/"thumbnail_imgs")
    thumb_name = f"{video_path.stem}_{uuid4().hex[:8]}.jpg"
    thumb_path = thumb_dir / thumb_name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="썸네일 생성 중 영상을 열지 못했습니다.")
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise HTTPException(status_code=400, detail="썸네일 생성 중 프레임을 읽지 못했습니다.")

    saved = cv2.imwrite(str(thumb_path), frame)
    if not saved:
        raise HTTPException(status_code=500, detail="썸네일 저장에 실패했습니다.")

    return thumb_path


# 박스 쳐져 있는 결과 영상 저장 디렉토리 생성 함수
def build_annotated_dir(video_path: Path) -> Path:
    infer_root = ensure_dir(MEDIA_PATH/"infer_result")
    run_name = f"{video_path.stem}_{uuid4().hex[:8]}"
    save_dir = infer_root / run_name
    ensure_dir(save_dir)
    return save_dir

# 바운더리 박스에 마진 추가해서 이미지 크롭 함수
def crop_bbox(image, box, margin: float):
    img_h, img_w = image.shape[:2]

    x1, y1, x2, y2 = [float(v) for v in box]
    box_w = x2 - x1
    box_h = y2 - y1

    ex1 = int(max(0, x1 - box_w * margin))
    ey1 = int(max(0, y1 - box_h * margin))
    ex2 = int(min(img_w, x2 + box_w * margin))
    ey2 = int(min(img_h, y2 + box_h * margin))

    crop = image[ey1:ey2, ex1:ex2]

    if crop.size == 0:
        return None, [ex1, ey1, ex2, ey2]

    return crop, [ex1, ey1, ex2, ey2]


# 클래스 이름 통일 함수
def normalize_species_name(name: str) -> str:
    name = str(name).strip().lower()

    if name in {"skygazor", "skygazer"}:
        return "skygazor"

    if name in {"others", "other"}:
        return "others"

    return name


# 분류 함수(2차 작업)
def classify_crop(crop, classifier: YOLO, imgsz: int, cls_conf_threshold: float) -> dict:
    result = classifier.predict(
        source=crop,
        imgsz=imgsz,
        verbose=False
    )[0]

    if result.probs is None:
        return {
            "class_id": None,
            "class_name": "unknown",
            "confidence": 0.0,
        }

    top1 = int(result.probs.top1)

    top1conf = result.probs.top1conf
    if hasattr(top1conf, "item"):
        top1conf = top1conf.item()
    top1conf = float(top1conf)

    class_name = str(classifier.names.get(top1, str(top1)))
    class_name = normalize_species_name(class_name)

    if top1conf < cls_conf_threshold:
        class_name = "unknown"

    return {
        "class_id": top1,
        "class_name": class_name,
        "confidence": round(top1conf, 4),
    }


# 메인 AI 모델 기반 추론 함수
# 아직 같은 개체인지 판별하는 건 없어서 프레임별 탐지된 fish 수가 전부 합쳐져서 나옴
def run_inference(req: VideoInferRequest) -> dict:
    video_path = validate_media_path(req.video_path)
    video_meta = get_video_meta(video_path)

    # 썸네일 생성 여부에 따라 썸네일 이미지 생성
    thumbnail_rel_path: Optional[str] = None
    if req.save_thumbnail:
        thumbnail_abs_path = create_thumbnail(video_path)
        thumbnail_rel_path = to_media_relative_path(thumbnail_abs_path)

    predict_kwargs = {
        "source": str(video_path),
        "conf": req.conf,
        "imgsz": req.imgsz,
        "vid_stride": req.vid_stride,
        "stream": True,
        "verbose": False,
        "save": False,
    }

    # 결과 영상 저장 여부에 따라 저장 경로 생성
    annotated_rel_path: Optional[str] = None
    save_dir = None
    if req.save_annotated:
        save_dir = build_annotated_dir(video_path)
        predict_kwargs.update(
            {
                "save": True,
                "project": str(save_dir.parent),
                "name": save_dir.name,
                "exist_ok": True,
            }
        )

    try:
        results = model.predict(**predict_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO 추론 중 오류가 발생했습니다: {e}")

    class_summary = defaultdict(int)
    fish_count = 0
    total_detections = 0
    num_frames_processed = 0
    frames = [] if req.include_frames else None

    fps = video_meta["fps"] if video_meta["fps"] > 0 else 0.0

    try:
        for idx, result in enumerate(results):
            num_frames_processed += 1
            detections = [] if req.include_frames else None

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                clses = result.boxes.cls.cpu().tolist()

                for box, score, cls_id in zip(boxes, confs, clses):
                    cls_id = int(cls_id)
                    class_name = str(result.names.get(cls_id, str(cls_id))).strip().lower()

                    class_summary[class_name] += 1
                    total_detections += 1

                    if class_name == "fish":
                        fish_count += 1

                    if req.include_frames:
                        detections.append(
                            {
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": round(float(score), 4),
                                "bbox_xyxy": [round(float(v), 2) for v in box],
                            }
                        )

            if req.include_frames:
                frame_index = idx * req.vid_stride
                time_sec = round(frame_index / fps, 3) if fps > 0 else None

                frames.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "detections": detections,
                    }
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 결과 후처리 중 오류가 발생했습니다: {e}")

    if req.save_annotated and save_dir is not None:
        candidate = save_dir / video_path.name
        if candidate.exists():
            annotated_rel_path = to_media_relative_path(candidate)
    
    class_summary_dict = dict(class_summary)

    return{
        "status": "success",
        # "video_id": req.video_id,
        "video_path": str(video_path),
        "count_mode": "frame_level_detection_sum",
        "video_meta": video_meta,
        "params": {
            "conf": req.conf,
            "imgsz": req.imgsz,
            "vid_stride": req.vid_stride,
            "include_frames": req.include_frames,
            "save_thumbnail": req.save_thumbnail,
            "save_annotated": req.save_annotated,
        },
        "thumbnail_path": thumbnail_rel_path,
        "annotated_video_path": annotated_rel_path,
        "fish_count": fish_count,
        "summary": {
            "num_frames_processed": num_frames_processed,
            "total_detections": total_detections,
            "class_summary": class_summary_dict,
        },
        "frames": frames,

    }

# AI 모델 기반 트래킹 함수(같은 개체인지 구별)
def run_track(req: VideoTrackRequest) -> dict:
    video_path = validate_media_path(req.video_path)
    video_meta = get_video_meta(video_path)

    if req.tracker not in {"botsort.yaml", "bytetrack.yaml"}:
        raise HTTPException(
            status_code=400,
            detail="tracker는 'botsort.yaml' 또는 'bytetrack.yaml' 이어야 합니다."
        )

    thumbnail_rel_path: Optional[str] = None
    if req.save_thumbnail:
        thumbnail_abs_path = create_thumbnail(video_path)
        thumbnail_rel_path = to_media_relative_path(thumbnail_abs_path)

    track_kwargs = {
        "source": str(video_path),
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
        "vid_stride": req.vid_stride,
        "stream": True,
        "verbose": False,
        "save": False,
        "tracker": req.tracker,
    }

    annotated_rel_path: Optional[str] = None
    save_dir = None
    if req.save_annotated:
        save_dir = build_annotated_dir(video_path)
        track_kwargs.update(
            {
                "save": True,
                "project": str(save_dir.parent),
                "name": save_dir.name,
                "exist_ok": True,
            }
        )

    try:
        results = model.track(**track_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO tracking 중 오류가 발생했습니다: {e}")

    class_summary = defaultdict(int)   # 프레임 단위 전체 검출 합
    fish_track_ids = set()             # 영상 전체 고유 fish track_id 집합
    total_detections = 0
    fish_detection_sum = 0             # 기존 infer 방식과 비교용
    tracked_detection_count = 0
    num_frames_processed = 0
    frames = [] if req.include_frames else None

    fps = video_meta["fps"] if video_meta["fps"] > 0 else 0.0

    try:
        for idx, result in enumerate(results):
            num_frames_processed += 1
            detections = [] if req.include_frames else None

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                clses = result.boxes.cls.cpu().tolist()

                if hasattr(result.boxes, "id") and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().tolist()
                else:
                    track_ids = [None] * len(boxes)

                for box, score, cls_id, track_id in zip(boxes, confs, clses, track_ids):
                    cls_id = int(cls_id)
                    class_name = str(result.names.get(cls_id, str(cls_id))).strip().lower()

                    class_summary[class_name] += 1
                    total_detections += 1

                    if class_name == "fish":
                        fish_detection_sum += 1

                    if track_id is not None:
                        track_id = int(track_id)
                        tracked_detection_count += 1

                    if class_name == "fish" and track_id is not None:
                        fish_track_ids.add(track_id)

                    if req.include_frames:
                        detections.append(
                            {
                                "track_id": track_id,
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": round(float(score), 4),
                                "bbox_xyxy": [round(float(v), 2) for v in box],
                            }
                        )

            if req.include_frames:
                frame_index = idx * req.vid_stride
                time_sec = round(frame_index / fps, 3) if fps > 0 else None
                frames.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "detections": detections,
                    }
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tracking 결과 후처리 중 오류가 발생했습니다: {e}")

    if req.save_annotated and save_dir is not None:
        candidate = save_dir / video_path.name
        if candidate.exists():
            annotated_rel_path = to_media_relative_path(candidate)

    return {
        "status": "success",
        "video_path": str(video_path),
        "count_mode": "unique_fish_track_ids",
        "video_meta": video_meta,
        "params": {
            "conf": req.conf,
            "iou": req.iou,
            "imgsz": req.imgsz,
            "vid_stride": req.vid_stride,
            "include_frames": req.include_frames,
            "save_thumbnail": req.save_thumbnail,
            "save_annotated": req.save_annotated,
            "tracker": req.tracker,
        },
        "thumbnail_path": thumbnail_rel_path,
        "annotated_video_path": annotated_rel_path,
        "fish_count": len(fish_track_ids),
        "summary": {
            "num_frames_processed": num_frames_processed,
            "total_detections": total_detections,
            "fish_detection_sum": fish_detection_sum,
            "tracked_detection_count": tracked_detection_count,
            "unique_fish_track_ids": sorted(list(fish_track_ids)),
            "class_summary": dict(class_summary),
        },
        "frames": frames,
    }


# 2차 개체 분류까지 포함한 트래킹 함수
def run_species_track(req: VideoSpeciesTrackRequest) -> dict:
    video_path = validate_media_path(req.video_path)
    video_meta = get_video_meta(video_path)
    classifier = get_cls_model()

    if req.tracker not in {"botsort.yaml", "bytetrack.yaml"}:
        raise HTTPException(
            status_code=400,
            detail="tracker는 'botsort.yaml' 또는 'bytetrack.yaml' 이어야 합니다."
        )

    thumbnail_rel_path: Optional[str] = None
    if req.save_thumbnail:
        thumbnail_abs_path = create_thumbnail(video_path)
        thumbnail_rel_path = to_media_relative_path(thumbnail_abs_path)

    track_kwargs = {
        "source": str(video_path),
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
        "vid_stride": req.vid_stride,
        "stream": True,
        "verbose": False,
        "save": False,
        "tracker": req.tracker,
    }

    try:
        results = model.track(**track_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO tracking 중 오류가 발생했습니다: {e}")

    fps = video_meta["fps"] if video_meta["fps"] > 0 else 0.0

    num_frames_processed = 0
    fish_detection_sum = 0
    tracked_detection_count = 0
    untracked_detection_count = 0

    frame_level_species_summary = defaultdict(int)
    track_records = {}
    frames = [] if req.include_frames else None

    try:
        for idx, result in enumerate(results):
            num_frames_processed += 1

            frame_index = idx * req.vid_stride
            time_sec = round(frame_index / fps, 3) if fps > 0 else None
            detections = [] if req.include_frames else None

            frame_img = result.orig_img

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                clses = result.boxes.cls.cpu().tolist()

                if hasattr(result.boxes, "id") and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().tolist()
                else:
                    track_ids = [None] * len(boxes)

                for box, det_conf, cls_id, track_id in zip(boxes, confs, clses, track_ids):
                    cls_id = int(cls_id)
                    det_class_name = str(result.names.get(cls_id, str(cls_id))).strip().lower()

                    if det_class_name != "fish":
                        continue

                    fish_detection_sum += 1

                    crop, crop_xyxy = crop_bbox(
                        image=frame_img,
                        box=box,
                        margin=req.crop_margin
                    )

                    if crop is None:
                        species_result = {
                            "class_id": None,
                            "class_name": "unknown",
                            "confidence": 0.0,
                        }
                    else:
                        species_result = classify_crop(
                            crop=crop,
                            classifier=classifier,
                            imgsz=req.cls_imgsz,
                            cls_conf_threshold=req.cls_conf,
                        )

                    species_name = species_result["class_name"]
                    frame_level_species_summary[species_name] += 1

                    if track_id is not None:
                        track_id = int(track_id)
                        tracked_detection_count += 1

                        if track_id not in track_records:
                            track_records[track_id] = {
                                "track_id": track_id,
                                "num_detections": 0,
                                "first_frame_index": frame_index,
                                "last_frame_index": frame_index,
                                "first_time_sec": time_sec,
                                "last_time_sec": time_sec,
                                "species_votes": defaultdict(float),
                                "species_detection_counts": defaultdict(int),
                            }

                        rec = track_records[track_id]
                        rec["num_detections"] += 1
                        rec["last_frame_index"] = frame_index
                        rec["last_time_sec"] = time_sec
                        rec["species_votes"][species_name] += float(species_result["confidence"])
                        rec["species_detection_counts"][species_name] += 1
                    else:
                        untracked_detection_count += 1

                    if req.include_frames:
                        detections.append(
                            {
                                "track_id": track_id,
                                "det_class_id": cls_id,
                                "det_class_name": det_class_name,
                                "det_confidence": round(float(det_conf), 4),
                                "bbox_xyxy": [round(float(v), 2) for v in box],
                                "crop_xyxy": crop_xyxy,
                                "species_class_id": species_result["class_id"],
                                "species_class_name": species_result["class_name"],
                                "species_confidence": species_result["confidence"],
                            }
                        )

            if req.include_frames:
                frames.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "detections": detections,
                    }
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"species 분석 후처리 중 오류가 발생했습니다: {e}")

    tracks = []
    track_level_species_summary = defaultdict(int)

    for track_id in sorted(track_records.keys()):
        rec = track_records[track_id]

        species_votes = dict(rec["species_votes"])
        species_detection_counts = dict(rec["species_detection_counts"])

        if len(species_votes) > 0:
            final_species = max(species_votes.items(), key=lambda x: x[1])[0]
            final_count = species_detection_counts.get(final_species, 1)
            final_confidence = species_votes[final_species] / final_count
        else:
            final_species = "unknown"
            final_confidence = 0.0

        track_level_species_summary[final_species] += 1

        tracks.append(
            {
                "track_id": track_id,
                "species": final_species,
                "species_confidence_avg": round(float(final_confidence), 4),
                "num_detections": rec["num_detections"],
                "first_frame_index": rec["first_frame_index"],
                "last_frame_index": rec["last_frame_index"],
                "first_time_sec": rec["first_time_sec"],
                "last_time_sec": rec["last_time_sec"],
                "species_votes": {
                    k: round(float(v), 4)
                    for k, v in species_votes.items()
                },
                "species_detection_counts": species_detection_counts,
            }
        )

    track_level_species_summary = dict(track_level_species_summary)
    frame_level_species_summary = dict(frame_level_species_summary)

    return {
        "status": "success",
        "video_path": str(video_path),
        "count_mode": "unique_track_ids_with_crop_classification",
        "video_meta": video_meta,
        "params": {
            "conf": req.conf,
            "cls_conf": req.cls_conf,
            "iou": req.iou,
            "imgsz": req.imgsz,
            "cls_imgsz": req.cls_imgsz,
            "vid_stride": req.vid_stride,
            "include_frames": req.include_frames,
            "save_thumbnail": req.save_thumbnail,
            "tracker": req.tracker,
            "crop_margin": req.crop_margin,
        },
        "thumbnail_path": thumbnail_rel_path,
        "fish_count": len(tracks),
        "skygazor_count": track_level_species_summary.get("skygazor", 0),
        "others_count": track_level_species_summary.get("others", 0),
        "unknown_count": track_level_species_summary.get("unknown", 0),
        "summary": {
            "num_frames_processed": num_frames_processed,
            "fish_detection_sum": fish_detection_sum,
            "tracked_detection_count": tracked_detection_count,
            "untracked_detection_count": untracked_detection_count,
            "frame_level_species_summary": frame_level_species_summary,
            "track_level_species_summary": track_level_species_summary,
        },
        "tracks": tracks,
        "frames": frames,
    }


@router.get("/ping")
def ping():
    return {"result":"pong"}

@router.get("/status")
def status_check():
    return{
        "status":"ok",
        "root_path":str(ROOT_DIR),
        "model_path":str(MODEL_PATH),
        "media_path":str(MEDIA_PATH),
        "cls_model_path":str(CLS_MODEL_PATH),
        "cls_model_exists":CLS_MODEL_PATH.exists()
    }

@router.post("/infer")
def infer(req: VideoInferRequest):
    return run_inference(req)

@router.post("/track")
def track(req: VideoTrackRequest):
    return run_track(req)

@router.post("/analyze")
def species_track(req: VideoSpeciesTrackRequest):
    return run_species_track(req)

app.include_router(router)
