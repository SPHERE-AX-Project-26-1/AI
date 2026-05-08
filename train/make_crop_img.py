from pathlib import Path
import cv2
from ultralytics import YOLO

DETECT_MODEL_PATH = "/fish_project/ai/model/fish_yolo26m/weights/best.pt"

DATA_ROOT = Path("/fish_project/data/dataset1")
IMAGE_ROOT = DATA_ROOT / "images"
CROP_ROOT = DATA_ROOT / "crops"

CONF = 0.25
IMGSZ = 640

MARGIN = 0.15

FISH_CLASS_ID = 0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def expand_box(x1, y1, x2, y2, img_w, img_h, margin=0.15):

    box_w = x2 - x1
    box_h = y2 - y1

    x1_new = int(max(0, x1 - box_w * margin))
    y1_new = int(max(0, y1 - box_h * margin))
    x2_new = int(min(img_w, x2 + box_w * margin))
    y2_new = int(min(img_h, y2 + box_h * margin))

    return x1_new, y1_new, x2_new, y2_new


def crop_split(model, split: str):

    image_dir = IMAGE_ROOT / split
    out_dir = CROP_ROOT / f"raw_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ]

    print(f"[INFO] split={split}, images={len(image_paths)}")
    print(f"[INFO] output dir={out_dir}")

    crop_count = 0

    for img_path in image_paths:
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        results = model.predict(
            source=str(img_path),
            conf=CONF,
            imgsz=IMGSZ,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            for box_idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                if FISH_CLASS_ID is not None and cls_id != FISH_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = expand_box(
                    x1, y1, x2, y2,
                    img_w=img_w,
                    img_h=img_h,
                    margin=MARGIN
                )

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                out_name = (
                    f"{img_path.stem}"
                    f"_crop{box_idx:03d}"
                    f"_conf{score:.2f}.jpg"
                )
                out_path = out_dir / out_name

                cv2.imwrite(str(out_path), crop)
                crop_count += 1

    print(f"[DONE] split={split}, saved crops={crop_count}")


def main():
    model = YOLO(DETECT_MODEL_PATH)

    crop_split(model, "train")
    crop_split(model, "val")


if __name__ == "__main__":
    main()
