from pathlib import Path
from ultralytics import YOLO


def main():
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent.parent
    data_dir = root_dir / "data" / "skygazer_cls"
    project_dir = base_dir.parent / "model"

    model = YOLO("yolo26m-cls.pt")

    results = model.train(
        data=str(data_dir),
        epochs=100,
        imgsz=224,
        batch=16,
        device=0,
        save=True,
        pretrained=True,
        exist_ok=True,
        val=True,
        project=str(project_dir),
        name="skygazer_cls_yolo26m",
        patience=15,
        plots=True,
    )

    print(results)


if __name__ == "__main__":
    main()
