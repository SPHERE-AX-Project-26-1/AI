from pathlib import Path
from ultralytics import YOLO


def main():
    base_dir = Path(__file__).resolve().parent
    data_yaml = base_dir / "data.yaml"
    project_dir = base_dir.parent / "model"   

    model = YOLO("yolo26m.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=-1,
        device=0,
        workers=8,
        amp=True,
        patience=30,
        pretrained=True,
        val=True,
        plots=True,
        project=str(project_dir),
        name="fish_yolo26m",
        exist_ok=True,
    )

    print(results)


if __name__ == "__main__":
    main()
