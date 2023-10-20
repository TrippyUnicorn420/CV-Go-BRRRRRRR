from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

if __name__ == "__main__":
    results = model.train(
        data="./yolo_fishies/data.yaml", epochs=100, workers=8, patience=5, imgsz=360
    )
    results = model.val()
    print(results)
