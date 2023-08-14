from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model = YOLO("runs/detect/this is the one/weights/best.pt")

if __name__ == "__main__":
    results = model.train(data="./yolo_fishies/data.yaml", epochs=200, workers=4)
    results = model.val()
    print(results)
