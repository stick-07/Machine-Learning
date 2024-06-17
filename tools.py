from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load an official model
results = model.predict("0.jpg")
print(results.names[int(box.cls[0])])