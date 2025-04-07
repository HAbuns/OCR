from ultralytics import YOLO
import os
import shutil

model = YOLO('yolov8s.yaml').load('yolov8s.pt')
yolo_yaml_path = "/dataset/yolo_data/data.yml"

epochs = 200
imgsz = 1024
results = model.train(
    data = yolo_yaml_path,
    epochs = epochs,
    imgsz = imgsz,
    project = 'models',
    name = 'yolov8/detect/train'
)

source_dir = "/dataset/models/yolov8"

drive_save_path = "/dataset/yolo_models"

os.makedirs(drive_save_path, exist_ok=True)

shutil.copytree(source_dir, drive_save_path, dirs_exist_ok=True)