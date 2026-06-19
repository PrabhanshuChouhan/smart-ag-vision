from ultralytics import YOLO

print("Initializing YOLOv8 Nano Model...")
# Load a lightweight, pre-trained YOLOv8n Classification model
model = YOLO("yolov8n-cls.pt")

print("Starting Transfer Learning Pipeline...")
# Train the model on your specific plant dataset
results = model.train(
    data="PlantVillage-1",             # The folder containing your dataset
    epochs=10,                         # Number of times to study the full dataset
    imgsz=224,                         # Standardize all image sizes to 224x224 pixels
    device="cpu"                       # Use CPU for the initial training phase
)

print("Training Complete! Check the 'runs/classify/train/weights/' folder for your new model.")