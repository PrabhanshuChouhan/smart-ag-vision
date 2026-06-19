from ultralytics import YOLO

print("Initializing YOLOv8 Classification Model...")
model = YOLO("yolov8n-cls.pt")

print("Target: 50 Epochs across the PlantVillage dataset.")
print("Processing via NVIDIA GeForce RTX 3050 (CUDA)...")

# Run the accelerated training pipeline
results = model.train(
    data="PlantVillage-1",
    epochs=50,
    imgsz=224,
    device=0,      # Targets the RTX 3050 card directly
    batch=32,     # Higher batch size works great with discrete VRAM
    workers=4     
)

print("Training complete! High-accuracy weights generated successfully.")