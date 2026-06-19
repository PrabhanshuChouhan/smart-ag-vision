from ultralytics import YOLO

print("Loading your custom trained PyTorch model...")
# Pointing directly to the model trained previously
model = YOLO("runs/classify/train-2/weights/best.pt")

print("Compiling via Intel OpenVINO...")
# This command optimizes the computation graph for Intel hardware architectures
model.export(format="openvino")

print("Compilation complete. Your edge-ready model is ready for deployment.")