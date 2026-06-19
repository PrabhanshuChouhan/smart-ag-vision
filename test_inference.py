import glob
from ultralytics import YOLO

print("Loading Intel OpenVINO Edge Model...")
model = YOLO("runs/classify/train-2/weights/best_openvino_model/")

print("Locating a test image from your dataset...")
# Automatically find any .jpg in your validation folder
test_images = glob.glob("PlantVillage-1/valid/**/*.jpg", recursive=True)

if not test_images:
    print("Could not find any test images.")
else:
    # Grab the very first image it finds
    test_image_path = test_images[0]
    print(f"Testing on: {test_image_path}\n")

    # Run hardware-accelerated inference
    # verbose=False keeps the terminal clean
    results = model(test_image_path, verbose=False)

    # Extract the data
    top_class_id = results[0].probs.top1
    top_class_name = results[0].names[top_class_id]
    confidence = results[0].probs.top1conf.item() * 100

    # Print a clean terminal dashboard
    print("=" * 50)
    print(" EDGE INFERENCE RESULTS")
    print("=" * 50)
    print(f"Predicted Disease : {top_class_name}")
    print(f"Confidence Score  : {confidence:.2f}%")
    print("=" * 50)
    print("Hardware Processed via Intel OpenVINO IR.")