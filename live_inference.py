import cv2
from ultralytics import YOLO

print("Loading Intel OpenVINO Edge Model...")
# Point directly to the optimized OpenVINO directory
model = YOLO("runs/classify/train-2/weights/best_openvino_model/")

print("Starting Webcam Stream...")
print("Press 'q' in the video window to quit.")

# Initialize the default webcam (Camera 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from webcam. Exiting...")
        break

    # Run hardware-accelerated inference on the current frame
    # verbose=False prevents flooding the terminal with text
    results = model(frame, verbose=False)

    # Extract the top predicted disease and its confidence percentage
    top_class_id = results[0].probs.top1
    top_class_name = results[0].names[top_class_id]
    confidence = results[0].probs.top1conf.item() * 100

    # Format the text to display on the screen
    display_text = f"{top_class_name}: {confidence:.1f}%"

    # Draw a black background rectangle for text readability
    cv2.rectangle(frame, (5, 15), (600, 65), (0, 0, 0), -1)
    
    # Overlay the prediction text in bright green
    cv2.putText(frame, display_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the live video feed
    cv2.imshow("Smart Ag Vision - Intel Core Ultra Edge AI", frame)

    # Listen for the 'q' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()