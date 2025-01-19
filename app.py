from ultralytics import YOLO
import cv2

# Load YOLO models
lp_model = YOLO("license_plate_model.pt")  # License plate detection model
char_model = YOLO("character_model.pt")  # Character detection model

# Open video file
cap = cv2.VideoCapture('video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect license plates
    lp_results = lp_model(frame)

    for result in lp_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf.item()  # Confidence score

            if confidence > 0.5:  # Confidence threshold
                # Crop the detected license plate
                lp_crop = frame[y1:y2, x1:x2]

                # Detect characters in the cropped license plate
                char_results = char_model(lp_crop)
                characters = []
                for char_result in char_results:
                    for char_box in char_result.boxes:
                        char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0])
                        char_conf = char_box.conf.item()
                        if char_conf > 0.5:  # Confidence threshold for characters
                            # Store detected characters (for annotation or ordering)
                            characters.append((char_x1, char_y1, char_x2, char_y2, char_conf))

                            # Optionally draw characters on lp_crop
                            cv2.rectangle(lp_crop, (char_x1, char_y1), (char_x2, char_y2), (255, 0, 0), 2)

                # Draw the license plate bounding box on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add recognized characters to the frame
                if characters:
                    text = " ".join([f"char_{i}" for i, _ in enumerate(characters)])  # Placeholder for real characters
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
