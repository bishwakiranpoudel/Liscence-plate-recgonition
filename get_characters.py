import os
import cv2
import numpy as np
from ultralytics import YOLO

def create_folder(folder_name):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def load_model(model_path):
    """Load the YOLO model from the given path."""
    return YOLO(model_path)

def predict_and_save(model, image_path, save_path):
    """Run predictions on an image and save the annotated output."""
    results = model.predict(source=image_path, save=True, show=False)
    return results

def process_predictions(results, save_cropped_folder):
    """Process YOLO predictions and save cropped images."""
    create_folder(save_cropped_folder)
    
    for i, result in enumerate(results):
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cropped_image = result.orig_img[y1:y2, x1:x2]
            cropped_path = os.path.join(save_cropped_folder, f"cropped_{i}.jpg")
            cv2.imwrite(cropped_path, cropped_image)
    
    return save_cropped_folder

def preprocess_image(image_path, debug_folder):
    """Apply preprocessing steps to the image and save intermediate results."""
    create_folder(debug_folder)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(debug_folder, "1_grayscale.png"), gray)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(debug_folder, "2_threshold.png"), thresh)

    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morphed = cv2.erode(cv2.dilate(thresh, kernel, iterations=1), kernel, iterations=1)
    cv2.imwrite(os.path.join(debug_folder, "3_morphed.png"), morphed)

    return morphed, image

def extract_and_save_characters(morphed, original_image, debug_folder, output_folder):
    """Extract individual characters from the morphed image and save them."""
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    create_folder(output_folder)

    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 2 < w < 50 and 5 < h < 60:  # Filtering small/large noise
            filtered_contours.append((x, y, w, h))

    filtered_contours = sorted(filtered_contours, key=lambda b: b[0])

    for i, (x, y, w, h) in enumerate(filtered_contours):
        char = morphed[y:y+h, x:x+w]
        char_path = os.path.join(output_folder, f"char_{i}.png")
        cv2.imwrite(char_path, char)

    return filtered_contours

def main():
    model_path = './best.pt'
    test_image_path = './test.jpg'
    cropped_folder = 'cropped_images'
    debug_folder = 'debug_steps'
    output_folder = 'characters'

    # Step 1: Load model
    model = load_model(model_path)

    # Step 2: Predict and save annotated image
    results = predict_and_save(model, test_image_path, cropped_folder)

    # Step 3: Process predictions to save cropped images
    process_predictions(results, cropped_folder)

    # Step 4: Preprocess first cropped image
    cropped_image_path = os.path.join(cropped_folder, "cropped_0.jpg")
    morphed, original_image = preprocess_image(cropped_image_path, debug_folder)

    # Step 5: Extract and save characters
    extract_and_save_characters(morphed, original_image, debug_folder, output_folder)

    print(f"Debug images saved in: {os.path.abspath(debug_folder)}")
    print(f"Characters saved in: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    main()
