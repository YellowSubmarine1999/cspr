import os
import glob
import cv2
import numpy as np

# Ultralytics YOLOv8 model import
from ultralytics import YOLO

# Change this to the class name that your model recognizes for the plant.
TARGET_CLASS = "potted plant"

# Replace with your own YOLOv8 model weights file (e.g., "yolov8n.pt", "best.pt", etc.)
MODEL_PATH = "best.pt"

def detect_plant_center_distance_and_direction():
    """
    1. Find the latest (newest) .jpg file in the 'uploads/' folder.
    2. Load the image.
    3. Use a YOLOv8 model to detect the specified class (TARGET_CLASS).
    4. Pick the bounding box with the largest area for that class.
    5. Compute:
       - The distance in pixels between the bounding box center and the image center.
       - The angle in degrees, based on atan2(dy, dx), where 0 degrees is to the right,
         90 degrees is down, 180 degrees is to the left, and 270 degrees is up.

    Returns:
        (distance, angle_degrees) if detection is successful, otherwise None.
    """

    # 1. Look for .jpg files in the 'uploads/' folder
    search_pattern = "uploads/*.jpg"
    image_paths = glob.glob(search_pattern)

    if not image_paths:
        print("No .jpg images found in 'uploads/' folder.")
        return None

    # Get the newest .jpg file by modification time
    newest_image_path = max(image_paths, key=os.path.getmtime)

    # 2. Read the image using OpenCV
    img = cv2.imread(newest_image_path)
    if img is None:
        print(f"Failed to read image: {newest_image_path}")
        return None

    # Image dimensions
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2

    # 3. Load the YOLOv8 model
    model = YOLO(MODEL_PATH)

    # 4. Perform inference on the image with a chosen confidence threshold (e.g. 0.25)
    results = model.predict(source=img, conf=0.25)

    # In YOLOv8, 'results' is typically a list; usually we only need results[0]
    if len(results) == 0 or results[0].boxes is None:
        print("No detections were found.")
        return None

    # Extract boxes from results
    boxes = results[0].boxes

    max_area = 0
    best_bbox_center = None

    # Iterate through all detected boxes
    for box in boxes:
        # box.cls, box.conf, box.xyxy
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        label_name = results[0].names[cls_id]  # Class label

        # Check if this box corresponds to our TARGET_CLASS
        if label_name == TARGET_CLASS:
            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height

            if area > max_area:
                max_area = area
                # Calculate the center of this bounding box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                best_bbox_center = (cx, cy)

    if best_bbox_center is None:
        print(f"No '{TARGET_CLASS}' found in the image.")
        return None

    # Extract the bounding box center
    box_cx, box_cy = best_bbox_center

    # 5. Compute Euclidean distance from the image center
    distance = np.sqrt((box_cx - center_x) ** 2 + (box_cy - center_y) ** 2)

    # Compute angle using atan2(dy, dx)
    dx = box_cx - center_x
    dy = box_cy - center_y
    angle_radians = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle_radians)

    # Optionally normalize angle to [0, 360)
    if angle_degrees < 0:
        angle_degrees += 360

    print(f"Newest image: {os.path.basename(newest_image_path)}")
    print(f"Distance from center: {distance:.2f} pixels")
    print(f"Direction (angle): {angle_degrees:.2f} degrees")

    return (distance, angle_degrees)

if __name__ == "__main__":
    result = detect_plant_center_distance_and_direction()
    if result is None:
        print("No plant detected or an error occurred.")
    else:
        dist, angle = result
        print(f"Final result => distance: {dist:.2f} pixels, angle: {angle:.2f} degrees")
