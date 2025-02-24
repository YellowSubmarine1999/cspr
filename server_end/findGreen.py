import os
import glob
import cv2
import numpy as np

def detect_green_center_distance_and_direction():
    """
    Searches the current folder for the newest .jpg image, detects the largest
    green region in that image, finds its centroid, and calculates both
    the distance and the direction (angle) from the image center.

    Returns:
        tuple or None: (distance, angle_in_degrees) if the green region is found,
                       otherwise None.

    Notes on angle convention:
        - The image coordinate system has (0,0) at the top-left corner.
        - x increases to the right, y increases downward.
        - angle = 0 degrees means the centroid is directly to the right of the center.
        - angle = 90 degrees means the centroid is directly below the center.
        - angle = 180 degrees means to the left, and angle = 270 or -90 degrees means above.
    """

    # 1. Search for all .jpg images in the current folder
    search_pattern = "uploads/*.jpg"

    image_paths = glob.glob(search_pattern)

    if not image_paths:
        print("No .jpg images found in the current folder.")
        return None

    # 2. Find the newest (most recently modified) .jpg image
    newest_image_path = max(image_paths, key=os.path.getmtime)

    # 3. Read the image
    img = cv2.imread(newest_image_path)
    if img is None:
        print(f"Failed to read image: {newest_image_path}")
        return None

    # 4. Get the image center coordinates
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2

    # 5. Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 6. Define the HSV range for green (adjust values if needed)
    lower_green = np.array([35, 43, 46], dtype=np.uint8)
    upper_green = np.array([77, 255, 255], dtype=np.uint8)

    # 7. Create a mask for the green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 8. Find contours of the green regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No green region detected.")
        return None

    # 9. Select the largest contour (by area)
    max_contour = max(contours, key=cv2.contourArea)

    # 10. Calculate the centroid using moments
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        print("Unable to calculate the centroid of the green region.")
        return None

    green_cx = int(M["m10"] / M["m00"])
    green_cy = int(M["m01"] / M["m00"])

    # 11. Compute the Euclidean distance between the image center and the centroid
    distance = np.sqrt((green_cx - center_x)**2 + (green_cy - center_y)**2)

    # 12. Calculate the direction (angle in degrees) from the image center
    #     atan2(dy, dx), where dy = (green_cy - center_y) and dx = (green_cx - center_x)
    #     0 degrees means right; 90 degrees means down, etc.
    dy = green_cy - center_y
    dx = green_cx - center_x
    angle_radians = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle_radians)

    # Optional: Keep angle in [0, 360) range
    if angle_degrees < 0:
        angle_degrees += 360

    print(f"Newest image: {os.path.basename(newest_image_path)}")
    print(f"Distance from center: {distance:.2f} pixels")
    print(f"Direction (angle): {angle_degrees:.2f} degrees")

    return (distance, angle_degrees)
