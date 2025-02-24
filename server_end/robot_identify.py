import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import time

SERVER_URL = 'http://10.0.0.200:5000'

def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams instead of IR
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start pipeline
    pipeline.start(config)

    # Align depth to color (so depth_frame pixels match color_frame coordinates)
    align = rs.align(rs.stream.color)

    return pipeline, align

def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise ValueError("Unable to get color or depth frames.")

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image, depth_frame

def apply_transformation(point, transformation):
    homogeneous_point = np.append(point, 1)
    transformed_point = transformation @ homogeneous_point
    return transformed_point[:3]

def fetch_transformation_matrix():
    try:
        response = requests.get(f'{SERVER_URL}/get_coordinates')
        data = response.json()
        transformation_matrix = np.array(data['transformation_matrix']).reshape((4, 4))
        return transformation_matrix
    except Exception as e:
        print(f"Error fetching transformation matrix: {e}")
        return np.eye(4)

def fetch_target_coordinates():
    try:
        response = requests.get(f'{SERVER_URL}/get_coordinates')
        if response.status_code == 200:
            data = response.json()
            target_coords = (
                float(data['target']['x']),
                float(data['target']['y']),
                float(data['target']['z'])
            )
            return target_coords
        else:
            print('Failed to fetch target coordinates.')
            return None
    except Exception as e:
        print(f"Error fetching target coordinates: {e}")
        return None

def get_flags():
    try:
        response = requests.get(f'{SERVER_URL}/get_coordinates')
        if response.status_code == 200:
            data = response.json()
            flags = {
                'pi1': data['pi1']['flag'],
                'pi2': data['pi2']['flag'],
                'pi3': data['pi3']['flag'],
                'pi4': data['pi4']['flag']
            }
            return flags
        else:
            print('Failed to fetch flags.')
            return None
    except Exception as e:
        print(f"Error fetching flags: {e}")
        return None

def send_target_coordinates(target_coords):
    payload = {
        'update_type': 'target',
        'tx': str(target_coords['x']),
        'ty': str(target_coords['y']),
        'tz': str(target_coords['z'])
    }
    response = requests.post(f'{SERVER_URL}/', data=payload)
    if response.status_code == 200:
        print(f'Sent target coordinates: {target_coords}')
    else:
        print('Failed to send target coordinates.')

# Global variables for callback use
last_depth_frame = None
transformation = np.eye(4)
initial_point = None
clicked = False

def fetch_estimated_ground_normal():
    try:
        response = requests.get(f'{SERVER_URL}/get_coordinates')
        if response.status_code == 200:
            data = response.json()
            nx = float(data['estimated_ground_normal']['x'])
            ny = float(data['estimated_ground_normal']['y'])
            nz = float(data['estimated_ground_normal']['z'])
            normal = np.array([nx, ny, nz], dtype=float)
            # Ensure normal is unit-length
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6:
                normal = normal / norm_len
            return normal
        else:
            print("Failed to fetch estimated ground normal, using default [0,0,1].")
            return np.array([0,0,1], dtype=float)
    except Exception as e:
        print(f"Error fetching estimated ground normal: {e}, using default [0,0,1].")
        return np.array([0,0,1], dtype=float)

def on_mouse_click(event, x, y, flags, param):
    global initial_point, clicked, last_depth_frame, transformation
    if event == cv2.EVENT_LBUTTONDOWN:
        # Directly set the initial_point to the clicked location
        initial_point = (x, y)
        clicked = True
        print(f"Initial position selected at ({x}, {y})")

    if event == cv2.EVENT_RBUTTONDOWN:
        if last_depth_frame is not None:
            depth = last_depth_frame.get_distance(x, y)
            if depth > 0:
                depth_intrinsics = last_depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                transformed_point = apply_transformation([point_3d[0], point_3d[1], point_3d[2]], transformation)

                # Fetch the estimated ground normal from server
                normal = fetch_estimated_ground_normal()

                # Move the point 5cm along the ground normal direction
                offset = 0.05
                point_3d_above = transformed_point - normal * offset

                target_dict = {
                    'x': point_3d_above[0] * 100,  # convert to cm
                    'y': point_3d_above[1] * 100,
                    'z': point_3d_above[2] * 100
                }
                send_target_coordinates(target_dict)
            else:
                print("Unable to get depth at the clicked point.")
        else:
            print("Depth frame not ready.")

###############################################################################
# Tracking Yellow in the RGB Image
###############################################################################
def track_yellow(color_image, previous_point, search_radius=20):
    """
    Detect the largest yellow area near the 'previous_point' in a local ROI.
    Return the centroid (x, y) if found, otherwise None.
    """

    # Convert to HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define a yellow range in HSV (adjust if your yellow is different)
    # These values are just an example; you might need to tweak them.
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)   # Create a mask for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Restrict search to a region near previous_point
    x_prev, y_prev = previous_point
    h, w = mask.shape[:2]
    x_start = max(0, x_prev - search_radius)
    x_end = min(w, x_prev + search_radius)
    y_start = max(0, y_prev - search_radius)
    y_end = min(h, y_prev + search_radius)

    roi = mask[y_start:y_end, x_start:x_end]

    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    # Compute centroid in ROI, then translate to full image coordinates
    cX = int(M["m10"] / M["m00"]) + x_start
    cY = int(M["m01"] / M["m00"]) + y_start

    return (cX, cY)

if __name__ == "__main__":
    pipeline, align = initialize_camera()
    cv2.namedWindow('Color Image')
    cv2.setMouseCallback('Color Image', on_mouse_click)

    upload_coordinates = True
    last_target_coords = None

    # Fetch transformation matrix
    transformation = fetch_transformation_matrix()

    try:
        print("Please click on the YELLOW object in the image to initialize tracking.")
        while True:
            color_image, depth_image, depth_frame = get_aligned_images(pipeline, align)
            last_depth_frame = depth_frame

            display_image = color_image.copy()

            if clicked:
                # Draw initial point
                cv2.circle(display_image, initial_point, 5, (0, 255, 0), -1)
                print("Tracking started.")
                break

            cv2.imshow('Color Image', display_image)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                pipeline.stop()
                cv2.destroyAllWindows()
                exit(0)

        previous_point = initial_point

        while True:
            color_image, depth_image, depth_frame = get_aligned_images(pipeline, align)
            last_depth_frame = depth_frame

            # Attempt to track yellow near the previous point
            current_point = track_yellow(color_image, previous_point)

            if current_point is not None:
                x, y = current_point
                depth = depth_frame.get_distance(x, y)
                if depth > 0:
                    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                    transformed_point = apply_transformation([point_3d[0], point_3d[1], point_3d[2]], transformation)

                    display_image = color_image.copy()
                    cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow('Color Image', display_image)

                    current_target_coords = fetch_target_coordinates()

                    if current_target_coords is not None:
                        if last_target_coords != current_target_coords:
                            print("Target coordinates updated. Pausing robot coordinates upload.")
                            upload_coordinates = False
                            last_target_coords = current_target_coords

                    if not upload_coordinates and current_target_coords is not None:
                        flags = get_flags()
                        if flags is not None and '0' in flags.values():
                            print("Detected a Pi flag as 0. Resuming robot coordinates upload.")
                            upload_coordinates = True

                    if upload_coordinates:
                        try:
                            response = requests.post(f'{SERVER_URL}/update_coordinates', json={
                                'dcx': str(transformed_point[0] * 100),
                                'dcy': str(transformed_point[1] * 100),
                                'dcz': str(transformed_point[2] * 100)
                            })
                            if response.status_code != 200:
                                print('Failed to upload robot coordinates.')
                        except Exception as e:
                            print(f"Error uploading coordinates: {e}")

                    previous_point = current_point
                else:
                    print("Unable to get depth at the tracked point.")
            else:
                # If we cannot detect yellow near the previous point, just show the image
                display_image = color_image.copy()
                cv2.imshow('Color Image', display_image)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    break

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
