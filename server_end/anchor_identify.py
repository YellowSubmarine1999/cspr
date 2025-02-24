import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import sys
import copy
import time
import requests
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    # Configure the pipeline to stream depth and color
    config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Delay to allow the camera to stabilize
    time.sleep(1)
    return pipeline, align, profile

def get_point_cloud(aligned_depth_frame, color_frame, intrinsic):
    color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
    depth_image = o3d.geometry.Image(np.asanyarray(aligned_depth_frame.get_data()))
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image,
        depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False)
    # Generate point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    # Do not flip axes
    return pcd

def get_intrinsic_from_profile(profile):
    video_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video_profile.get_intrinsics()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        video_profile.width(),
        video_profile.height(),
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy)
    return intrinsic

def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        return None, None, None, None

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image, aligned_depth_frame, color_frame

def preprocess_point_cloud(pcd, voxel_size=0.02):
    # Downsample the point cloud with a voxel size
    downsampled = pcd.voxel_down_sample(voxel_size)
    # Estimate normals
    downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))
    # Compute FPFH feature
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
    return downsampled, fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def apply_transformation(point, transformation):
    homogeneous_point = np.append(point, 1)
    transformed_point = transformation @ homogeneous_point
    return transformed_point[:3]

def handle_mouse_click(event, x, y, flags, shared_data):
    if event == cv2.EVENT_LBUTTONDOWN:
        pipeline = shared_data['pipeline']
        align = shared_data['align']
        profile = shared_data['profile']
        intrinsic = shared_data['intrinsic']
        cumulative_transformation = copy.deepcopy(shared_data['cumulative_transformation'])
        coordinates = shared_data['coordinates']

        print("Mouse clicked at ({}, {})".format(x, y))
        print("Starting registration for mouse click...")

        # Use local reference downsampled cloud and features
        local_reference_down = copy.deepcopy(shared_data['reference_down'])
        local_reference_fpfh = copy.deepcopy(shared_data['reference_fpfh'])

        # Get current frame
        color_image, depth_image, aligned_depth_frame, color_frame = get_aligned_images(pipeline, align)
        if color_image is None or depth_image is None:
            print("Failed to get images during mouse click handling.")
            return
        current_cloud = get_point_cloud(aligned_depth_frame, color_frame, intrinsic)
        if current_cloud is None:
            print("Failed to get point cloud during mouse click handling.")
            return
        current_down, current_fpfh = preprocess_point_cloud(current_cloud)
        if current_down is None or current_fpfh is None:
            print("Failed to preprocess point cloud during mouse click handling.")
            return

        # Register current frame to reference frame
        result = execute_global_registration(current_down, local_reference_down, current_fpfh, local_reference_fpfh, voxel_size=0.05)
        if result is None:
            print("Registration failed during mouse click handling.")
            return

        # Update cumulative transformation correctly (post-multiply)
        cumulative_transformation = cumulative_transformation @ result.transformation

        # Print the transformation matrix
        print("Transformation Matrix from registration:")
        print(result.transformation)

        # Get depth at clicked point
        depth = aligned_depth_frame.get_distance(x, y)
        if depth > 0:
            depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            # Do not flip axes
            point_3d = [point_3d[0], point_3d[1], point_3d[2]]
            # Apply cumulative transformation to point_3d
            transformed_point = apply_transformation(point_3d, cumulative_transformation)
            coordinates.append(transformed_point)
            print("Transformed coordinates of the selected point: X={:.2f}, Y={:.2f}, Z={:.2f}".format(
                transformed_point[0], transformed_point[1], transformed_point[2]))

            if len(coordinates) == 4:
                upload_coordinates(coordinates, cumulative_transformation)
                print("Collected 4 points, exiting program.")
                sys.exit(0)  # Exit the program
        else:
            print("No depth information at the clicked point.")
def upload_coordinates(coordinates, transformation_matrix):
    url = 'http://10.0.0.200:5000/update_coordinates'
    # Ensure the transformation_matrix is flattened properly
    if isinstance(transformation_matrix, np.ndarray):
        transformation_matrix = transformation_matrix.flatten().tolist()

    payload = {
        'x1': coordinates[0][0] * 100, 'y1': coordinates[0][1] * 100, 'z1': coordinates[0][2] * 100,
        'x2': coordinates[1][0] * 100, 'y2': coordinates[1][1] * 100, 'z2': coordinates[1][2] * 100,
        'x3': coordinates[2][0] * 100, 'y3': coordinates[2][1] * 100, 'z3': coordinates[2][2] * 100,
        'x4': coordinates[3][0] * 100, 'y4': coordinates[3][1] * 100, 'z4': coordinates[3][2] * 100,
        'transformation_matrix': transformation_matrix
    }
    response = requests.post(url, json=payload)
    print("Coordinates uploaded successfully.")
    print("Server response:", response.json())
def main():
    pipeline, align, profile = initialize_camera()
    intrinsic = get_intrinsic_from_profile(profile)
    # Get initial frame
    color_image, depth_image, aligned_depth_frame, color_frame = get_aligned_images(pipeline, align)
    if color_image is None or depth_image is None:
        print("Failed to get initial images.")
        sys.exit(1)
    reference_cloud = get_point_cloud(aligned_depth_frame, color_frame, intrinsic)
    if reference_cloud is None:
        print("Failed to get initial point cloud.")
        sys.exit(1)
    reference_down, reference_fpfh = preprocess_point_cloud(reference_cloud)
    if reference_down is None or reference_fpfh is None:
        print("Failed to preprocess initial point cloud.")
        sys.exit(1)

    cumulative_transformation = np.eye(4)
    coordinates = []

    shared_data = {
        'pipeline': pipeline,
        'align': align,
        'profile': profile,
        'intrinsic': intrinsic,
        'cumulative_transformation': cumulative_transformation,
        'reference_down': reference_down,
        'reference_fpfh': reference_fpfh,
        'coordinates': coordinates
    }

    cv2.namedWindow('RGB image')

    try:
        while True:
            # Get current frame
            color_image, depth_image, aligned_depth_frame, color_frame = get_aligned_images(pipeline, align)
            if color_image is None or depth_image is None:
                continue
            current_cloud = get_point_cloud(aligned_depth_frame, color_frame, intrinsic)
            if current_cloud is None:
                continue
            current_down, current_fpfh = preprocess_point_cloud(current_cloud)
            if current_down is None or current_fpfh is None:
                continue

            # Register current frame to reference frame
            result = execute_global_registration(current_down, shared_data['reference_down'], current_fpfh, shared_data['reference_fpfh'], voxel_size=0.05)
            if result is None:
                continue

            # Update cumulative transformation correctly (post-multiply)
            shared_data['cumulative_transformation'] = shared_data['cumulative_transformation'] @ result.transformation

            # Print the cumulative transformation matrix
            print("Updated Cumulative Transformation Matrix:")
            print(shared_data['cumulative_transformation'])

            # Update reference for next iteration
            shared_data['reference_down'] = current_down
            shared_data['reference_fpfh'] = current_fpfh

            # Display image and set mouse callback
            cv2.imshow('RGB image', color_image)
            cv2.setMouseCallback('RGB image', handle_mouse_click, shared_data)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("Exiting program.")
                break
        cv2.destroyAllWindows()
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
