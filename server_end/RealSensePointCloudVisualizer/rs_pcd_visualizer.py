import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import time
import copy
def create_point_cloud_from_frames(color_frame, depth_frame, intrinsic):
    color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
    depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Align with RealSense coordinate system
    return pcd

def get_intrinsic_from_profile(profile):
    video_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        video_profile.width(), video_profile.height(),
        video_profile.get_intrinsics().fx, video_profile.get_intrinsics().fy,
        video_profile.get_intrinsics().ppx, video_profile.get_intrinsics().ppy)
    return intrinsic

def downsample_and_compute_fpfh(point_cloud, voxel_size):
    downsampled = point_cloud.voxel_down_sample(voxel_size)
    downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return downsampled, fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
    )
    return result
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)  # Use deepcopy to clone the point cloud
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    intrinsic = get_intrinsic_from_profile(profile)

    try:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        pcd = create_point_cloud_from_frames(color_frame, depth_frame, intrinsic)

        transformation = [[0.999, 0.0, 0.0, 0.02], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.01], [0.0, 0.0, 0.0, 1.0]]
        target_pcd = pcd.transform(transformation)

        voxel_size = 0.02  # voxel size
        source_down, source_fpfh = downsample_and_compute_fpfh(pcd, voxel_size)
        target_down, target_fpfh = downsample_and_compute_fpfh(target_pcd, voxel_size)

        start = time.time()
        result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print("Fast global registration took %.3f sec." % (time.time() - start))
        print(result_fast)

        draw_registration_result(source_down, target_down, result_fast.transformation)

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
