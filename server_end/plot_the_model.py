import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull

# --------------------- Parameter Settings ---------------------
input_csv_filename = "platform_workspace_data_more.csv"
feasible_angle_threshold = 45  # Used elsewhere in this script; not applied as initial filter on df_orig
z_step = 0.05                  # Step size for grouping Z values
target_z = 0.74                # Target Z value used for coverage ratio comparison

# --------------------- Rotation Matrix Functions ---------------------
def rot_z(yaw_deg):
    """
    Rotate by yaw_deg degrees about the Z-axis.
    """
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]
    ])

def rot_y(pitch_deg):
    """
    Rotate by pitch_deg degrees about the Y-axis.
    """
    pitch = np.radians(pitch_deg)
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rot_x(roll_deg):
    """
    Rotate by roll_deg degrees about the X-axis.
    """
    roll = np.radians(roll_deg)
    c, s = np.cos(roll), np.sin(roll)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

# --------------------- Compute Laser Mapping Point (Mapping to Z=0) ---------------------
def compute_laser_mapping(x_sol):
    """
    For a given platform solution x_sol = [X0, Y0, Z0, ROLL, PITCH, YAW],
    compute the intersection point of the platform's bottom laser (pointing in the local (0,0,-1) direction)
    with the Z=0 plane.

    Steps:
    1) Negate the YAW from the data to obtain the actual yaw_deg used for computation.
    2) Compute the rotation matrix R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll).
    3) Compute the global direction d = R @ [0, 0, -1]. If d[2] > 0, then set d = -d (to ensure downward direction).
    4) If |d[2]| < 1e-6, set t = 0; otherwise, t = -Z / d[2].
    5) The mapping point (MapX, MapY) = (X0 + t*d[0], Y0 + t*d[1]).
    """
    X, Y, Z, roll_deg, pitch_deg, yaw_deg = x_sol[:6]
    yaw_deg = -yaw_deg  # Negate YAW from the data

    # Compute rotation matrix
    R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    d = R @ np.array([0, 0, -1])

    # If the direction is upward, reverse it
    if d[2] > 0:
        d = -d

    # Avoid division by zero if d[2] is too small
    if abs(d[2]) < 1e-6:
        t = 0
    else:
        t = -Z / d[2]

    MapX = X + t * d[0]
    MapY = Y + t * d[1]
    return (MapX, MapY)

# --------------------- Compute Mapping Point on a Specified Plane (z = target_z) ---------------------
def compute_mapping_at_plane(x_sol, target_z):
    """
    Similar to compute_laser_mapping, but computes the intersection point on the plane z = target_z.
    Solve the equation: Z0 + t*d[2] = target_z  =>  t = (target_z - Z0) / d[2].
    """
    X, Y, Z, roll_deg, pitch_deg, yaw_deg = x_sol[:6]
    yaw_deg = -yaw_deg  # Negate YAW from the data

    R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    d = R @ np.array([0, 0, -1])

    if d[2] > 0:
        d = -d

    if abs(d[2]) < 1e-6:
        t = 0
    else:
        t = (target_z - Z) / d[2]

    MapX = X + t * d[0]
    MapY = Y + t * d[1]
    return (MapX, MapY)

# --------------------- Main Function ---------------------
def main():
    # Read CSV file and preprocess
    df = pd.read_csv(input_csv_filename)

    # Keep only rows with Z0 <= 0.85
    df = df[df['Z0'] <= 0.85].copy()

    # Filter based on the feasible angle threshold:
    # Keep rows where the absolute value of ROLL and PITCH is <= feasible_angle_threshold or >= (180 - feasible_angle_threshold)
    df = df[
        (
            (df['ROLL'].abs() <= feasible_angle_threshold)
            | (df['ROLL'].abs() >= (180 - feasible_angle_threshold))
        )
        &
        (
            (df['PITCH'].abs() <= feasible_angle_threshold)
            | (df['PITCH'].abs() >= (180 - feasible_angle_threshold))
        )
    ].copy()

    # Calculate the mapping point for each row on the Z=0 plane
    mapping_points = df.apply(
        lambda row: compute_laser_mapping(
            [row['X0'], row['Y0'], row['Z0'], row['ROLL'], row['PITCH'], row['YAW']]
        ),
        axis=1
    )
    df['MapX'] = mapping_points.apply(lambda pt: pt[0])
    df['MapY'] = mapping_points.apply(lambda pt: pt[1])

    # Group Z0 by z_step
    df['Z_round'] = (df['Z0'] / z_step).round() * z_step

    # --------------------- The following plotting code has been removed ---------------------
    # The script only generates data files without visualization.

    # Save the processed data to a CSV file
    df.to_csv("platform_workspace_data_processed.csv", index=False)
    print("Processed data file 'platform_workspace_data_processed.csv' has been generated.")

    # Additionally, compute mapping areas per Z group and save annotations data
    group_areas = []
    for z_val, group in df.groupby('Z_round'):
        pts = group[['MapX', 'MapY']].to_numpy()
        # Filter out points that are farther than 20 m from the origin (considered outliers)
        distances = np.linalg.norm(pts, axis=1)
        pts_filtered = pts[distances <= 20]

        if len(pts_filtered) >= 3:
            try:
                hull = ConvexHull(pts_filtered)
                area = hull.volume  # In 2D, use hull.volume as the area
            except Exception as e:
                print(f"ConvexHull error for Z={z_val}: {e}")
                area = 0
        else:
            area = 0

        group_areas.append({'Z': z_val, 'Area': area})

    area_df = pd.DataFrame(group_areas)
    area_df.to_csv("platform_workspace_annotations.csv", index=False)
    print("Annotations file 'platform_workspace_annotations.csv' has been generated.")

if __name__ == '__main__':
    main()
