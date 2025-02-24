import math
import numpy as np

def transition(coord1, coord2, coord3, coord4, coord5, position):
    """
    Calculate the adjustments for the four motors to move the robot from its current position
    to the target position while keeping the y coordinate constant.
    """
    # Convert the target position from pixel coordinates to real-world coordinates.
    # The scale factor needs to be adjusted based on actual conditions.
    scale_factor = 0.01  # Assume each pixel represents 0.01 meters; adjust as needed.
    pos_x_local = position[0] * scale_factor
    pos_z_local = position[1] * scale_factor  # Assume position[1] represents movement in the z direction.

    # Define the robot's local coordinate system.
    # Robot's local x-axis: the vector from coord1 to coord4.
    local_x = np.array(coord4) - np.array(coord1)
    norm_local_x = np.linalg.norm(local_x)
    if norm_local_x == 0:
        print("[Error] local_x vector has zero magnitude. coord1 and coord4 may be the same.")
        return None
    local_x_unit = local_x / norm_local_x

    # Robot's local z-axis: the vector from coord1 to coord2.
    local_z = np.array(coord2) - np.array(coord1)
    norm_local_z = np.linalg.norm(local_z)
    if norm_local_z == 0:
        print("[Error] local_z vector has zero magnitude. coord1 and coord2 may be the same.")
        return None
    local_z_unit = local_z / norm_local_z

    # Robot's local y-axis: the vector perpendicular to the local x and z axes.
    local_y = np.cross(local_x_unit, local_z_unit)
    norm_local_y = np.linalg.norm(local_y)
    if norm_local_y == 0:
        print("[Error] local_y vector has zero magnitude. local_x and local_z may be parallel.")
        return None
    local_y_unit = local_y / norm_local_y

    # Construct the rotation matrix (from the local coordinate system to the world coordinate system).
    rotation_matrix = np.column_stack((local_x_unit, local_y_unit, local_z_unit))

    # Define the target position vector in the local coordinate system (keeping y_local as 0).
    target_local = np.array([pos_x_local, 0, pos_z_local])  # [x_local, y_local (height), z_local]

    target_world = np.array(coord1) + rotation_matrix @ target_local
    target_world[1] = coord5[1]

    target_x, target_y, target_z = target_world
    print(f"[Info] Target World Coordinates: x={target_x}, y={target_y}, z={target_z}")

    L1 = distance(coord1, coord5)
    L2 = distance(coord2, coord5)
    L3 = distance(coord3, coord5)
    L4 = distance(coord4, coord5)

    new_L1 = distance(coord1, (target_x, target_y, target_z))
    new_L2 = distance(coord2, (target_x, target_y, target_z))
    new_L3 = distance(coord3, (target_x, target_y, target_z))
    new_L4 = distance(coord4, (target_x, target_y, target_z))

    print(f"[Info] New Distances: L1={new_L1}, L2={new_L2}, L3={new_L3}, L4={new_L4}")
    print(f"[Info] Current Distances: L1={L1}, L2={L2}, L3={L3}, L4={L4}")

    motor1 = new_L1 - L1
    motor2 = new_L2 - L2
    motor3 = new_L3 - L3
    motor4 = new_L4 - L4

    if any(math.isnan(m) or math.isinf(m) for m in [motor1, motor2, motor3, motor4]):
        print("[Error] Invalid motor adjustments calculated. Cannot proceed.")
        return None

    print(f"[Info] Motor Adjustments: motor1={motor1}, motor2={motor2}, motor3={motor3}, motor4={motor4}")

    return motor1, motor2, motor3, motor4

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 +
                     (p1[1] - p2[1]) ** 2 +
                     (p1[2] - p2[2]) ** 2)
