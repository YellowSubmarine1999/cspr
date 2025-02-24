import numpy as np
import math
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

# ========== Parameter Settings ==========
G = 9.81            # Gravitational acceleration (m/s^2)
MASS = 1.0          # Platform mass (kg)
PLATFORM_LENGTH = 0.20
PLATFORM_WIDTH  = 0.15

# Working area range (platform center) and step size
X_MIN, X_MAX, X_STEP = 0.0, 2.0, 0.02
Y_MIN, Y_MAX, Y_STEP = 0.0, 2.0, 0.02
# Z scanning layers: from 0.85 m downward, each layer with a step of 0.05 m
Z_MIN, Z_MAX, Z_STEP = 0.0, 0.8, 0.03

# Anchor point arrangement
ANCHOR_Z = 1.1
anchor_points = np.array([
    [0.0, 0.0, ANCHOR_Z],
    [2.0, 0.0, ANCHOR_Z],
    [2.0, 2.0, ANCHOR_Z],
    [0.0, 2.0, ANCHOR_Z],
])

# ========== Rotation Matrices ==========
def rot_z(yaw_deg):
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, -s,  0],
                     [ s,  c,  0],
                     [ 0,  0,  1]])

def rot_y(pitch_deg):
    pitch = np.radians(pitch_deg)
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[ c,  0,  s],
                     [ 0,  1,  0],
                     [-s,  0,  c]])

def rot_x(roll_deg):
    roll = np.radians(roll_deg)
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[ 1,  0,   0],
                     [ 0,  c,  -s],
                     [ 0,  s,   c]])

def platform_corners_world(x):
    """
    x = [X, Y, Z, roll_deg, pitch_deg, yaw_deg, T1, T2, T3, T4]
    Returns the coordinates of the platform's four corners in the world coordinate system (4×3)
    Rotation order: Rz(yaw) -> Ry(pitch) -> Rx(roll).
    """
    X, Y, Z, roll_deg, pitch_deg, yaw_deg = x[:6]
    R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    center = np.array([X, Y, Z])
    local_pts = np.array([
        [ PLATFORM_LENGTH/2,  PLATFORM_WIDTH/2,  0],
        [ PLATFORM_LENGTH/2, -PLATFORM_WIDTH/2,  0],
        [-PLATFORM_LENGTH/2, -PLATFORM_WIDTH/2,  0],
        [-PLATFORM_LENGTH/2,  PLATFORM_WIDTH/2,  0],
    ])
    corners = []
    for pt in local_pts:
        corners.append(center + R @ pt)
    return np.array(corners)

# ========== Static Equilibrium Equations (10 dimensions) ==========
def equilibrium_equations(x, L_list):
    """
    x = [X, Y, Z, roll, pitch, yaw, T1, T2, T3, T4]
    L_list = [L1, L2, L3, L4]
    Returns the residuals of 10 equations:
      - 4 geometric (cable length) constraints
      - 3 force equilibrium equations (Fx, Fy, Fz=0)
      - 3 moment equilibrium equations (Mx, My, Mz=0)
    """
    X, Y, Z, roll, pitch, yaw, T1, T2, T3, T4 = x
    L1, L2, L3, L4 = L_list

    # Cable length geometric constraints
    cw = platform_corners_world(x)
    dist1 = np.linalg.norm(cw[0] - anchor_points[0]) - L1
    dist2 = np.linalg.norm(cw[1] - anchor_points[1]) - L2
    dist3 = np.linalg.norm(cw[2] - anchor_points[2]) - L3
    dist4 = np.linalg.norm(cw[3] - anchor_points[3]) - L4

    # Cable tension direction
    def cable_force(ci, ai, T):
        vec = ci - ai
        length = np.linalg.norm(vec)
        if length < 1e-12:
            return np.zeros(3)
        return T * (vec / length)

    F1 = cable_force(cw[0], anchor_points[0], T1)
    F2 = cable_force(cw[1], anchor_points[1], T2)
    F3 = cable_force(cw[2], anchor_points[2], T3)
    F4 = cable_force(cw[3], anchor_points[3], T4)
    Fg = np.array([0, 0, -MASS * G])

    # Force equilibrium
    Fx, Fy, Fz = F1 + F2 + F3 + F4 + Fg

    # Moment equilibrium (taking the platform center as reference)
    center = np.array([X, Y, Z])
    r1 = cw[0] - center
    r2 = cw[1] - center
    r3 = cw[2] - center
    r4 = cw[3] - center
    M1 = np.cross(r1, F1)
    M2 = np.cross(r2, F2)
    M3 = np.cross(r3, F3)
    M4 = np.cross(r4, F4)
    Mx, My, Mz = M1 + M2 + M3 + M4

    return np.array([
        dist1, dist2, dist3, dist4,
        Fx, Fy, Fz,
        Mx, My, Mz
    ])

def solve_cspr_pose_bounded(L_list, x_guess):
    """
    Solve using least_squares with bounds to constrain T1..T4:
      0 <= T_i <= 2000
    Also set appropriate bounds for X, Y, Z, roll, pitch, yaw.
    """
    def residuals(vars_):
        return equilibrium_equations(vars_, L_list)

    # Variable order: [X, Y, Z, roll, pitch, yaw, T1, T2, T3, T4]
    lb = [0.0, 0.0, 0.0,  -180.0, -180.0, -180.0,   0.0,   0.0,   0.0,   0.0]
    ub = [3.0, 3.0, 3.0,   180.0,  180.0,  180.0, 2000.0, 2000.0, 2000.0, 2000.0]

    result = least_squares(
        fun=residuals,
        x0=x_guess,
        bounds=(lb, ub),
        xtol=1e-6,
        ftol=1e-6,
        max_nfev=1000
    )
    # Check convergence
    ok = (result.success and np.linalg.norm(result.fun) < 1e-3)
    return result.x, ok, result.message

def approx_l_i(center_approx):
    """Estimate each cable length L_i based on an approximate platform center position."""
    return np.linalg.norm(anchor_points - center_approx, axis=1)

def normalize_angle_deg(a):
    """Normalize an angle to the range [-180, 180)."""
    a = a % 360
    if a >= 180:
        a -= 360
    return a

def check_feasible(roll_deg, pitch_deg):
    """
    Determine if the attitude is feasible: (|roll|<=20 or >=160) and (|pitch|<=20 or >=160)
    """
    cond_roll = (abs(roll_deg) <= 20 or abs(roll_deg) >= 160)
    cond_pitch = (abs(pitch_deg) <= 20 or abs(pitch_deg) >= 160)
    return (cond_roll and cond_pitch)

def build_greedy_path_2d(start_xy):
    """
    Construct a path (x,y) using BFS or a greedy algorithm, starting from start_xy.
    """
    coords = []
    xs = np.arange(X_MIN, X_MAX + X_STEP/2, X_STEP)
    ys = np.arange(Y_MIN, Y_MAX + Y_STEP/2, Y_STEP)
    for yy in ys:
        for xx in xs:
            coords.append((xx, yy))
    unvisited = set(coords)
    start_pt = min(coords, key=lambda c: (c[0]-start_xy[0])**2 + (c[1]-start_xy[1])**2)
    path = [start_pt]
    unvisited.remove(start_pt)
    current = start_pt
    while unvisited:
        near = min(unvisited, key=lambda c: (c[0]-current[0])**2 + (c[1]-current[1])**2)
        path.append(near)
        unvisited.remove(near)
        current = near
    return path

def compute_laser_mapping(x_sol):
    """
    Compute the laser mapping point: the intersection of a downward ray from the platform center with the z=0 plane.
    x_sol = [X, Y, Z, roll, pitch, yaw, ...]
    """
    X, Y, Z, roll_deg, pitch_deg, yaw_deg = x_sol[:6]
    R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    d = R @ np.array([0, 0, -1])
    if d[2] > 0:
        d = -d
    if abs(d[2]) < 1e-6:
        return (X, Y)
    t = -Z / d[2]
    return (X + t*d[0], Y + t*d[1])

def bfs_crosslayer_dynamic():
    """
    Perform a layer-by-layer (in Z) scan.
    For each layer, use BFS to traverse (x,y) points, with the initial guess for each point fixed as [0,0,z0,0,0,0,5,5,5,5].
    Returns (data, annotations) for subsequent analysis.
    """
    data = []
    annotations = []
    z_values = np.arange(Z_MAX, Z_MIN - 1e-9, -Z_STEP)  # From Z_MAX downward
    z_values = z_values[z_values > 0.0]                   # Ensure Z > 0

    point_count = 0

    for i, z0 in enumerate(z_values):
        print(f"=== Layer Z={z0:.2f}, i={i} ===")

        # Build BFS path: starting point set to (0,0)
        path_2d = build_greedy_path_2d((0.0, 0.0))

        layer_feasible_mappings = []
        layer_feasible_tensions = []

        success_count = 0

        # Use a fixed initial guess for the entire layer
        init_guess_layer = [0.0, 0.0, z0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]

        for (xv, yv) in path_2d:
            point_count += 1
            if point_count % 100 == 0:
                print(f"  ... processed {point_count} points so far ...")

            # Estimate cable lengths
            L_list = approx_l_i([xv, yv, z0])
            # Use the fixed initial guess for this layer
            x_sol, ok, _msg = solve_cspr_pose_bounded(L_list, init_guess_layer)

            if not ok:
                continue  # Did not converge, skip

            Xr, Yr, Zr, roll_deg, pitch_deg, yaw_deg, t1, t2, t3, t4 = x_sol
            roll_deg  = normalize_angle_deg(roll_deg)
            pitch_deg = normalize_angle_deg(pitch_deg)
            yaw_deg   = normalize_angle_deg(yaw_deg)
            max_t = max(abs(t1), abs(t2), abs(t3), abs(t4))

            # Record data
            feasible = check_feasible(roll_deg, pitch_deg)
            data.append([xv, yv, z0, roll_deg, pitch_deg, yaw_deg, feasible, max_t])
            success_count += 1

            # If the attitude is feasible, record the mapping point
            if feasible:
                mapping_pt = compute_laser_mapping(x_sol)
                layer_feasible_mappings.append(mapping_pt)
                layer_feasible_tensions.append(max_t)

        print(f"  Layer Z={z0:.2f}, success_count={success_count}")
        # Compute the convex hull and maximum tension for this layer
        if layer_feasible_mappings:
            xs_map = [pt[0] for pt in layer_feasible_mappings]
            ys_map = [pt[1] for pt in layer_feasible_mappings]
            avg_x = np.mean(xs_map)
            avg_y = np.mean(ys_map)

            if len(layer_feasible_mappings) >= 3:
                pts = np.array(layer_feasible_mappings)
                try:
                    hull = ConvexHull(pts)
                    area = hull.volume  # In 2D, use hull.volume as the area
                except Exception as e:
                    print(f"ConvexHull error at Z={z0:.2f}: {e}")
                    area = 0.0
            else:
                area = 0.0

            max_t_layer = max(layer_feasible_tensions)
            print(f"    => avg_mapping=({avg_x:.2f}, {avg_y:.2f}), area={area:.4f}, maxT={max_t_layer:.2f}")
            annotations.append((z0, avg_x, avg_y, area, max_t_layer))
        else:
            annotations.append((z0, 0.0, 0.0, 0.0, 0.0))

    return data, annotations

def main():
    # Generate data from the BFS crosslayer dynamic scan
    raw_data, annotations = bfs_crosslayer_dynamic()
    df = pd.DataFrame(
        raw_data,
        columns=['X0', 'Y0', 'Z0', 'ROLL', 'PITCH', 'YAW', 'Feasible', 'MaxT']
    )
    df['Feasible'] = df['Feasible'].map(lambda b: 'Yes' if b else 'No')
    print(f"Total points collected: {len(df)}")

    # Save the data to a CSV file
    df.to_csv("platform_workspace_data_more.csv", index=False)
    print("Data file 'platform_workspace_data_more.csv' has been generated.")

    # Optionally, you can also save the annotations data to a separate CSV file:
    ann_df = pd.DataFrame(annotations, columns=['Z', 'MapX', 'MapY', 'Area', 'MaxT'])
    ann_df.to_csv("platform_workspace_annotations.csv", index=False)
    print("Annotations file 'platform_workspace_annotations.csv' has been generated.")

if __name__ == '__main__':
    main()
