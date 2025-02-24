import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import fsolve

# Select the TkAgg backend
matplotlib.use('TkAgg')

############################################
# 1. Compute a similarity transform (scale + rotation + translation)
############################################
def compute_transformation_matrix(source_points, target_points):
    """
    Returns (scale, R, t, T_mat):
      scale: scalar
      R: (3,3) rotation matrix (orthonormal, det>0)
      t: (3,) translation
      T_mat: 4x4 homogeneous transform with T_mat[:3,:3] = scale * R
    """
    assert source_points.shape == target_points.shape

    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # 1) Global scale
    norm_s = np.sum(source_centered**2)
    norm_t = np.sum(target_centered**2)
    if abs(norm_s) < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(norm_t / norm_s)

    # 2) Kabsch rotation on (source_centered) -> (target_centered/scale)
    T_scaled = target_centered / scale
    H = source_centered.T @ T_scaled
    U, _, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T

    # 3) Translation
    t_ = centroid_target - scale * R_ @ centroid_source

    # 4) 4x4 homogeneous transformation matrix
    T_mat = np.eye(4)
    T_mat[:3, :3] = scale * R_
    T_mat[:3, 3] = t_

    return scale, R_, t_, T_mat


############################################
# 2. Some input data
############################################
# "Ideal" anchors in a 2x2 square at z=1.1
source_points = np.array([
    [0, 0, 1.1],
    [2, 0, 1.1],
    [2, 2, 1.1],
    [0, 2, 1.1],
])

# Real measured anchors (example)
pi1 = np.array([-0.34867938, -0.08703495, 1.7683378])
pi2 = np.array([0.62301098, -0.26469535, 1.79582044])
pi3 = np.array([0.31014294, -0.69673523, 2.84133737])
pi4 = np.array([-0.45098935, -0.44436382, 2.46842482])
target_points = np.array([pi1, pi2, pi3, pi4])

scale_trans, R_trans, t_trans, T_trans = compute_transformation_matrix(source_points, target_points)
print("[Similarity] scale =", scale_trans)
print("[Similarity] R =\n", R_trans)
print("[Similarity] t =", t_trans)
print("[Similarity] T =\n", T_trans)

############################################
# 3. Simplified CSPR model in the "ideal" system
############################################
g = 9.81
mass = 1.0
platform_length = 0.20
platform_width  = 0.15

anchors_simple = source_points.copy()

local_corners = np.array([
    [platform_length / 2,  platform_width / 2, 0],
    [platform_length / 2, -platform_width / 2, 0],
    [-platform_length / 2, -platform_width / 2, 0],
    [-platform_length / 2,  platform_width / 2, 0],
])

def rot_x(rx_deg):
    rx = math.radians(rx_deg)
    c, s = math.cos(rx), math.sin(rx)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])

def rot_y(ry_deg):
    ry = math.radians(ry_deg)
    c, s = math.cos(ry), math.sin(ry)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rot_z(rz_deg):
    rz = math.radians(rz_deg)
    c, s = math.cos(rz), math.sin(rz)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

def platform_corners_world(x):
    X, Y, Z, roll, pitch, yaw = x[:6]
    R_ = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    center = np.array([X, Y, Z])
    c_ = []
    for cc in local_corners:
        c_.append(center + R_ @ cc)
    return np.array(c_)

def equilibrium_equations(x, L_list):
    """
    x = [X, Y, Z, roll, pitch, yaw, T1, T2, T3, T4]
    L_list = [L1, L2, L3, L4], each = ||anchor_i - center||
    => eq0..3: ||corner_i - anchor_i|| - L_i = 0
    => eq4..6: sum of forces = 0
    => eq7..9: sum of moments = 0
    """
    X, Y, Z, roll, pitch, yaw, T1, T2, T3, T4 = x
    corners = platform_corners_world(x)

    eq = []
    # (1) Length constraints
    for i in range(4):
        dist = np.linalg.norm(corners[i] - anchors_simple[i])
        eq.append(dist - L_list[i])

    def cable_force(i, tension):
        vec = corners[i] - anchors_simple[i]
        L_ = np.linalg.norm(vec)
        if L_ < 1e-12:
            return np.zeros(3)
        return tension * (vec / L_)

    F_sum = (cable_force(0, T1) +
             cable_force(1, T2) +
             cable_force(2, T3) +
             cable_force(3, T4) +
             np.array([0, 0, -mass * g]))
    eq.extend(F_sum)

    center = np.array([X, Y, Z])
    M_sum = np.zeros(3)
    for i, T_ in enumerate([T1, T2, T3, T4]):
        r_ = corners[i] - center
        F_ = cable_force(i, T_)
        M_sum += np.cross(r_, F_)
    eq.extend(M_sum)

    return np.array(eq)

def solve_cspr_pose(L_list, x_guess=None):
    if x_guess is None:
        x_guess = [1.0, 1.0, 1.0, 0, 0, 0, 5, 5, 5, 5]
    sol = fsolve(equilibrium_equations, x0=x_guess, args=(L_list,),
                 full_output=True, xtol=1e-7, maxfev=1000)
    x_res = sol[0]
    info, ier, msg = sol[1], sol[2], sol[3]
    if ier != 1:
        print("[Warning] solve_cspr_pose did not converge:", msg)
    return x_res

def compute_pose_for_center(center, parent_sol=None):
    """
    Compute L_i = ||anchor_i - center||
    """
    L_list = np.linalg.norm(anchors_simple - center, axis=1)
    if parent_sol is None:
        anchor_center = np.mean(anchors_simple, axis=0)
        init_center = anchor_center + 0.60 * np.array([0, 0, 1])
        x_guess = [init_center[0], init_center[1], init_center[2],
                   0, 0, 0, 5, 5, 5, 5]
    else:
        x_guess = parent_sol.copy()
        x_guess[:3] = center

    sol = solve_cspr_pose(L_list, x_guess)
    return sol

def euler_from_matrix(R):
    pitch = math.asin(-R[2, 0])
    cp = math.cos(pitch)
    if abs(cp) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

############################################
# 4. Visualization with Z <= 1.1
############################################
def interactive_visualization():
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Ideal Coordinates")
    ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(-0.5, 2.5); ax1.set_zlim(0, 3)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Target Coordinates")
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(1, 3)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    for pt in anchors_simple:
        ax1.scatter(pt[0], pt[1], pt[2], c='red', s=50)
    for pt in target_points:
        ax2.scatter(pt[0], pt[1], pt[2], c='red', s=50)

    plt.subplots_adjust(left=0.1, bottom=0.25)
    ax_slider_x = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_slider_y = plt.axes([0.1, 0.10, 0.8, 0.03])
    ax_slider_z = plt.axes([0.1, 0.05, 0.8, 0.03])

    slider_x = Slider(ax_slider_x, 'Center X', 0.0, 2.0, valinit=1.0)
    slider_y = Slider(ax_slider_y, 'Center Y', 0.0, 2.0, valinit=1.0)
    # Key: Z range [0, 1.1] to prevent exceeding anchor height
    slider_z = Slider(ax_slider_z, 'Center Z', 0.0, 1.1, valinit=1.1)

    center_init = np.array([slider_x.val, slider_y.val, slider_z.val])
    sol_ideal = compute_pose_for_center(center_init, None)
    last_sol = sol_ideal.copy()

    corners_i = platform_corners_world(sol_ideal)
    line_ideal, = ax1.plot(
        np.append(corners_i[:, 0], corners_i[0, 0]),
        np.append(corners_i[:, 1], corners_i[0, 1]),
        np.append(corners_i[:, 2], corners_i[0, 2]),
        'b-', lw=2
    )

    # Prepare similarity transformation: T_trans
    # Extract scale + pure rotation for orientation, or directly use 4x4 multiplication
    def transform_solution(sol):
        # 1) Center transformation
        center_hom = np.array([sol[0], sol[1], sol[2], 1.0])
        center_new_hom = T_trans @ center_hom
        center_new = center_new_hom[:3]

        # 2) Rotation transformation
        # T_trans[:3, :3] = scale * R_trans
        # => First extract the scale
        R_scale = T_trans[:3, :3]
        det_val = np.linalg.det(R_scale)
        if det_val < 0:
            det_val = -det_val
        scale_factor = det_val ** (1/3)
        R_trans_pure = R_scale / scale_factor

        # R_new = R_trans_pure * R_simple
        R_simple = rot_z(sol[5]) @ rot_y(sol[4]) @ rot_x(sol[3])
        R_new = R_trans_pure @ R_simple

        roll_r, pitch_r, yaw_r = euler_from_matrix(R_new)
        tensions = sol[6:]
        return np.hstack([center_new, [roll_r, pitch_r, yaw_r], tensions])

    sol_trans = transform_solution(sol_ideal)
    corners_r = platform_corners_world(sol_trans)
    line_real, = ax2.plot(
        np.append(corners_r[:, 0], corners_r[0, 0]),
        np.append(corners_r[:, 1], corners_r[0, 1]),
        np.append(corners_r[:, 2], corners_r[0, 2]),
        'g-', lw=2
    )

    txt_ideal = ax1.text2D(0.05, 0.95, "", transform=ax1.transAxes, color='blue')
    txt_real  = ax2.text2D(0.05, 0.95, "", transform=ax2.transAxes, color='green')

    def update(val):
        nonlocal last_sol
        c_new = np.array([slider_x.val, slider_y.val, slider_z.val])
        sol_new = compute_pose_for_center(c_new, last_sol)
        last_sol = sol_new.copy()

        # Update Ideal plot
        corners_i = platform_corners_world(sol_new)
        line_ideal.set_data(
            np.append(corners_i[:, 0], corners_i[0, 0]),
            np.append(corners_i[:, 1], corners_i[0, 1])
        )
        line_ideal.set_3d_properties(np.append(corners_i[:, 2], corners_i[0, 2]))

        txt_ideal.set_text(
            ("Ideal:\n"
             "X={:.2f}, Y={:.2f}, Z={:.2f}\n"
             "roll={:.1f}, pitch={:.1f}, yaw={:.1f}\n"
             "T1={:.1f}, T2={:.1f}, T3={:.1f}, T4={:.1f}"
            ).format(sol_new[0], sol_new[1], sol_new[2],
                     sol_new[3], sol_new[4], sol_new[5],
                     sol_new[6], sol_new[7], sol_new[8], sol_new[9])
        )

        # Transform to real coordinates
        sol_r = transform_solution(sol_new)
        corners_r = platform_corners_world(sol_r)
        line_real.set_data(
            np.append(corners_r[:, 0], corners_r[0, 0]),
            np.append(corners_r[:, 1], corners_r[0, 1])
        )
        line_real.set_3d_properties(np.append(corners_r[:, 2], corners_r[0, 2]))

        txt_real.set_text(
            ("Real:\n"
             "X={:.2f}, Y={:.2f}, Z={:.2f}\n"
             "roll={:.1f}, pitch={:.1f}, yaw={:.1f}\n"
             "T1={:.1f}, T2={:.1f}, T3={:.1f}, T4={:.1f}"
            ).format(sol_r[0], sol_r[1], sol_r[2],
                     sol_r[3], sol_r[4], sol_r[5],
                     sol_r[6], sol_r[7], sol_r[8], sol_r[9])
        )

        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    plt.show()

if __name__ == "__main__":
    interactive_visualization()
