import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

MAX_MOVEMENT_LINES = 50  # Only read the first 50 lines for "Without Model"
MAX_OUTLIER_ERROR = 15  # Skip records if error exceeds this threshold


def distance_3d(a, b):
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    dz = a['z'] - b['z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def load_movement_records_first_50(movement_file, filter_outliers=True):
    """
    Load the first 50 lines from movement_records.json (if fewer than 50, load all).
    Only consider the "final" error:
      - If attempts == 4, treat attempts as 3 and use the distance from the second-to-last attempt's robot_after to target as error.
      - Otherwise, use the error from the last attempt as originally defined.
    Returns:
      - movement_errors: the final error used
      - movement_attempts: the final number of attempts used
      - movement_after_xs: the final robot_after.x used
    """
    if not os.path.exists(movement_file):
        print(f"[load_movement_records_first_50] File {movement_file} does not exist.")
        return [], [], []

    errors = []
    attempt_counts = []
    after_xs = []

    count = 0
    with open(movement_file, 'r') as fin:
        for line in fin:
            if count >= MAX_MOVEMENT_LINES:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                if not all(k in record for k in ("robot_before", "target", "attempts")):
                    continue

                attempts = record["attempts"]
                if not attempts:
                    continue

                n_att = len(attempts)
                if n_att == 4:
                    used_attempt_count = 3
                    second_last_robot_after = attempts[-2].get("robot_after", {})
                    target = record["target"]
                    err = distance_3d(second_last_robot_after, target)
                    ra_x = second_last_robot_after.get("x", float('nan'))
                else:
                    used_attempt_count = n_att
                    final_attempt = attempts[-1]
                    if "error" not in final_attempt:
                        continue
                    err = final_attempt["error"]
                    ra_x = final_attempt.get("robot_after", {}).get("x", float('nan'))

                if filter_outliers and err > MAX_OUTLIER_ERROR:
                    continue

                errors.append(err)
                attempt_counts.append(used_attempt_count)
                after_xs.append(ra_x)

                count += 1

            except json.JSONDecodeError:
                print("[load_movement_records_first_50] JSON decode error, skipping line.")
                continue
            except KeyError as e:
                print(f"[load_movement_records_first_50] Missing key {e}, skipping line.")
                continue
            except Exception as e:
                print(f"[load_movement_records_first_50] An error occurred: {e}, skipping line.")
                continue

    return errors, attempt_counts, after_xs


def load_movement_records_after_50(movement_file, filter_outliers=True):
    """
    Load all lines from movement_records.json starting from line 51.
    Only consider the "final" error:
      - If attempts == 4, treat attempts as 3 and use the distance from the second-to-last attempt's robot_after to target as error.
      - Otherwise, use the error from the last attempt as originally defined.
    Returns:
      - errors: the final error used
      - attempt_counts: the final number of attempts used
      - after_xs: the final robot_after.x used
    """
    if not os.path.exists(movement_file):
        print(f"[load_movement_records_after_50] File {movement_file} does not exist.")
        return [], [], []

    errors = []
    attempt_counts = []
    after_xs = []

    with open(movement_file, 'r') as fin:
        for idx, line in enumerate(fin):
            if idx < 50:
                continue

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "attempts" not in record or "target" not in record:
                    continue

                attempts = record["attempts"]
                if not attempts:
                    continue

                n_att = len(attempts)
                if n_att == 4:
                    used_attempt_count = 3
                    second_last_robot_after = attempts[-2].get("robot_after", {})
                    target = record["target"]
                    err = distance_3d(second_last_robot_after, target)
                    ra_x = second_last_robot_after.get("x", float('nan'))
                else:
                    used_attempt_count = n_att
                    final_attempt = attempts[-1]
                    if "error" not in final_attempt:
                        continue
                    err = final_attempt["error"]
                    ra_x = final_attempt.get("robot_after", {}).get("x", float('nan'))

                if filter_outliers and err > MAX_OUTLIER_ERROR:
                    continue

                errors.append(err)
                attempt_counts.append(used_attempt_count)
                after_xs.append(ra_x)

            except json.JSONDecodeError:
                print("[load_movement_records_after_50] JSON decode error, skipping line.")
                continue
            except KeyError as e:
                print(f"[load_movement_records_after_50] Missing key {e}, skipping line.")
                continue
            except Exception as e:
                print(f"[load_movement_records_after_50] An error occurred: {e}, skipping line.")
                continue

    return errors, attempt_counts, after_xs


def plot_scatter_with_trend(
        mov_after_x_arr, mov_err_arr, valid_mov_mask,
        pred_after_x_arr, pred_err_arr, valid_pred_mask
):
    """
    Use polynomial regression (degree=2) to plot a scatter plot (X = robot_after.x, Y = error),
    and display the trend lines for "Without Model" (blue) and "With Model" (red) data.
    """
    plt.figure(figsize=(10, 6))

    # ========== Without Model scatter ==========
    x_m = mov_after_x_arr[valid_mov_mask]
    y_m = mov_err_arr[valid_mov_mask]

    plt.scatter(x_m, y_m, color='blue', alpha=0.6, label='Without Model Data')
    if len(x_m) > 1:
        # Sort x and y arrays to ensure x is increasing
        sort_idx = np.argsort(x_m)
        x_m_sorted = x_m[sort_idx]
        y_m_sorted = y_m[sort_idx]

        # Sample uniformly over the x range for plotting the fitted curve
        x_fit = np.linspace(x_m_sorted[0], x_m_sorted[-1], 100)

        # =============== Use polynomial regression instead of spline interpolation ===============
        # Here we choose a quadratic polynomial; degree can be adjusted to 3, 4, etc.
        degree = 2
        coeffs_m = np.polyfit(x_m_sorted, y_m_sorted, deg=degree)
        poly_m = np.poly1d(coeffs_m)
        y_fit = poly_m(x_fit)
        # =========================================================

        plt.plot(x_fit, y_fit, color='blue', linewidth=2,
                 label=f'Without Model Polynomial Fit (deg={degree})')

    # ========== With Model scatter ==========
    x_p = pred_after_x_arr[valid_pred_mask]
    y_p = pred_err_arr[valid_pred_mask]

    plt.scatter(x_p, y_p, color='red', alpha=0.6, label='With Model Data')
    if len(x_p) > 1:
        sort_idx_p = np.argsort(x_p)
        x_p_sorted = x_p[sort_idx_p]
        y_p_sorted = y_p[sort_idx_p]

        x_fit_p = np.linspace(x_p_sorted[0], x_p_sorted[-1], 100)

        # =============== Use polynomial regression instead of spline interpolation ===============
        degree_p = 2
        coeffs_p = np.polyfit(x_p_sorted, y_p_sorted, deg=degree_p)
        poly_p = np.poly1d(coeffs_p)
        y_fit_p = poly_p(x_fit_p)
        # =========================================================

        plt.plot(
            x_fit_p, y_fit_p,
            color='red', linewidth=2,
            label=f'With Model Polynomial Fit (deg={degree_p})'
        )

    plt.title("Error vs. Robot_After.x")
    plt.xlabel("Robot_After.x")
    plt.ylabel("Error(cm)")
    plt.grid(True)
    plt.legend()
    plt.show()


def compare_data(
        movement_errors, movement_attempts, movement_after_xs,
        prediction_errors, prediction_attempts, prediction_after_xs
):
    """
    1) Scale up the x coordinate by 3 times.
    2) Perform statistics and visualization for Error and Attempts.
    3) Plot scatter plots with (polynomial) fitted curves.
    4) Perform overlaid comparisons for three groups (1-50, 51-100, 101-150) if data is sufficient.
    """
    # ------- Convert to numpy arrays -------
    mov_err_arr = np.array(movement_errors)
    mov_att_arr = np.array(movement_attempts)
    mov_after_x_arr = np.array(movement_after_xs)

    pred_err_arr = np.array(prediction_errors)
    pred_att_arr = np.array(prediction_attempts)
    pred_after_x_arr = np.array(prediction_after_xs)

    # ------- Scale up x coordinate by 3 times -------
    mov_after_x_arr = mov_after_x_arr * 3
    pred_after_x_arr = pred_after_x_arr * 3

    # =========== Error Comparison ===========
    avg_mov_err = np.mean(mov_err_arr) if len(mov_err_arr) else 0.0
    avg_pred_err = np.mean(pred_err_arr) if len(pred_err_arr) else 0.0

    print("===== Error Averages (Filtered) =====")
    print(f"Without Model error avg: {avg_mov_err:.3f}  (count={len(mov_err_arr)})")
    print(f"With Model error avg:    {avg_pred_err:.3f} (count={len(pred_err_arr)})")
    print("=====================================")

    # Plot error bar charts separately
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    indices_mov = np.arange(len(mov_err_arr))
    plt.bar(indices_mov, mov_err_arr, color='blue', label='Without Model Error')
    plt.title("Without Model Error")
    plt.xlabel("Index")
    plt.ylabel("Error(cm)")
    plt.grid(True, axis='y')
    plt.legend()

    plt.subplot(1, 2, 2)
    indices_pred = np.arange(len(pred_err_arr))
    plt.bar(indices_pred, pred_err_arr, color='red', label='With Model Error')
    plt.title("With Model Error")
    plt.xlabel("Index")
    plt.ylabel("Error(cm)")
    plt.grid(True, axis='y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Overlaid errors (compare first min_len_e records)
    min_len_e = min(len(mov_err_arr), len(pred_err_arr))
    if min_len_e > 0:
        overlay_indices = np.arange(min_len_e)
        width = 0.35

        plt.figure(figsize=(14, 6))
        plt.bar(overlay_indices - width / 2, mov_err_arr[:min_len_e], width,
                color='blue', label='Without Model')
        plt.bar(overlay_indices + width / 2, pred_err_arr[:min_len_e], width,
                color='red', label='With Model')
        plt.title("Without Model vs. With Model Error (Overlaid)")
        plt.xlabel("Index")
        plt.ylabel("Error(cm)")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()

        diff_err = mov_err_arr[:min_len_e] - pred_err_arr[:min_len_e]
        plt.figure(figsize=(14, 6))
        plt.bar(overlay_indices, diff_err, color='green',
                label='Diff (Without Model - With Model)')
        plt.title("Difference in Error (Without Model - With Model)")
        plt.xlabel("Index")
        plt.ylabel("Error Difference")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()

        avg_diff_err = np.mean(diff_err)
        print(
            f"Average difference (Without Model - With Model) over first {min_len_e} matched records: {avg_diff_err:.3f}")

    # =========== Attempts Comparison ===========
    avg_mov_att = np.mean(mov_att_arr) if len(mov_att_arr) else 0.0
    avg_pred_att = np.mean(pred_att_arr) if len(pred_att_arr) else 0.0

    print("\n===== Attempts Averages (Filtered) =====")
    print(f"Without Model attempts avg: {avg_mov_att:.2f} (count={len(mov_att_arr)})")
    print(f"With Model attempts avg:    {avg_pred_att:.2f} (count={len(pred_att_arr)})")
    print("========================================")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    indices_mov_att = np.arange(len(mov_att_arr))
    plt.bar(indices_mov_att, mov_att_arr, color='blue', label='Without Model Attempts')
    plt.title("Without Model Attempts (Filtered)")
    plt.xlabel("Index")
    plt.ylabel("Attempts Count")
    plt.grid(True, axis='y')
    plt.legend()

    plt.subplot(1, 2, 2)
    indices_pred_att = np.arange(len(pred_att_arr))
    plt.bar(indices_pred_att, pred_att_arr, color='red', label='With Model Attempts')
    plt.title("With Model Attempts (Filtered)")
    plt.xlabel("Index")
    plt.ylabel("Attempts Count")
    plt.grid(True, axis='y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Overlaid attempts comparison
    min_len_a = min(len(mov_att_arr), len(pred_att_arr))
    if min_len_a > 0:
        overlay_indices_att = np.arange(min_len_a)
        width_att = 0.35

        plt.figure(figsize=(14, 6))
        plt.bar(overlay_indices_att - width_att / 2, mov_att_arr[:min_len_a], width_att,
                color='blue', label='Without Model')
        plt.bar(overlay_indices_att + width_att / 2, pred_att_arr[:min_len_a], width_att,
                color='red', label='With Model')
        plt.title("Without Model vs. With Model Attempts (Overlaid)")
        plt.xlabel("Index")
        plt.ylabel("Attempts Count")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()

        diff_att = mov_att_arr[:min_len_a] - pred_att_arr[:min_len_a]
        plt.figure(figsize=(14, 6))
        plt.bar(overlay_indices_att, diff_att, color='green',
                label='Diff (Without Model - With Model)')
        plt.title("Difference in Attempts (Without Model - With Model)")
        plt.xlabel("Index")
        plt.ylabel("Attempts Difference")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()

        avg_diff_att = np.mean(diff_att)
        print(
            f"Average difference (Without Model - With Model) attempts over first {min_len_a} matched records: {avg_diff_att:.2f}")

    # =========== Scatter Plot + Trendline (Quadratic Polynomial Fit) ===========
    valid_mov_mask = ~np.isnan(mov_after_x_arr)
    valid_pred_mask = ~np.isnan(pred_after_x_arr)

    plot_scatter_with_trend(
        mov_after_x_arr, mov_err_arr, valid_mov_mask,
        pred_after_x_arr, pred_err_arr, valid_pred_mask
    )

    # =========== Three-group overlaid error comparison (optional) ===========
    if len(mov_err_arr) >= 50 and len(pred_err_arr) >= 100:
        group1 = mov_err_arr[:50]
        group2 = pred_err_arr[:50]
        group3 = pred_err_arr[50:100]

        mean_group1 = np.mean(group1)
        mean_group2 = np.mean(group2)
        mean_group3 = np.mean(group3)

        print("\n===== Overlaid Error Means for Three Groups =====")
        print(f"Without Model:                   {mean_group1:.3f} (count={len(group1)})")
        print(f"Trained with 50 Movement Records: {mean_group2:.3f} (count={len(group2)})")
        print(f"Trained with 100 Movement Records:{mean_group3:.3f} (count={len(group3)})")
        print("===================================================")

        indices = np.arange(50)
        bar_width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(indices - bar_width, group1, bar_width, color='blue', label='Without Model')
        plt.bar(indices, group2, bar_width, color='red', label='Trained with 50 Movement Records')
        plt.bar(indices + bar_width, group3, bar_width, color='green', label='Trained with 100 Movement Records')
        plt.title("Overlaid Error Comparison: Without Model vs. Trained with 50 vs. 100 Movement Records")
        plt.xlabel("Index (within each group)")
        plt.ylabel("Error (cm)")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()
    else:
        print("Not enough records to perform three-group overlaid error comparison.")

    # =========== Three-group overlaid attempts comparison (optional) ===========
    if len(mov_att_arr) >= 50 and len(pred_att_arr) >= 100:
        group1_att = mov_att_arr[:50]
        group2_att = pred_att_arr[:50]
        group3_att = pred_att_arr[50:100]

        mean_group1_att = np.mean(group1_att)
        mean_group2_att = np.mean(group2_att)
        mean_group3_att = np.mean(group3_att)

        print("\n===== Overlaid Attempts Means for Three Groups =====")
        print(f"Without Model:                   {mean_group1_att:.3f} (count={len(group1_att)})")
        print(f"Trained with 50 Movement Records: {mean_group2_att:.3f} (count={len(group2_att)})")
        print(f"Trained with 100 Movement Records:{mean_group3_att:.3f} (count={len(group3_att)})")
        print("======================================================")

        indices = np.arange(50)
        bar_width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(indices - bar_width, group1_att, bar_width, color='blue', label='Without Model')
        plt.bar(indices, group2_att, bar_width, color='red', label='Trained with 50 Movement Records')
        plt.bar(indices + bar_width, group3_att, bar_width, color='green', label='Trained with 100 Movement Records')
        plt.title("Overlaid Attempts Comparison: Without Model vs. Trained with 50 vs. 100 Movement Records")
        plt.xlabel("Index (within each group)")
        plt.ylabel("Attempts Count")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()
    else:
        print("Not enough records to perform three-group overlaid attempts comparison.")


def main():
    movement_file = "results/家中大平衡/movement_records.json"

    # 1) Load the first 50 lines of data (equivalent to "Without Model")
    movement_errors, movement_attempts, movement_after_xs = load_movement_records_first_50(
        movement_file, filter_outliers=True
    )

    # 2) Load data from line 51 onward (equivalent to "With Model")
    prediction_errors, prediction_attempts, prediction_after_xs = load_movement_records_after_50(
        movement_file, filter_outliers=True
    )

    # 3) Data comparison and visualization
    compare_data(
        movement_errors, movement_attempts, movement_after_xs,
        prediction_errors, prediction_attempts, prediction_after_xs
    )


if __name__ == "__main__":
    main()
