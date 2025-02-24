import os
import json
import time
import math
import numpy as np
import requests

import findGreen  # Assumed module for detecting the distance and angle of the green target center

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

################################################################################
# Config
################################################################################
SERVER_URL = "http://10.0.0.200:5000"

MOVEMENT_RECORDS_FILE = "movement_records.json"
FINETUNE_RECORDS_FILE = "finetune_records.json"

MAIN_MODEL_FILE = "NN_model.h5"           # Main model
FINETUNE_MODEL_FILE = "FinetuneModel.h5"    # Finetune model

SCALERS_MAIN = "scalers.pkl"         # Scaler for main model
SCALERS_FINETUNE = "scalers_ft.pkl"   # Scaler for finetune model

MAX_ERROR = 5.0
MAX_RETRIES = 3
GREEN_DIST_THRESH = 300.0  # If <300, considered close enough
FIRST_TIME_STEPS = 10
LATER_TIME_STEPS = 3

# -------------------------------
# New configuration: Targets for the initial fine-tuning process (offsets relative to central_coord).
# The list can contain one or more targets (e.g., two) to control the "shooting points" during fine-tuning.
# -------------------------------
INITIAL_FINETUNE_RELATIVE_TARGETS = [
    {'x': 0,  'y':  0,  'z':  0},  # Second shooting point (optional, can be removed if not needed)
]

# -------------------------------
# New configuration: Used to determine whether the current target in the main loop is a shooting point.
# If the offset of the current target relative to central_coord matches any element in the list (within tolerance),
# it is considered a shooting point.
# -------------------------------
SHOOTING_OFFSETS = [
    {'x': 0,  'y':  0,  'z':  0}
]

# This is the fixed target list used in the main loop.
# If you need to change the number of shooting points that appear, you can modify the RELATIVE_TARGETS below.
RELATIVE_TARGETS = [
    {'x': -4, 'y':  0, 'z':  0},
    {'x': 0, 'y':  0, 'z':  0},
    {'x': -10, 'y':  0, 'z':  0},
    {'x': 0, 'y':  0, 'z':  0},
    {'x': 5, 'y':  0, 'z':  0},
    {'x': 0,  'y':  0, 'z':  0},
    {'x': 10, 'y':  0, 'z':  0},
    {'x': 0,  'y':  0, 'z':  0},
    {'x': 15, 'y':  0, 'z':  0},
    {'x': 0,   'y':  0, 'z':  0},
]

# 1) Center point
central_coord = [-30, -10, 217]
FIXED_TARGETS = [
    {
        'x': central_coord[0] + rel['x'],
        'y': central_coord[1] + rel['y'],
        'z': central_coord[2] + rel['z'],
    }
    for rel in RELATIVE_TARGETS
]

################################################################################
# Globals
################################################################################
movement_records = []
main_model = None
scaler_X = None
scaler_y = None

finetune_records = []
finetune_model = None
ft_scaler_X = None
ft_scaler_y = None

first_finetune_done = False
second_finetune_done = False

################################################################################
# 1) Main model training + prediction: 8-dimensional input (before.x, y, z, pitch, roll; after.x, y, z)
################################################################################
def load_first_50_records():
    """
    Read records line by line from movement_records.json.
    For each line, directly use the initial pose data in robot_before (including pitch and roll),
    and reconstruct the record by rearranging the key order of robot_before and robot_after to:
      [pitch, roll, x, y, z]
    The record format will be similar to:
    {
      "robot_before": {"pitch": -31.293370651670593, "roll": -10.068181261459337,
                       "x": 20.454143657410437, "y": -11.153293331440826, "z": 190.41774119458486},
      "target": {"x": 5, "y": -20, "z": 200},
      "attempts": [
         {"robot_after": {"pitch": -20.409545093086127, "roll": -11.719561652568412,
                           "x": 14.475813503678204, "y": -17.429850186598205, "z": 200.6205796783672},
          "error": 9.837775701703196},
         ...
      ]
    }
    """
    global movement_records
    movement_records.clear()

    if not os.path.exists(MOVEMENT_RECORDS_FILE):
        print(f"[load_first_50_records] {MOVEMENT_RECORDS_FILE} not found.")
        return

    with open(MOVEMENT_RECORDS_FILE, "r") as fin:
        all_lines = [line.strip() for line in fin if line.strip()]

    records = []
    for idx, line in enumerate(all_lines):
        try:
            rec = json.loads(line)
            records.append(rec)
        except json.JSONDecodeError:
            print(f"[load_first_50_records] JSON decode error at line {idx + 1}, skipping.")
            continue

    count = 0
    for i, rec in enumerate(records):
        if count >= 50:
            break

        # Check if the current record contains the necessary fields
        if not all(k in rec for k in ("robot_before", "target", "attempts")):
            print(f"[load_first_50_records] Missing keys in record {i + 1}, skipping.")
            continue

        # Directly use the robot_before data from the record (initial pose) and rearrange the key order
        rb = rec["robot_before"]
        new_robot_before = {
            "pitch": rb.get("pitch", 0.0),
            "roll": rb.get("roll", 0.0),
            "x": rb.get("x", 0.0),
            "y": rb.get("y", 0.0),
            "z": rb.get("z", 0.0)
        }

        # For robot_after, take the robot_after data from the first attempt in the current record and rearrange the key order
        attempts_curr = rec["attempts"]
        if not attempts_curr:
            print(f"[load_first_50_records] No attempts in record {i + 1}, skipping.")
            continue
        first_att = attempts_curr[0]
        if "robot_after" not in first_att:
            print(f"[load_first_50_records] Missing robot_after in record {i + 1}, skipping.")
            continue

        ra = first_att["robot_after"]
        new_robot_after = {
            "pitch": ra.get("pitch", 0.0),
            "roll": ra.get("roll", 0.0),
            "x": ra.get("x", 0.0),
            "y": ra.get("y", 0.0),
            "z": ra.get("z", 0.0)
        }

        new_rec = {
            "robot_before": new_robot_before,
            "robot_after": new_robot_after,
            "target": rec["target"]
        }
        movement_records.append(new_rec)
        count += 1

    print(f"[load_first_50_records] loaded={count} records into movement_records.")


def train_main_model(print_data=False):
    """
    Train the main model:
      Input (X) = 8 dimensions => [rb.x, rb.y, rb.z, rb.pitch, rb.roll, ra.x, ra.y, ra.z]
      Output (y) = [target.x, target.y, target.z]
    """
    global main_model, scaler_X, scaler_y
    if len(movement_records) < 5:
        print("[train_main_model] not enough data (<5). skip.")
        return

    X_list = []
    y_list = []

    for rec in movement_records:
        rb = rec["robot_before"]  # dict: {pitch, roll, x, y, z}
        ra = rec["robot_after"]   # dict: {pitch, roll, x, y, z} (but only x, y, z are used for training)
        tg = rec["target"]        # dict: {x, y, z}

        # Note: The order here is [x, y, z, pitch, roll, x, y, z]
        feats = [
            rb["x"], rb["y"], rb["z"],
            rb["pitch"], rb["roll"],
            ra["x"], ra["y"], ra["z"]
        ]
        lbls = [tg["x"], tg["y"], tg["z"]]

        X_list.append(feats)
        y_list.append(lbls)

    if print_data:
        print("\n[train_main_model] ==> Training Data Samples:")
        for i, (feats, lbls) in enumerate(zip(X_list, y_list), start=1):
            print(f" Sample #{i}: X={feats}, y={lbls}")

    X = np.array(X_list)
    y = np.array(y_list)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_sca = scaler_X.fit_transform(X)
    y_sca = scaler_y.fit_transform(y)

    main_model = Sequential()
    main_model.add(Dense(64, activation="relu", input_dim=8))
    main_model.add(Dense(32, activation="relu"))
    main_model.add(Dense(3, activation="linear"))
    main_model.compile(optimizer=Adam(0.001), loss="mse")

    print(f"[train_main_model] training with {len(X_list)} records...")
    main_model.fit(X_sca, y_sca, epochs=50, batch_size=16, verbose=0)
    print("[train_main_model] done")

    # Evaluate
    y_pred_sca = main_model.predict(X_sca)
    y_pred = scaler_y.inverse_transform(y_pred_sca)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"[train_main_model] MSE={mse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    main_model.save(MAIN_MODEL_FILE)
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, SCALERS_MAIN)
    print("[train_main_model] saved main model & scalers.")


def predict_main_model(robot_before, robot_after):
    """
    Use the main model to predict the "target".
    Input 8 dimensions:
      [before.x, before.y, before.z, before.pitch, before.roll, after.x, after.y, after.z]
    """
    global main_model, scaler_X, scaler_y
    if main_model is None:
        print("[predict_main_model] no main_model => None")
        return None

    feats = [
        robot_before["x"], robot_before["y"], robot_before["z"],
        robot_before["pitch"], robot_before["roll"],
        robot_after["x"], robot_after["y"], robot_after["z"]
    ]
    X_in = np.array([feats])
    X_sca = scaler_X.transform(X_in)
    y_sca = main_model.predict(X_sca)
    y_pred = scaler_y.inverse_transform(y_sca)

    return {
        "x": float(y_pred[0][0]),
        "y": float(y_pred[0][1]),
        "z": float(y_pred[0][2])
    }


################################################################################
# 2) Finetune model: 7-dimensional (x, y, z, pitch, roll, dist, angle) => (dx, dy, dz)
################################################################################
def load_finetune_records():
    """
    Load finetune_records.json. Each record is in the form:
    {
      "before": {
          "robot": { "pitch":..., "roll":..., "x":..., "y":..., "z":... },
          "dist": float,
          "angle": float
      },
      "command": { "dx":..., "dy":..., "dz":... },
      "after": {
          "robot": { "pitch":..., "roll":..., "x":..., "y":..., "z":... },
          "dist": float,
          "angle": float
      }
    }
    During loading, reconstruct the robot data in both "before" and "after" to the same format.
    """
    global finetune_records
    finetune_records.clear()
    if not os.path.exists(FINETUNE_RECORDS_FILE):
        print(f"[load_finetune_records] {FINETUNE_RECORDS_FILE} not found.")
        return

    with open(FINETUNE_RECORDS_FILE, "r") as fin:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if all(k in rec for k in ("before", "command", "after")):
                    b = rec["before"]
                    c = rec["command"]
                    a = rec["after"]
                    if all(k in b for k in ("robot", "dist", "angle")) and \
                       all(k in c for k in ("dx", "dy", "dz")) and \
                       all(k in a for k in ("robot", "dist", "angle")):
                        # Reconstruct the robot data in "before" and "after" to the format {pitch, roll, x, y, z}
                        b_robot = b["robot"]
                        a_robot = a["robot"]
                        new_b_robot = {
                            "pitch": b_robot.get("pitch", 0.0),
                            "roll": b_robot.get("roll", 0.0),
                            "x": b_robot.get("x", 0.0),
                            "y": b_robot.get("y", 0.0),
                            "z": b_robot.get("z", 0.0)
                        }
                        new_a_robot = {
                            "pitch": a_robot.get("pitch", 0.0),
                            "roll": a_robot.get("roll", 0.0),
                            "x": a_robot.get("x", 0.0),
                            "y": a_robot.get("y", 0.0),
                            "z": a_robot.get("z", 0.0)
                        }
                        rec["before"]["robot"] = new_b_robot
                        rec["after"]["robot"] = new_a_robot
                        finetune_records.append(rec)
                    else:
                        print(f"[load_finetune_records] Missing keys in record {idx}, skipping.")
                else:
                    print(f"[load_finetune_records] Missing keys in record {idx}, skipping.")
            except json.JSONDecodeError:
                print(f"[load_finetune_records] JSON decode error at line {idx}, skipping.")
                continue
    print(f"[load_finetune_records] total={len(finetune_records)}")


def train_finetune_model():
    """
    Finetune model:
      Input (X) = 7 dimensions => [rb.x, rb.y, rb.z, rb.pitch, rb.roll, dist, angle]
      Output (y) = [dx, dy, dz]
    """
    global finetune_model, ft_scaler_X, ft_scaler_y
    if len(finetune_records) < 5:
        print("[train_finetune_model] not enough data (<5). skip.")
        return

    X_list = []
    y_list = []
    for rec in finetune_records:
        b = rec["before"]
        c = rec["command"]
        rb = b["robot"]

        rb_pitch = rb.get("pitch", 0.0)
        rb_roll  = rb.get("roll", 0.0)
        dist_val = b["dist"]
        angle_val = b["angle"]

        feats = [
            rb["x"], rb["y"], rb["z"],
            rb_pitch, rb_roll,
            dist_val, angle_val
        ]
        outs = [c["dx"], c["dy"], c["dz"]]

        X_list.append(feats)
        y_list.append(outs)

    X = np.array(X_list)
    y = np.array(y_list)

    if X.size == 0 or y.size == 0:
        print("[train_finetune_model] Empty training data.")
        return

    ft_scaler_X = StandardScaler()
    ft_scaler_y = StandardScaler()

    X_sca = ft_scaler_X.fit_transform(X)
    y_sca = ft_scaler_y.fit_transform(y)

    finetune_model = Sequential()
    finetune_model.add(Dense(32, activation="relu", input_dim=7))
    finetune_model.add(Dense(16, activation="relu"))
    finetune_model.add(Dense(3, activation="linear"))
    finetune_model.compile(optimizer=Adam(0.001), loss="mse")

    print(f"[train_finetune_model] training with {len(X_list)} records...")
    finetune_model.fit(X_sca, y_sca, epochs=30, batch_size=8, verbose=0)
    print("[train_finetune_model] done")

    # Evaluate
    y_pred_sca = finetune_model.predict(X_sca)
    y_pred = ft_scaler_y.inverse_transform(y_pred_sca)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"[train_finetune_model] MSE={mse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    finetune_model.save(FINETUNE_MODEL_FILE)
    joblib.dump({'scaler_X': ft_scaler_X, 'scaler_y': ft_scaler_y}, SCALERS_FINETUNE)
    print("[train_finetune_model] finetune model & scalers saved.")

    # If needed, print some training data samples
    print("[train_finetune_model] Sample Training Data:")
    for i in range(min(5, len(X_list))):
        print(f" Sample #{i + 1}: X={X_list[i]}, y={y_list[i]}")


def predict_finetune_model(robot_coord, dist_val, angle_val):
    """
    Predict => (dx, dy, dz)
    Input 7 dimensions => [x, y, z, pitch, roll, dist, angle]
    """
    if finetune_model is None:
        print("[predict_finetune_model] no finetune_model => fallback 0 offset")
        return (0.0, 0.0, 0.0)

    rb_pitch = robot_coord.get("pitch", 0.0)
    rb_roll  = robot_coord.get("roll", 0.0)

    feats = [
        robot_coord["x"], robot_coord["y"], robot_coord["z"],
        rb_pitch, rb_roll,
        dist_val, angle_val
    ]
    X_in = np.array([feats])
    X_sca = ft_scaler_X.transform(X_in)
    y_sca = finetune_model.predict(X_sca)
    y_pred = ft_scaler_y.inverse_transform(y_sca)
    return (float(y_pred[0][0]), float(y_pred[0][1]), float(y_pred[0][2]))


def save_finetune_record(before_dict, command_dict, after_dict):
    """
    Save a finetune record:
    {
      "before": {
          "robot": { "pitch":..., "roll":..., "x":..., "y":..., "z":... },
          "dist": float,
          "angle": float
      },
      "command": { "dx":..., "dy":..., "dz":... },
      "after": {
          "robot": { "pitch":..., "roll":..., "x":..., "y":..., "z":... },
          "dist": float,
          "angle": float
      }
    }
    Before saving, reconstruct the robot data to ensure the key order is pitch, roll, x, y, z.
    """
    # Reconstruct the robot data in before and after
    before_robot = before_dict.get("robot", {})
    after_robot = after_dict.get("robot", {})
    before_dict["robot"] = {
        "pitch": before_robot.get("pitch", 0.0),
        "roll": before_robot.get("roll", 0.0),
        "x": before_robot.get("x", 0.0),
        "y": before_robot.get("y", 0.0),
        "z": before_robot.get("z", 0.0)
    }
    after_dict["robot"] = {
        "pitch": after_robot.get("pitch", 0.0),
        "roll": after_robot.get("roll", 0.0),
        "x": after_robot.get("x", 0.0),
        "y": after_robot.get("y", 0.0),
        "z": after_robot.get("z", 0.0)
    }
    try:
        with open(FINETUNE_RECORDS_FILE, "a") as fout:
            json.dump({
                "before": before_dict,
                "command": command_dict,
                "after": after_dict
            }, fout)
            fout.write("\n")
        print("[save_finetune_record] Record saved.")
    except Exception as e:
        print(f"[save_finetune_record] Exception: {e}")


################################################################################
# 3) Rule-based fine-tuning: 20 steps -> split into two phases (10 steps each), using only the pitch/roll from BEFORE
################################################################################
def naive_finetune_initial():
    """
    Initialize the fine-tuning process:
    - Sequentially move to the coordinates specified in INITIAL_FINETUNE_RELATIVE_TARGETS,
      perform fine-tuning for FIRST_TIME_STEPS steps at each coordinate, and record data.
    """
    print("[naive_finetune_initial] Starting initial fine-tuning process...")
    for target in INITIAL_FINETUNE_RELATIVE_TARGETS:
        do_finetune_rule_steps(target, steps=FIRST_TIME_STEPS)
    load_finetune_records()
    train_finetune_model()
    print("[naive_finetune_initial] Initial fine-tuning completed. Finetune model is ready.")


def do_finetune_rule_steps(relative_xyz, steps=10):
    """
    Execute a specified number of fine-tuning steps (rule-based) with hard-coded handling for dx, dy, dz.
    relative_xyz is the offset relative to central_coord (e.g., {7, -15, 10})
    """
    for step in range(steps):
        dist_info = findGreen.detect_green_center_distance_and_direction()
        if not dist_info:
            print("[do_finetune_rule_steps] can't detect green => break.")
            break
        dist_val, angle_val = dist_info

        cur_robot = get_robot_coordinates()
        if not cur_robot:
            print("[do_finetune_rule_steps] no coords => break.")
            break

        print(f"[do_finetune_rule_steps] target={relative_xyz}, step={step + 1}/{steps}, dist={dist_val:.1f}, angle={angle_val:.1f}")

        if dist_val < GREEN_DIST_THRESH:
            print("[do_finetune_rule_steps] dist < 300 => done early.")
            break

        # Assuming the image is flipped horizontally, adjust dx and dz (dy remains unchanged)
        if 180 <= angle_val < 270:  # Left Upper
            dx = +0.3
            dz = -0.3
        elif 270 <= angle_val < 360:  # Right Upper
            dx = -0.3
            dz = -0.3
        elif 0 <= angle_val < 90:  # Right Lower
            dx = -0.3
            dz = +0.3
        elif 90 <= angle_val < 180:  # Left Lower
            dx = +0.3
            dz = +0.3
        else:
            dx = 0.0
            dz = 0.0
        dy = 0.0  # Assume no adjustment in the vertical direction

        before_dict = {
            "robot": cur_robot.copy(),
            "dist": dist_val,
            "angle": angle_val
        }
        command_dict = {"dx": dx, "dy": dy, "dz": dz}

        new_target = {
            "x": cur_robot["x"] + dx,
            "y": cur_robot["y"] + dy,
            "z": cur_robot["z"] + dz
        }
        send_target_coordinates(new_target)
        wait_for_flags()

        new_robot = get_robot_coordinates()
        new_info = findGreen.detect_green_center_distance_and_direction()

        if not new_robot or not new_info:
            break
        ndist, nangle = new_info
        after_dict = {
            "robot": new_robot.copy(),
            "dist": ndist,
            "angle": nangle
        }
        save_finetune_record(before_dict, command_dict, after_dict)


################################################################################
# 4) Subsequent Finetune Loop
################################################################################
def finetune_loop_incremental(max_steps=LATER_TIME_STEPS):
    """
    For subsequent shooting points, use the finetune model to predict (dx, dy, dz).
    If the distance is still large, perform partial training and then continue.
    """
    step_count = 0
    while step_count < max_steps:
        dist_info = findGreen.detect_green_center_distance_and_direction()
        if not dist_info:
            print("[finetune_loop_incremental] can't detect green => break.")
            break
        dist_val, angle_val = dist_info
        cur_robot = get_robot_coordinates()
        if not cur_robot:
            break

        print(f"[finetune_loop_incremental] step={step_count + 1}/{max_steps}, dist={dist_val:.1f}, angle={angle_val:.1f}")
        if dist_val < GREEN_DIST_THRESH:
            print("[finetune_loop_incremental] dist < 300 => done.")
            break

        before_dict = {
            "robot": cur_robot.copy(),
            "dist": dist_val,
            "angle": angle_val
        }
        dx, dy, dz = predict_finetune_model(cur_robot, dist_val, angle_val)
        command_dict = {"dx": dx, "dy": dy, "dz": dz}

        new_target = {
            "x": cur_robot["x"] + dx,
            "y": cur_robot["y"] + dy,
            "z": cur_robot["z"] + dz
        }
        send_target_coordinates(new_target)
        wait_for_flags()

        new_robot = get_robot_coordinates()
        new_info = findGreen.detect_green_center_distance_and_direction()
        if not new_robot or not new_info:
            break
        nd, na = new_info
        after_dict = {
            "robot": new_robot.copy(),
            "dist": nd,
            "angle": na
        }
        save_finetune_record(before_dict, command_dict, after_dict)

        if nd >= GREEN_DIST_THRESH:
            print("[finetune_loop_incremental] partial train => continue")
            load_finetune_records()
            train_finetune_model()
            # Continue
        else:
            print("[finetune_loop_incremental] dist < 300 => done.")
            break
        step_count += 1
    print("[finetune_loop_incremental] done or max steps used.")


################################################################################
# 5) Interaction with the Server
################################################################################
def get_robot_coordinates():
    """
    Get the robot's initial pose data. The returned dictionary places pitch and roll at the beginning,
    in the format:
      {"pitch": ..., "roll": ..., "x": ..., "y": ..., "z": ...}
    """
    try:
        resp = requests.get(f"{SERVER_URL}/get_coordinates")
        if resp.status_code == 200:
            data = resp.json()
            rb = data["robot"]  # {"x":..., "y":..., "z":...}

            # Assume pitch and roll are also in data["robot"]
            pitch = rb.get("pitch", 0.0)
            roll  = rb.get("roll", 0.0)

            # Return with pitch and roll at the beginning
            return {
                "pitch": float(pitch),
                "roll": float(roll),
                "x": float(rb["x"]),
                "y": float(rb["y"]),
                "z": float(rb["z"])
            }
    except Exception as e:
        print(f"[get_robot_coordinates] Exception: {e}")
    return None


def get_error():
    """
    Get the error (this function is currently not used)
    """
    try:
        resp = requests.get(f"{SERVER_URL}/get_coordinates")
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("error", 999.0))
    except Exception as e:
        print(f"[get_error] Exception: {e}")
    return 999.0


def send_target_coordinates(coord):
    """
    Send the target to the backend.
    """
    payload = {
        "update_type": "target",
        "tx": str(coord["x"]),
        "ty": str(coord["y"]),
        "tz": str(coord["z"])
    }
    try:
        resp = requests.post(f"{SERVER_URL}/", data=payload)
        if resp.status_code == 200:
            print(f"Sent target: {coord}")
        else:
            print(f"Failed to send target, code={resp.status_code}")
    except Exception as e:
        print(f"send_target_coordinates exception: {e}")


def wait_for_flags():
    """
    Wait for the robotic arm to be ready.
    """
    print("Waiting for flags == 0...")
    while True:
        try:
            resp = requests.get(f"{SERVER_URL}/get_coordinates")
            if resp.status_code == 200:
                data = resp.json()
                flags = [
                    data["pi1"]["flag"],
                    data["pi2"]["flag"],
                    data["pi3"]["flag"],
                    data["pi4"]["flag"]
                ]
                if all(f == "0" for f in flags):
                    print("All flags are 0 => ready.")
                    break
            time.sleep(1)
        except Exception as e:
            print(f"[wait_for_flags] Exception: {e}")
            time.sleep(1)


def euclidean_distance(posA, posB):
    if not posA or not posB:
        return 999999.0
    return math.sqrt(
        (posA["x"] - posB["x"])**2 +
        (posA["y"] - posB["y"])**2 +
        (posA["z"] - posB["z"])**2
    )


################################################################################
# Main
################################################################################
def main():
    global first_finetune_done, second_finetune_done

    # 1) Load first 50 records => train main model if there are at least 5 records
    load_first_50_records()
    if len(movement_records) >= 5:
        # To view training data, set print_data=True
        train_main_model(print_data=True)

    # 2) Load existing finetune data => train finetune model if there are at least 5 records
    load_finetune_records()
    if len(finetune_records) >= 5:
        train_finetune_model()

    # 3) Initial fine-tuning steps
    if not first_finetune_done:
        print("[main] Starting initial fine-tuning...")
        naive_finetune_initial()
        first_finetune_done = True

    fixed_index = 0

    while True:
        # Get robot_before (including pitch and roll)
        robot_before = get_robot_coordinates()
        if not robot_before:
            print("No robot coords => skip.")
            time.sleep(2)
            continue

        # Pick real_target
        real_target = FIXED_TARGETS[fixed_index]
        print(f"\nUsing real_target [{fixed_index + 1}/{len(FIXED_TARGETS)}]: {real_target}")
        fixed_index = (fixed_index + 1) % len(FIXED_TARGETS)

        if main_model is None:
            print("No main_model => skip movement.")
            time.sleep(2)
            continue

        # Predict (8-dimensional input => 3-dimensional output)
        predicted_target = predict_main_model(robot_before, real_target)
        if not predicted_target:
            print("predict_main_model => None => skip.")
            time.sleep(2)
            continue
        print("Predicted target =>", predicted_target)

        big_entry_movement = {
            "robot_before": robot_before.copy(),
            "target": predicted_target.copy(),
            "attempts": []
        }

        attempt_count = 0
        current_target = predicted_target.copy()

        while True:
            send_target_coordinates(current_target)
            wait_for_flags()

            new_robot = get_robot_coordinates()
            # Calculate error directly based on new_robot and real_target
            err_val = euclidean_distance(new_robot, real_target) if new_robot else 999.0
            print(f"Arrived => {new_robot}, error={err_val}")

            att_obj = {
                "robot_after": new_robot.copy() if new_robot else {},
                "error": err_val
            }

            # Detect green
            dist_info = findGreen.detect_green_center_distance_and_direction()
            if dist_info:
                dist_val, angle_val = dist_info
                att_obj["green_distance"] = dist_val
                att_obj["green_angle"] = angle_val
            big_entry_movement["attempts"].append(att_obj)

            if not new_robot:
                print("No new coords => break attempts.")
                break

            dist = euclidean_distance(new_robot, real_target)
            if dist > MAX_ERROR:
                attempt_count += 1
                if attempt_count < MAX_RETRIES:
                    print(f"Error={dist:.2f} > {MAX_ERROR}, partial train + re-predict.")
                    # Write partial data into movement_records
                    partial_rec = {
                        "robot_before": robot_before.copy(),
                        "robot_after": {
                            "x": new_robot["x"],
                            "y": new_robot["y"],
                            "z": new_robot["z"]
                        },
                        "target": current_target.copy()
                    }
                    movement_records.append(partial_rec)
                    train_main_model()

                    # Re-predict
                    new_pred = predict_main_model(new_robot, real_target)
                    if new_pred:
                        current_target = new_pred
                    time.sleep(1)
                    continue
                else:
                    print("Max retries => break.")
            break

        # Store record to file
        try:
            with open(MOVEMENT_RECORDS_FILE, "a") as f:
                json.dump(big_entry_movement, f)
                f.write("\n")
            print("Movement record saved.")
        except Exception as e:
            print(f"[main] Exception while saving movement record: {e}")

        # Also store the first attempt in memory
        if big_entry_movement["attempts"]:
            first_att = big_entry_movement["attempts"][0]
            if "robot_after" in first_att and first_att["robot_after"]:
                # Only store x, y, z
                ra = first_att["robot_after"]
                movement_records.append({
                    "robot_before": robot_before.copy(),
                    "robot_after": {
                        "x": ra["x"],
                        "y": ra["y"],
                        "z": ra["z"]
                    },
                    "target": predicted_target.copy()
                })

        # Retrain if data is sufficient
        if len(movement_records) >= 5:
            train_main_model()

        # Determine if the target is a shooting point:
        # Calculate the offset of real_target relative to central_coord
        rel_used = {
            'x': real_target['x'] - central_coord[0],
            'y': real_target['y'] - central_coord[1],
            'z': real_target['z'] - central_coord[2]
        }
        # Check if this offset matches any offset in SHOOTING_OFFSETS (allowing a small tolerance)
        is_shooting_point = False
        for offset in SHOOTING_OFFSETS:
            if abs(rel_used['x'] - offset['x']) < 1e-6 and \
               abs(rel_used['y'] - offset['y']) < 1e-6 and \
               abs(rel_used['z'] - offset['z']) < 1e-6:
                is_shooting_point = True
                break
        if is_shooting_point:
            print("\n*** This is a shooting point ***\n")
            print("=> Starting incremental fine-tuning.")
            finetune_loop_incremental(LATER_TIME_STEPS)

        if len(movement_records) >= 1000:
            print("Reached 1000 => stopping.")
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
