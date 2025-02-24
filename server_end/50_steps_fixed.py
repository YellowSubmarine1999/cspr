import requests
import json
import numpy as np
import time
import os
from shapely.geometry import Point, Polygon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# >>> NEW IMPORT <<<
import findGreen

SERVER_URL = 'http://10.0.0.200:5000'
MOVEMENT_RECORDS_FILE = 'movement_records.json'
PREDICTION_RECORDS_FILE = 'prediction_records.json'
MODEL_FILE = 'NN_model.h5'

movement_records = []

# Load existing movement_records
if os.path.exists(MOVEMENT_RECORDS_FILE):
    with open(MOVEMENT_RECORDS_FILE, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                movement_records.append(record)
            except json.JSONDecodeError:
                print("Error parsing movement_records.json, skipping line.")

# Remove prediction flag and predicted_target_coords since we now finish after 50 records
target_index = 0

def get_robot_coordinates():
    """
    Modified get_robot_coordinates() returns pitch and roll as well.
    """
    try:
        resp = requests.get(f'{SERVER_URL}/get_coordinates')
        if resp.status_code == 200:
            data = resp.json()
            rb = data['robot']
            return {
                'x': float(rb['x']),
                'y': float(rb['y']),
                'z': float(rb['z']),
                'pitch': float(rb.get('pitch', 0.0)),
                'roll': float(rb.get('roll', 0.0))
            }
    except Exception as e:
        print(f"get_robot_coordinates exception: {e}")
    return None

def get_motor_coordinates():
    try:
        resp = requests.get(f'{SERVER_URL}/get_coordinates')
        if resp.status_code == 200:
            data = resp.json()
            mc = {}
            for pi in ['pi1', 'pi2', 'pi3', 'pi4']:
                mc[pi] = {
                    'x': float(data[pi]['x']),
                    'y': float(data[pi]['y']),
                    'z': float(data[pi]['z'])
                }
            return mc
    except:
        pass
    return None

def get_error():
    try:
        resp = requests.get(f'{SERVER_URL}/get_coordinates')
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get('error', '0'))
    except:
        pass
    return None

def get_flags():
    try:
        resp = requests.get(f'{SERVER_URL}/get_coordinates')
        if resp.status_code == 200:
            data = resp.json()
            return {
                'pi1': data['pi1']['flag'],
                'pi2': data['pi2']['flag'],
                'pi3': data['pi3']['flag'],
                'pi4': data['pi4']['flag']
            }
    except:
        pass
    return None

def send_target_coordinates(tg):
    payload = {
        'update_type': 'target',
        'tx': str(tg['x']),
        'ty': str(tg['y']),
        'tz': str(tg['z'])
    }
    try:
        resp = requests.post(f'{SERVER_URL}/', data=payload)
        if resp.status_code == 200:
            print(f'Sent target: {tg}')
        else:
            print(f'Failed to send target, code={resp.status_code}')
    except Exception as e:
        print(f"send_target_coordinates exception: {e}")

def wait_for_flags():
    print('Waiting for flags == 0...')
    while True:
        fl = get_flags()
        if fl is None:
            time.sleep(1)
            continue
        if all(v == '0' for v in fl.values()):
            print('All flags are 0.')
            break
        time.sleep(1)

def log_movement_record(big_entry):
    with open(MOVEMENT_RECORDS_FILE, 'a') as fout:
        json.dump(big_entry, fout)
        fout.write('\n')

def log_prediction_record(entry):
    with open(PREDICTION_RECORDS_FILE, 'a') as fout:
        json.dump(entry, fout)
        fout.write('\n')

def train_neural_network_model():
    global model, scaler_X, scaler_y
    if not movement_records:
        print("No data to train on.")
        return

    X_list = []
    y_list = []
    for record in movement_records:
        if 'attempts' not in record:
            continue
        if not record['attempts']:
            continue

        # Use robot_before (which includes pitch and roll, if available)
        robot_before = record['robot_before']
        target = record['target']
        first_attempt = record['attempts'][0]
        robot_after = first_attempt['robot_after']

        feat = []
        feat.extend([robot_before['x'], robot_before['y'], robot_before['z']])
        # Optionally include pitch/roll as extra features if needed
        feat.extend([target['x'], target['y'], target['z']])
        X_list.append(feat)

        lbl = [robot_after['x'], robot_after['y'], robot_after['z']]
        y_list.append(lbl)

    if not X_list:
        print("No valid attempts found in movement records for training.")
        return

    X = np.array(X_list)
    y = np.array(y_list)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='linear'))
    model.compile(optimizer=Adam(0.001), loss='mse')

    print("Training NN model...")
    model.fit(X_scaled, y_scaled, epochs=50, batch_size=16, verbose=0)
    print("Training done.")

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    r2  = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y, axis=0)) ** 2))
    print(f"Train metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    model.save(MODEL_FILE)
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, 'scalers.pkl')
    print("Model and scalers saved.")

def load_trained_model():
    global model, scaler_X, scaler_y
    if os.path.exists(MODEL_FILE) and os.path.exists('scalers.pkl'):
        model = load_model(MODEL_FILE)
        sc = joblib.load('scalers.pkl')
        scaler_X = sc['scaler_X']
        scaler_y = sc['scaler_y']
        print("Loaded trained model and scalers.")
    else:
        print("No trained model found.")
        model = None

def predict_target_coordinates(current_robot_coords, target_coords):
    if model is None:
        print("No model loaded, can't predict.")
        return None
    if not movement_records:
        print("No historical data to use.")
        return None

    feat = []
    feat.extend([current_robot_coords['x'], current_robot_coords['y'], current_robot_coords['z']])
    feat.extend([target_coords['x'], target_coords['y'], target_coords['z']])

    X_input = np.array([feat])
    X_input_scaled = scaler_X.transform(X_input)
    y_pred_scaled = model.predict(X_input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return {
        'x': float(y_pred[0][0]),
        'y': float(y_pred[0][1]),
        'z': float(y_pred[0][2])
    }

# >>> NEW: Helper function to fetch pitch/roll from server <<<
def get_pitch_roll():
    try:
        resp = requests.get(f'{SERVER_URL}/get_coordinates')
        if resp.status_code == 200:
            data = resp.json()
            robot_data = data.get('robot', {})
            pitch = float(robot_data.get('pitch', 0.0))
            roll  = float(robot_data.get('roll', 0.0))
            return pitch, roll
    except:
        pass
    return None, None

def main():
    load_trained_model()

    # Get initial robot coordinates (including pitch and roll)
    curr_robot = get_robot_coordinates()
    if not curr_robot:
        print("Failed to get initial robot coords.")
        return

    print(f"Initial robot coordinates: {curr_robot}")

    previous_robot = curr_robot.copy()
    same_count = 0
    global target_index

    while True:
        wait_for_flags()

        # Use the next fixed target from FIXED_TARGETS
        target_coords = FIXED_TARGETS[target_index]
        print(f"Using fixed target {target_index+1}/{len(FIXED_TARGETS)}: {target_coords}")
        target_index += 1
        if target_index >= len(FIXED_TARGETS):
            target_index = 0

        # Prepare the record entry (robot_before includes x, y, z, pitch, roll)
        big_entry = {
            "robot_before":  curr_robot,
            "target":        target_coords,
            "attempts":      []
        }

        # Send target
        send_target_coordinates(target_coords)
        wait_for_flags()

        new_robot = get_robot_coordinates()
        err = get_error()
        if not new_robot or err is None:
            print("Failed to get new_robot or error.")
            continue

        # Retry logic: if error > 5.0, retry up to 3 times
        retry_limit = 3
        attempt_count = 0
        while err > 5.0 and attempt_count < retry_limit:
            print(f"High error={err:.2f}, retry {attempt_count+1}/{retry_limit}")

            big_entry["attempts"].append({
                "robot_after": new_robot,
                "error":       err
            })

            wait_for_flags()
            send_target_coordinates(target_coords)
            wait_for_flags()

            new_robot = get_robot_coordinates()
            err = get_error()
            attempt_count += 1
            if not new_robot or err is None:
                break

        final_attempt = {
            "robot_after": new_robot,
            "error":       err
        }

        # Call findGreen to detect green target center (if available)
        distance_direction = findGreen.detect_green_center_distance_and_direction()
        if distance_direction is not None:
            dist, angle = distance_direction
            final_attempt["green_distance"] = dist
            final_attempt["green_angle"]    = angle

        big_entry["attempts"].append(final_attempt)
        movement_records.append(big_entry)
        log_movement_record(big_entry)

        # When 50 or more movement records are collected, immediately train the model and end training.
        if len(movement_records) >= 50:
            print("Reached 50 records, training model and ending training.")
            train_neural_network_model()
            break

        # Check if robot did not move for two iterations, then stop.
        if (new_robot['x'] == previous_robot['x'] and
            new_robot['y'] == previous_robot['y'] and
            new_robot['z'] == previous_robot['z']):
            same_count += 1
            if same_count >= 2:
                print("Robot did not move for two iterations, stopping.")
                break
        else:
            same_count = 0

        previous_robot = new_robot.copy()
        curr_robot = new_robot

        time.sleep(1)

    print("Done collecting data.")

if __name__ == "__main__":
    main()
