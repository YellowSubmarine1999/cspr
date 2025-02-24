import requests
import time
import numpy as np
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import json

# Global server configuration
SERVER_IP = '10.0.0.200'
SERVER_PORT = 5000
SERVER_URL = f'http://{SERVER_IP}:{SERVER_PORT}'

# GPIO Pins setup
GPIO_pins = (14, 15, 18)  # Microstep Resolution MS1-MS3
direction = 20  # Direction
step = 21  # Step

# Create an instance of the A4988Nema class
mymotortest = RpiMotorLib.A4988Nema(direction, step, GPIO_pins, "A4988")

# Coordinates and node info
initial_coordinates = None
reference_coordinates = None
last_coordinates = None
pi_number = 'pi1'  # Change to your node if needed


def set_flag(value):
    """
    Sets the current node's (pi_number) flag to the specified value.
    """
    url = f'{SERVER_URL}/update_coordinates'
    payload = {f'flag_{pi_number}': str(value)}
    try:
        response = requests.post(url, json=payload)
        print(f"Set flag_{pi_number} to {value}. Server response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Failed to set flag_{pi_number} to {value}:", e)


def fetch_data():
    global last_coordinates, initial_coordinates, reference_coordinates
    url = f'{SERVER_URL}/get_coordinates'
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()

                # Set initial robot coordinates if not set
                if (initial_coordinates is None
                        and 'robot' in data
                        and all(data['robot'][k] is not None for k in ['x', 'y', 'z'])):
                    initial_coordinates = {
                        'x': float(data['robot']['x']),
                        'y': float(data['robot']['y']),
                        'z': float(data['robot']['z'])
                    }
                    print("Initial coordinates set to:", initial_coordinates)

                # Set reference coordinates for this node if not set
                if (reference_coordinates is None
                        and pi_number in data
                        and all(data[pi_number][k] is not None for k in ['x', 'y', 'z'])):
                    reference_coordinates = {
                        'x': float(data[pi_number]['x']),
                        'y': float(data[pi_number]['y']),
                        'z': float(data[pi_number]['z'])
                    }
                    print("Reference coordinates set to:", reference_coordinates)

                # Check if 'target' is valid and if current node has flag == '1'
                if ('target' in data
                        and all(data['target'][key] != '0' for key in ['x', 'y', 'z'])
                        and initial_coordinates
                        and reference_coordinates):

                    if pi_number in data and data[pi_number]['flag'] == '1':
                        robot_response = requests.get(url)
                        if robot_response.status_code == 200:
                            robot_data = robot_response.json()
                            if ('robot' in robot_data
                                    and all(robot_data['robot'][k] is not None for k in ['x', 'y', 'z'])):

                                current_robot_coordinates = {
                                    'x': float(robot_data['robot']['x']),
                                    'y': float(robot_data['robot']['y']),
                                    'z': float(robot_data['robot']['z'])
                                }
                                print("Current robot coordinates fetched:", current_robot_coordinates)

                                # 1) Perform motor operations to move to the target
                                perform_motor_operations(data['target'], current_robot_coordinates)

                                # 2) Wait 3s to ensure the platform is stable
                                time.sleep(3)

                                # 3) Set this node's flag to "2"
                                set_flag('2')

                                # 4) Wait for all nodes' flags to be "2"
                                while True:
                                    check_resp = requests.get(url)
                                    if check_resp.status_code == 200:
                                        check_data = check_resp.json()
                                        if all(
                                                check_data[section]['flag'] == '2'
                                                for section in ['pi4', 'pi3', 'pi2', 'pi1']
                                        ):
                                            print(
                                                "All flags are '2'. Proceeding to motor_control_loop (single balance)...")
                                            break
                                    time.sleep(1)

                                # 5) Perform a single balance operation (no loop / threshold check)
                                motor_control_once()

                                time.sleep(1)

                                # 6) After finishing, set this node's flag to "0"
                                set_flag('0')

                                # Break out of fetch_data loop
                                break
                            else:
                                print("Robot coordinates not found or None.")
                        else:
                            print("Failed to fetch current robot coordinates. Status code:", robot_response.status_code)
                else:
                    print("No valid target coordinates received. Waiting for new data.")
                    time.sleep(1)
            else:
                print("Failed to retrieve data. Status code:", response.status_code)
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            print("Error connecting to server:", e)
            time.sleep(1)


def motor_control_once():
    """
    Perform exactly one balance operation for this node.
    No repeated loop, no threshold check.
    """
    print("Performing a single balance operation for", pi_number)
    # Fetch adjustment data from the server for this node
    url = f'{SERVER_URL}/get_coordinates'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if pi_number in data and 'adjustment' in data[pi_number]:
                real_adjustment = float(data[pi_number]['adjustment'])
                print(f"Retrieved adjustment value: {real_adjustment}")
                control_adjustment_motor(real_adjustment)
                # Optionally notify the server we are done
                send_adjustment_done(f'{SERVER_URL}/adjustment_done')
            else:
                print(f"{pi_number} adjustment data not found.")
        else:
            print(f"Failed to retrieve data, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")


def fetch_adjustment():
    """
    (Optional) If you still need a separate function to fetch adjustment.
    """
    url = f'{SERVER_URL}/get_coordinates'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if pi_number in data and 'adjustment' in data[pi_number]:
                return float(data[pi_number]['adjustment'])
            else:
                print("Adjustment data not found.")
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
    return None


def control_adjustment_motor(adjustment):
    """
    Convert the adjustment value into motor steps and move the motor.
    """
    if adjustment is None:
        print("No adjustment to apply.")
        return
    steps = adjustment * 15  # Example multiplier
    clockwise = steps > 0
    steps = abs(steps)
    if steps < 1 and steps != 0:
        steps = 1
    mymotortest.motor_go(clockwise, "Full", int(steps), 0.005, False, 0.05)
    direction = "clockwise" if clockwise else "counterclockwise"
    print(f"Adjusted motor {direction} for {int(steps)} steps.")


def send_adjustment_done(url):
    """
    Notify the server that the node has finished an adjustment.
    """
    headers = {'Content-Type': 'application/json'}
    data = {'status': 'adjustment_done'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=2)
        if response.status_code == 200:
            print("Successfully sent adjustment done signal to the server")
        else:
            print(f"Failed to send adjustment done signal, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending adjustment done signal: {e}")


def perform_motor_operations(target_coordinates, robot_coordinates):
    """
    Execute the primary motor movement to approach the target from the current robot coordinates.
    """
    x = float(target_coordinates['x']) - float(reference_coordinates['x'])
    y = float(target_coordinates['y']) - float(reference_coordinates['y'])
    z = float(target_coordinates['z']) - float(reference_coordinates['z'])
    current_distance = np.linalg.norm(np.array([x, y, z]))

    last_distance = np.linalg.norm(np.array([
        robot_coordinates['x'] - reference_coordinates['x'],
        robot_coordinates['y'] - reference_coordinates['y'],
        robot_coordinates['z'] - reference_coordinates['z']
    ]))

    distance = abs(current_distance - last_distance)
    steps = int(distance * 43)
    clockwise = current_distance > last_distance
    rotate_motor(steps, clockwise)


def rotate_motor(steps, clockwise):
    """
    Rotate the stepper motor for the specified number of steps in the specified direction.
    """
    if steps > 0:
        mymotortest.motor_go(clockwise, "Full", steps, 0.004, False, 0.05)
        direction_str = "Clockwise" if clockwise else "Counter-Clockwise"
        print(f"Rotating {direction_str} {steps} steps")


# Main loop
while True:
    fetch_data()
    time.sleep(3)
