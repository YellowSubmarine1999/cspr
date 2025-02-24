import smbus
import time
import requests
import json
import math

# MPU6050 constants
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47
SCALE_MODIFIER_ACCEL = 16384.0
SCALE_MODIFIER_GYRO = 131.0

# Complementary filter parameters
ALPHA = 0.98
DT = 0.05  # Time interval in seconds

# Initialize I2C bus and wake up MPU6050
bus = smbus.SMBus(1)
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# Server URL
server_url = "http://10.0.0.200:5000"


def fetch_server_coordinates(server_url):
    """
    Fetch JSON data from the server's /get_coordinates endpoint.
    """
    try:
        response = requests.get(server_url + "/get_coordinates")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Error fetching coordinates from server:", e)
        return None


def send_adjustments(adjustments):
    """
    Send the calculated adjustments to the server's /update_coordinates endpoint.
    (The adjustments for each corner are negated.)
    Each adjustment is capped so that its absolute value does not exceed 20.
    """
    # Cap each adjustment to the range [-20, 20]
    capped_adjustments = [max(min(adj, 20), -20) for adj in adjustments]

    url = server_url + "/update_coordinates"
    headers = {'Content-Type': 'application/json'}
    data = {
        "pi4": {"adjustment": str(-capped_adjustments[0])},
        "pi3": {"adjustment": str(-capped_adjustments[1])},
        "pi2": {"adjustment": str(-capped_adjustments[2])},
        "pi1": {"adjustment": str(-capped_adjustments[3])}
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print("Successfully sent adjustments to the server")
        else:
            print("Failed to send adjustments, status code:", response.status_code)
    except Exception as e:
        print("Error sending adjustments:", e)


def send_sensor_angles(server_url, sensor_roll, sensor_pitch):
    """
    Upload the sensor angles to the server.
    The sensor angles are swapped: the sensor roll is uploaded as predicted_pitch
    and the sensor pitch as predicted_roll.
    """
    payload = {

        "pitch": sensor_roll,  # swapped: sensor roll becomes predicted_pitch
        "roll": sensor_pitch  # swapped: sensor pitch becomes predicted_roll

    }
    try:
        response = requests.post(server_url + "/update_coordinates", json=payload)
        print("Uploaded sensor angles, response:", response.json())
    except Exception as e:
        print("Error sending sensor angles:", e)


def compute_angles(accel_data, gyro_data, roll, pitch):
    """
    Compute the roll and pitch angles using a complementary filter.
    """
    roll = math.atan2(accel_data['y'], accel_data['z']) * (180 / math.pi)
    pitch = math.atan2(-accel_data['x'], math.sqrt(accel_data['y'] ** 2 + accel_data['z'] ** 2)) * (180 / math.pi)

    return roll, pitch


def calculate_adjustments(sensor_roll, sensor_pitch, length, width, predicted_pitch, predicted_roll):
    """
    Calculate adjustments based on the differences between the sensor angles and predicted angles.
    Note: The sensor angles passed to this function are swapped relative to our expected orientation.
    """
    # Compute the differences (in degrees) and convert to radians
    # sensor_roll here actually corresponds to the model's pitch (and vice versa)
    roll_rad = math.radians(sensor_roll - predicted_pitch)
    pitch_rad = math.radians(sensor_pitch - predicted_roll)

    delta_x = (length / 2) * math.tan(pitch_rad)
    delta_y = (width / 2) * math.tan(roll_rad)

    # Combine the effects for each corner (the order may be adjusted as needed)
    return [-delta_x + delta_y, -delta_x - delta_y, delta_x - delta_y, delta_x + delta_y]


def read_i2c_word(reg):
    """
    Read two bytes from the I2C bus and combine them into one value.
    """
    high = bus.read_byte_data(MPU6050_ADDR, reg)
    low = bus.read_byte_data(MPU6050_ADDR, reg + 1)
    value = (high << 8) + low
    if value >= 0x8000:
        return -((65535 - value) + 1)
    else:
        return value


def main():
    roll = 0
    pitch = 0
    # Platform dimensions in centimeters (example: length = 20cm, width = 14cm)
    length = 20
    width = 14
    while True:
        # Read accelerometer data (in g's)
        accel_data = {
            'x': read_i2c_word(ACCEL_XOUT_H) / SCALE_MODIFIER_ACCEL,
            'y': read_i2c_word(ACCEL_YOUT_H) / SCALE_MODIFIER_ACCEL,
            'z': read_i2c_word(ACCEL_ZOUT_H) / SCALE_MODIFIER_ACCEL
        }
        # Read gyroscope data
        gyro_data = {
            'x': read_i2c_word(GYRO_XOUT_H),
            'y': read_i2c_word(GYRO_YOUT_H),
            'z': read_i2c_word(GYRO_ZOUT_H)
        }
        # Compute sensor angles (roll and pitch)
        roll, pitch = compute_angles(accel_data, gyro_data, roll, pitch)
        print(f"Sensor Roll: {roll:.2f}, Sensor Pitch: {pitch:.2f}")

        # Fetch predicted angles from the server
        data_server = fetch_server_coordinates(server_url)
        if data_server is not None:
            try:
                predicted_pitch = float(data_server.get('robot', {}).get('predicted_pitch', 0))
                predicted_roll = float(data_server.get('robot', {}).get('predicted_roll', 0))
            except Exception as e:
                print("Error parsing predicted angles from server:", e)
                predicted_pitch = 0
                predicted_roll = 0
        else:
            predicted_pitch = 0
            predicted_roll = 0

        print(
            f"Predicted Pitch (from server): {predicted_pitch:.2f}, Predicted Roll (from server): {predicted_roll:.2f}")

        # Because the sensor's roll and pitch are swapped relative to our expected model,
        # swap them when calculating adjustments.
        # That is, pass sensor_pitch as the "roll" value and sensor_roll as the "pitch" value.
        adjustments = calculate_adjustments(pitch, roll, length, width, predicted_pitch, predicted_roll)
        print("Calculated adjustments:", adjustments)

        # Send the adjustments to the server
        # send_adjustments(adjustments)
        # Upload the sensor angles, but swap them so that sensor roll is uploaded as predicted_pitch
        # and sensor pitch as predicted_roll.
        send_sensor_angles(server_url, roll, pitch)

        time.sleep(DT)


if __name__ == "__main__":
    main()
