import time
import logging
from picamera2 import Picamera2
import requests
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_URL = "http://10.0.0.200:5000/upload"

def capture_and_upload():
    picam2 = Picamera2()
    picam2.start_preview()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    url = 'http://10.0.0.200:5000/get_coordinates'

    def format_coordinate(value):
        try:
            value_float = float(value)
        except ValueError:
            value_float = 0.0
        value_str = "{:.2f}".format(value_float)
        value_str = value_str.replace('.', '_')
        return value_str

    try:
        while True:
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to get data from server: {e}")
                time.sleep(1)
                continue

            # Retrieve target coordinates
            target_x = data.get('target', {}).get('x', 0.0)
            target_y = data.get('target', {}).get('y', 0.0)
            target_z = data.get('target', {}).get('z', 0.0)

            # Retrieve robot coordinates
            robot_x = data.get('robot', {}).get('x', 0.0)
            robot_y = data.get('robot', {}).get('y', 0.0)
            robot_z = data.get('robot', {}).get('z', 0.0)

            target_x_str = format_coordinate(target_x)
            target_y_str = format_coordinate(target_y)
            target_z_str = format_coordinate(target_z)

            robot_x_str = format_coordinate(robot_x)
            robot_y_str = format_coordinate(robot_y)
            robot_z_str = format_coordinate(robot_z)

            flags = [data.get(key, {}).get('flag', '1') for key in ['pi4', 'pi3', 'pi2', 'pi1']]

            # Proceed only if all flags are '0'
            if all(flag == '0' for flag in flags):
                filename = (
                    f"target_x{target_x_str}_y{target_y_str}_z{target_z_str}_"
                    f"robot_x{robot_x_str}_y{robot_y_str}_z{robot_z_str}.jpg"
                )

                try:
                    picam2.capture_file(filename)
                    logging.info(f"Captured image: {filename}")
                except Exception as e:
                    logging.error(f"Failed to capture image: {e}")
                    time.sleep(1)
                    continue

                try:
                    with open(filename, 'rb') as f:
                        files = {'file': f}
                        upload_response = requests.post(UPLOAD_URL, files=files)
                        upload_response.raise_for_status()
                        logging.info(f"Upload status: {upload_response.status_code} - {upload_response.reason}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Failed to upload image: {e}")
                except Exception as e:
                    logging.error(f"Error during file upload: {e}")
                finally:
                    try:
                        os.remove(filename)
                        logging.info(f"Removed file: {filename}")
                    except OSError as e:
                        logging.error(f"Failed to remove file {filename}: {e}")

                # Wait until at least one flag becomes '1'
                logging.info("Entering waiting state, waiting for at least one flag to be '1'")
                while True:
                    try:
                        flag_response = requests.get(url)
                        flag_response.raise_for_status()
                        flag_data = flag_response.json()

                        current_flags = [
                            flag_data.get(key, {}).get('flag', '1')
                            for key in ['pi4', 'pi3', 'pi2', 'pi1']
                        ]

                        if any(flag == '1' for flag in current_flags):
                            logging.info("Flag change detected, preparing for next capture")
                            break
                        else:
                            time.sleep(0.5)
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Failed to get data from server during waiting: {e}")
                        time.sleep(1)
            else:
                time.sleep(0.5)
    finally:
        picam2.stop_preview()
        logging.info("Stopped camera preview")

if __name__ == '__main__':
    capture_and_upload()
