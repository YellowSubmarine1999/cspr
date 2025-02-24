import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for
import os
import numpy as np
from werkzeug.utils import secure_filename
import json  # Import json for logging

app = Flask(__name__)

# Initialize coordinate storage and flags for each motor
coordinates = {
    'pi4': {'x': '1', 'y': '0', 'z': '0', 'flag': '0', 'adjustment': '0'},
    'pi3': {'x': '1', 'y': '0', 'z': '0', 'flag': '0', 'adjustment': '0'},
    'pi2': {'x': '1', 'y': '0', 'z': '0', 'flag': '0', 'adjustment': '0'},
    'pi1': {'x': '1', 'y': '0', 'z': '0', 'flag': '0', 'adjustment': '0'},
    'robot': {
        'x': '0',
        'y': '0',
        'z': '0',
        'pitch': '0',  # Original pitch
        'roll': '0',  # Original roll
        'predicted_pitch': '0',  # New: predicted pitch
        'predicted_roll': '0'  # New: predicted roll
    },
    'target': {'x': '0', 'y': '0', 'z': '0'},
    'model_input_target': {'x': '0', 'y': '0', 'z': '0'},  # New variable
    'position': {'x': '0', 'y': '0'},
    # NEW GROUND NORMAL: Add estimated ground normal entry
    'estimated_ground_normal': {'x': '0', 'y': '0', 'z': '0'},
    'motor1': '0',
    'motor2': '0',
    'motor3': '0',
    'motor4': '0',
    'transformation_matrix': [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ],
    'relative_error': '0'
}

# For logging
log_entries = []
previous_flags = {'pi1': '1', 'pi2': '1', 'pi3': '1', 'pi4': '1'}

# Movement log to store starting and ending points
movement_log = {}


def log_entry(entry_type, data):
    """
    Log an entry with a specific type and data.
    """
    entry = {'type': entry_type, 'data': data}
    log_entries.append(entry)
    # Write the entry to the file
    with open('log_file.json', 'a') as f:
        json.dump(entry, f)
        f.write('\n')


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# HTML template for displaying real-time coordinates and flags
TPL = '''
<!doctype html>
<html>
<head>
    <title>Real-time Coordinate Display</title>
    <script>

        function updateCoordinates() {
            fetch('/get_coordinates')
            .then(response => response.json())
            .then(data => {
                document.getElementById('pi4').textContent = `Pi4: X: ${data.pi4.x}, Y: ${data.pi4.y}, Z: ${data.pi4.z}, Flag: ${data.pi4.flag}, Adjustment: ${data.pi4.adjustment}`;
                document.getElementById('pi3').textContent = `Pi3: X: ${data.pi3.x}, Y: ${data.pi3.y}, Z: ${data.pi3.z}, Flag: ${data.pi3.flag}, Adjustment: ${data.pi3.adjustment}`;
                document.getElementById('pi2').textContent = `Pi2: X: ${data.pi2.x}, Y: ${data.pi2.y}, Z: ${data.pi2.z}, Flag: ${data.pi2.flag}, Adjustment: ${data.pi2.adjustment}`;
                document.getElementById('pi1').textContent = `Pi1: X: ${data.pi1.x}, Y: ${data.pi1.y}, Z: ${data.pi1.z}, Flag: ${data.pi1.flag}, Adjustment: ${data.pi1.adjustment}`;
                document.getElementById('robot').textContent = `Robot: X: ${data.robot.x}, Y: ${data.robot.y}, Z: ${data.robot.z}`;
                document.getElementById('target').textContent = `Target: X: ${data.target.x}, Y: ${data.target.y}, Z: ${data.target.z}`;
                document.getElementById('model_input_target').textContent = `Model Input Target: X: ${data.model_input_target.x}, Y: ${data.model_input_target.y}, Z: ${data.model_input_target.z}`; // New display
                document.getElementById('matrix').textContent = `Transformation Matrix: ${data.transformation_matrix.join(", ")}`;
                document.getElementById('error').textContent = `Current Error: ${data.error}`;
                document.getElementById('relative_error').textContent = `Relative Error: ${data.relative_error}%`;
                document.getElementById('motor1').textContent = `Motor1: ${data.motor1}`;
                document.getElementById('motor2').textContent = `Motor2: ${data.motor2}`;
                document.getElementById('motor3').textContent = `Motor3: ${data.motor3}`;
                document.getElementById('motor4').textContent = `Motor4: ${data.motor4}`;
                document.getElementById('position').textContent = `Target-Robot Position: X: ${data.position.x}, Y: ${data.position.y}`;

                // NEW GROUND NORMAL: update estimated ground normal display
                document.getElementById('ground_normal').textContent = `Estimated Ground Normal: X: ${data.estimated_ground_normal.x}, Y: ${data.estimated_ground_normal.y}, Z: ${data.estimated_ground_normal.z}`;

                // Show robot's angle information: original pitch/roll and predicted_pitch/predicted_roll
                document.getElementById('robot_angles').textContent = `Pitch: ${data.robot.pitch}, Roll: ${data.robot.roll}`;
                document.getElementById('predicted_robot_angles').textContent = `Predicted Pitch: ${data.robot.predicted_pitch}, Predicted Roll: ${data.robot.predicted_roll}`;
            })
            .catch(error => console.error('Error:', error));
        }
        setInterval(updateCoordinates, 1000); // Update frequency is every second

    </script>
</head>
<body>
    <h1>Real-time Coordinates</h1>
    <div id="pi4"><strong>Pi4:</strong></div>
    <div id="pi3"><strong>Pi3:</strong></div>
    <div id="pi2"><strong>Pi2:</strong></div>
    <div id="pi1"><strong>Pi1:</strong></div>
    <div id="robot"><strong>Robot:</strong></div>
    <div id="target"><strong>Target:</strong></div>
    <div id="model_input_target"><strong>Model Input Target:</strong></div> <!-- New display -->
    <div id="matrix"><strong>Transformation Matrix:</strong></div>
    <div id="error"><strong>Current Error:</strong></div>
    <div id="relative_error"><strong>Relative Error:</strong></div>
    <div id="motor1"><strong>Motor1:</strong></div>
    <div id="motor2"><strong>Motor2:</strong></div>
    <div id="motor3"><strong>Motor3:</strong></div>
    <div id="motor4"><strong>Motor4:</strong></div>

    <!-- NEW GROUND NORMAL: Add a line for Estimated Ground Normal -->
    <div id="ground_normal"><strong>Estimated Ground Normal:</strong></div>

    <div id="position"><strong>Target-Robot Position:</strong></div>

    <!-- Show robot angle information -->
    <div id="robot_angles"><strong>Robot Angles (pitch/roll):</strong></div>
    <div id="predicted_robot_angles"><strong>Predicted Robot Angles (pitch/roll):</strong></div>

    <h2>Update Target Coordinates</h2>
    <form action="/" method="post">
        <input type="hidden" name="update_type" value="target">
        <h3>Update Target Coordinates</h3>
        <input type="text" name="tx" placeholder="Enter target X" required>
        <input type="text" name="ty" placeholder="Enter target Y" required>
        <input type="text" name="tz" placeholder="Enter target Z" required>
        <button type="submit">Update Target Coordinates</button>
    </form>

    <h2>Update Model Input Target Coordinates</h2>
    <form action="/" method="post">
        <input type="hidden" name="update_type" value="model_input_target">
        <h3>Update Model Input Target Coordinates</h3>
        <input type="text" name="mtx" placeholder="Enter model target X" required>
        <input type="text" name="mty" placeholder="Enter model target Y" required>
        <input type="text" name="mtz" placeholder="Enter model target Z" required>
        <button type="submit">Update Model Input Target Coordinates</button>
    </form>

    <h2>Upload New Image</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" />
        <input type="submit" value="Upload" />
    </form>
    {% if image %}
        <h2>Most Recent Image:</h2>
        <img src="{{ url_for('uploaded_file', filename=image) }}" style="max-width: 600px;">
    {% else %}
        <p>No image uploaded yet.</p>
    {% endif %}

</body>
</html>
'''


@app.route("/adjustment_done", methods=["POST"])
def adjustment_done():
    """
    Handle adjustment done signal from motors.
    """
    print("Received adjustment done signal from motor.")
    return jsonify({"status": "success"}), 200


def get_latest_image():
    """
    Retrieve the latest uploaded image.
    """
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        if not files:
            return None
        # Get the most recent file by modification time
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)))
        return latest_file
    except Exception as e:
        print(f"Error retrieving latest image: {e}")
        return None


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads.
    """
    print("Received a request to upload a file.")
    file = request.files['file']
    if file and file.filename:  # Check if the file exists and has a filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved at {file_path}, now redirecting...")
        return render_template_string(TPL, image=filename)  # Show the newly uploaded image
    else:
        print("No file provided or file without a filename.")
    return 'No file provided', 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/", methods=["GET", "POST"])
def home():
    """
    Home route to display coordinates and handle updates.
    """
    image_filename = None
    if request.method == 'POST':
        update_type = request.form.get('update_type')
        if update_type == 'target':
            # Update target coordinates
            coordinates['target'] = {
                'x': request.form.get('tx', '0'),
                'y': request.form.get('ty', '0'),
                'z': request.form.get('tz', '0')
            }

            # Log the target update
            log_entry('target_update', coordinates['target'])

            # Store the starting point for relative error calculation
            movement_log['starting_point'] = coordinates['robot'].copy()

            print("Updated Target Coordinates:", coordinates['target'])

        elif update_type == 'model_input_target':
            # Update model_input_target coordinates
            coordinates['model_input_target'] = {
                'x': request.form.get('mtx', '0'),
                'y': request.form.get('mty', '0'),
                'z': request.form.get('mtz', '0')
            }

            # Log the model_input_target update
            log_entry('model_input_target_update', coordinates['model_input_target'])

            print("Updated Model Input Target Coordinates:", coordinates['model_input_target'])

        # Update flags and adjustments regardless of which form was submitted
        coordinates['pi4']['flag'] = '1'
        coordinates['pi3']['flag'] = '1'
        coordinates['pi2']['flag'] = '1'
        coordinates['pi1']['flag'] = '1'
        coordinates['pi4']['adjustment'] = '0'
        coordinates['pi3']['adjustment'] = '0'
        coordinates['pi2']['adjustment'] = '0'
        coordinates['pi1']['adjustment'] = '0'

        # Log robot coordinates when new target coordinates are obtained
        if update_type == 'target' or update_type == 'model_input_target':
            movement_log['starting_point'] = coordinates['robot'].copy()

    if not image_filename:
        image_filename = get_latest_image()
    return render_template_string(TPL, image=image_filename)


def send_adjustments(adjustments):
    """
    Send adjustments to another server.
    """
    url = 'http://10.0.0.200:5000/update_coordinates'
    payload = {
        'pi4': {'adjustment': str(adjustments[0])},
        'pi3': {'adjustment': str(adjustments[1])},
        'pi2': {'adjustment': str(adjustments[2])},
        'pi1': {'adjustment': str(adjustments[3])}
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Adjustments sent successfully, response:", response.json())
        else:
            print("Failed to send adjustments, status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Failed to send adjustments:", e)


@app.route("/calculate_error", methods=["POST"])
def calculate_error():
    """
    Calculate the error between robot and target coordinates.
    """
    try:
        # Convert coordinates to float and compute the Euclidean distance
        current = np.array(
            [float(coordinates['robot']['x']), float(coordinates['robot']['y']), float(coordinates['robot']['z'])])
        target = np.array(
            [float(coordinates['target']['x']), float(coordinates['target']['y']), float(coordinates['target']['z'])])
        error = np.linalg.norm(current - target)
        coordinates['error'] = str(error)
        return jsonify({'status': 'success', 'error': error})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route("/update_coordinates", methods=["POST"])
def update_coordinates_route():
    """
    Update coordinates based on incoming JSON data.
    """
    global previous_flags  # To modify the global variable
    data = request.get_json()
    if data:
        # Handle coordinate updates
        if 'x1' in data:
            coordinates['pi4']['x'] = data['x1']
            coordinates['pi4']['y'] = data['y1']
            coordinates['pi4']['z'] = data['z1']
            # Log the pi4 input
            log_entry('pi_input', {'pi': 'pi4', 'coordinates': coordinates['pi4']})
        if 'x2' in data:
            coordinates['pi3']['x'] = data['x2']
            coordinates['pi3']['y'] = data['y2']
            coordinates['pi3']['z'] = data['z2']
            # Log the pi3 input
            log_entry('pi_input', {'pi': 'pi3', 'coordinates': coordinates['pi3']})
        if 'x3' in data:
            coordinates['pi2']['x'] = data['x3']
            coordinates['pi2']['y'] = data['y3']
            coordinates['pi2']['z'] = data['z3']
            # Log the pi2 input
            log_entry('pi_input', {'pi': 'pi2', 'coordinates': coordinates['pi2']})
        if 'x4' in data:
            coordinates['pi1']['x'] = data['x4']
            coordinates['pi1']['y'] = data['y4']
            coordinates['pi1']['z'] = data['z4']
            # Log the pi1 input
            log_entry('pi_input', {'pi': 'pi1', 'coordinates': coordinates['pi1']})
        if 'dcx' in data:
            coordinates['robot']['x'] = data['dcx']
            coordinates['robot']['y'] = data['dcy']
            coordinates['robot']['z'] = data['dcz']
        if 'px' in data:
            coordinates['position']['x'] = data['px']
            coordinates['position']['y'] = data['py']
        if 'm1' in data:
            coordinates['motor1'] = data['m1']
        if 'm2' in data:
            coordinates['motor2'] = data['m2']
        if 'm3' in data:
            coordinates['motor3'] = data['m3']
        if 'm4' in data:
            coordinates['motor4'] = data['m4']
        if 'transformation_matrix' in data:
            coordinates['transformation_matrix'] = data['transformation_matrix']
        if 'flag_pi4' in data:
            coordinates['pi4']['flag'] = data['flag_pi4']
        if 'flag_pi3' in data:
            coordinates['pi3']['flag'] = data['flag_pi3']
        if 'flag_pi2' in data:
            coordinates['pi2']['flag'] = data['flag_pi2']
        if 'flag_pi1' in data:
            coordinates['pi1']['flag'] = data['flag_pi1']
        if 'pi4' in data and 'adjustment' in data['pi4']:
            coordinates['pi4']['adjustment'] = data['pi4']['adjustment']
        if 'pi3' in data and 'adjustment' in data['pi3']:
            coordinates['pi3']['adjustment'] = data['pi3']['adjustment']
        if 'pi2' in data and 'adjustment' in data['pi2']:
            coordinates['pi2']['adjustment'] = data['pi2']['adjustment']
        if 'pi1' in data and 'adjustment' in data['pi1']:
            coordinates['pi1']['adjustment'] = data['pi1']['adjustment']

        if 'target' in data:
            coordinates['target']['x'] = data['target'].get('x', '0')
            coordinates['target']['y'] = data['target'].get('y', '0')
            coordinates['target']['z'] = data['target'].get('z', '0')
            # Log the target update
            log_entry('target_update', coordinates['target'])
            # Store the starting point for relative error calculation
            movement_log['starting_point'] = coordinates['robot'].copy()

        if 'model_input_target' in data:
            coordinates['model_input_target']['x'] = data['model_input_target'].get('x', '0')
            coordinates['model_input_target']['y'] = data['model_input_target'].get('y', '0')
            coordinates['model_input_target']['z'] = data['model_input_target'].get('z', '0')
            # Log the model_input_target update
            log_entry('model_input_target_update', coordinates['model_input_target'])

        # NEW GROUND NORMAL: If 'estimated_ground_normal' in data, update it
        if 'estimated_ground_normal' in data:
            coordinates['estimated_ground_normal']['x'] = data['estimated_ground_normal'].get('x', '0')
            coordinates['estimated_ground_normal']['y'] = data['estimated_ground_normal'].get('y', '0')
            coordinates['estimated_ground_normal']['z'] = data['estimated_ground_normal'].get('z', '0')

        if 'pitch' in data:
            coordinates['robot']['pitch'] = data['pitch']
        else:
            # If the server does not have this key yet, initialize it
            if 'pitch' not in coordinates['robot']:
                coordinates['robot']['pitch'] = '0'

        if 'roll' in data:
            coordinates['robot']['roll'] = data['roll']
        else:
            if 'roll' not in coordinates['robot']:
                coordinates['robot']['roll'] = '0'

        # New: Handle predicted_pitch and predicted_roll
        if 'predicted_pitch' in data:
            coordinates['robot']['predicted_pitch'] = data['predicted_pitch']
        else:
            if 'predicted_pitch' not in coordinates['robot']:
                coordinates['robot']['predicted_pitch'] = '0'

        if 'predicted_roll' in data:
            coordinates['robot']['predicted_roll'] = data['predicted_roll']
        else:
            if 'predicted_roll' not in coordinates['robot']:
                coordinates['robot']['predicted_roll'] = '0'

        flags = {
            'pi1': coordinates['pi1']['flag'],
            'pi2': coordinates['pi2']['flag'],
            'pi3': coordinates['pi3']['flag'],
            'pi4': coordinates['pi4']['flag']
        }
        if all(flag == '0' for flag in flags.values()) and not all(flag == '0' for flag in previous_flags.values()):
            # All flags are now '0', and they were not all '0' before
            movement_log['ending_point'] = coordinates['robot'].copy()
            compute_relative_error()
            log_entry('robot_data_after_move', {
                'robot': coordinates['robot'],
                'target': coordinates['target'],
                'model_input_target': coordinates['model_input_target']  # Record new variable
            })
        previous_flags = flags.copy()
        calculate_error()
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid data"})


@app.route("/get_coordinates", methods=["GET"])
def get_coordinates():
    """
    Get current coordinates.
    """
    return jsonify(coordinates)


def calculate_error():
    """
    Calculate the current error between robot and target.
    """
    try:
        current = np.array(
            [float(coordinates['robot']['x']), float(coordinates['robot']['y']), float(coordinates['robot']['z'])])
        target = np.array(
            [float(coordinates['target']['x']), float(coordinates['target']['y']), float(coordinates['target']['z'])])
        error = np.linalg.norm(current - target)
        coordinates['error'] = str(error)
    except Exception as e:
        print(f"Error calculating error: {e}")


def compute_relative_error():
    """
    Compute the relative error based on movement.
    """
    try:
        starting_point = movement_log.get('starting_point')
        ending_point = coordinates['robot']
        target_point = coordinates['target']

        if not starting_point or not target_point:
            return

        sp = np.array([float(starting_point['x']), float(starting_point['y']), float(starting_point['z'])])
        ep = np.array([float(ending_point['x']), float(ending_point['y']), float(ending_point['z'])])
        tp = np.array([float(target_point['x']), float(target_point['y']), float(target_point['z'])])

        desired_distance = np.linalg.norm(tp - sp)
        current_error = np.linalg.norm(tp - ep)

        if desired_distance == 0:
            relative_error = 0.0
        else:
            relative_error = (current_error / desired_distance) * 100.0

        coordinates['relative_error'] = str(relative_error)
    except Exception as e:
        print("Error computing relative error:", e)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
