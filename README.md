# CSPR Plant Monitoring System Using Raspberry Pis and NEMA 17 Motors

These codes run on RealSense D435i. There are two different types of code: **client code** (run on Raspberry Pis) and **server code** (run on the computer). Since the system requires communication between the clients and the server, all devices need to be connected to the same network.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [1. Run the Server](#1-run-the-server)
  - [2. Anchor Identification](#2-anchor-identification)
  - [3. Robot Identification](#3-robot-identification)
  - [4. Run Client Scripts](#4-run-client-scripts)
  - [5. Next Steps (Movement and Fine-Tuning)](#5-next-steps-movement-and-fine-tuning)
- [Model Simulation Scripts](#model-simulation-scripts)
- [Conclusion](#conclusion)

---
## Prerequisites

1. **Python 3.x** installed on both the Raspberry Pis and the server computer.
2. **Intel RealSense D435i** camera and corresponding SDK/libraries (if you are using scripts that rely on RealSense).
3. All devices (Raspberry Pis + Server) must be connected to the **same network**.

---

## Usage

Below is the basic workflow of the system:

### 1. Run the Server

1. On your server (PC or a more powerful machine), first run  
   [`server.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/server.py).  
   It will print out a URL (e.g., `http://<your_server_ip>:5000`) — this is the endpoint for network communication.
2. Keep `server.py` **running in the background** to maintain the connection.

---

### 2. Anchor Identification

1. Run  
   [`anchor_identify.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/anchor_identify.py)  
   on the server.  
2. In the 3D visualization window, **click** on the motors of Pi4, Pi3, Pi2, and Pi1 **in this exact sequence** (4 → 3 → 2 → 1).  
   - The script uses global registration, so camera rotation is allowed. However, keep rotation minimal for better accuracy.  
3. After identifying all four anchors, `anchor_identify.py` will automatically shut down.  
4. You can see the anchors’ coordinates on the server URL.

---

### 3. Robot Identification

1. There are **two** versions of robot identification:
   - [`robot_identify.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/robot_identify.py): identifies **yellow** markers.
   - [`robot_identify_infrared.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/robot_identify_infrared.py): identifies **infrared** light.
2. Run the appropriate script, then **click** on the item you want to track (e.g., the robot’s center).
3. The script keeps running and updates the “robot coordinate” on the server URL webpage in real-time.

With `server.py` and `robot_identify*.py` running, the system is initialized.

---

### 4. Run Client Scripts

On each **Raspberry Pi**, you have different scripts in the `client_end` folder:

1. [`pi1-4.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/client_end/pi1-4.py)  
   - Controls **four motors**.  
   - Make sure the Pi’s name or motor order **matches** the anchor order from earlier (4 → 3 → 2 → 1).
2. [`pi5_camera.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/client_end/pi5_camera.py)  
   - Automatically takes pictures when the robot stops moving.  
   - The pictures are uploaded to the `uploads` folder on the server.
3. [`pi5_6050.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/client_end/pi5_6050.py)  
   - A **balance sensor** script that uploads the pitch & roll of the robot.  
   - If the user wants to adjust the robot’s posture, they can use `send_adjustments` inside this script.

---

### 5. Next Steps (Movement and Fine-Tuning)

1. **Manual Movement**: Manually enter target coordinates via the server URL.
2. **Automatic Movement**: Run  
   [`50_steps_fixed.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/50_steps_fixed.py)  
   on the server to move the robot **50 times**.  
   - The record of these movements is saved in [`movement_records.json`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/movement_records.json).
3. **Fine-Tuning**: Use  
   [`fine_tune_model.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/fine_tune_model.py)  
   (which also relies on [`calc.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/calc.py) and [`findGreen.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/findGreen.py))  
   to further optimize the robot’s movement and data collection.
4. **Evaluation**: Run  
   [`evaluation.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/evaluation.py)  
   to observe the system’s performance metrics.

---

## Model Simulation Scripts

1. [`model_compute_BFS.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/model_compute_BFS.py)  
   - Computes information (robot posture, cable tension, etc.) across a 3D workspace.  
   - Saves results in CSV files such as  
     [`platform_workspace_data_3m_3m.csv`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/platform_workspace_data_3m_3m.csv)  
     and  
     [`platform_workspace_data_more.csv`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/platform_workspace_data_more.csv).
2. [`model_demonstration.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/model_demonstration.py)  
   - A 3D model graph where the user can adjust X, Y, and Z to visualize the system.
3. [`plot_the_model.py`](https://github.com/YellowSubmarine1999/cspr/blob/main/server_end/plot_the_model.py)  
   - Plots data from the `platform_workspace_data_*.csv` files to analyze the feasible workspace and tension data.

---


## Conclusion

- **Server**: Run `server.py`, then run `anchor_identify.py` and `robot_identify*.py` to initialize.
- **Client (Raspberry Pis)**: Run `pi1-4.py`, `pi5_camera.py`, and `pi5_6050.py`.
- Move the robot manually or with `50_steps_fixed.py`, then collect data and used fine-tune model with `fine_tune_model.py`.
- Evaluate with `evaluation.py` or explore 3D simulations with `model_*` scripts.

##Pi Wire Connect
-Connect with 6050  
![image](https://github.com/user-attachments/assets/2e3a8322-14ab-4152-88df-aa90d887cd24)
-Connect with NEMA 17 driver 
![image](https://github.com/user-attachments/assets/33acc667-5320-481c-b710-b78364b14abe)

---

