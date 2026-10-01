import os
import csv
import time
import serial
import serial.tools.list_ports
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls

# Try importing gpiozero for Raspberry Pi 5 GPIO controls
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: gpiozero library not found. Falling back to keyboard inputs for simulation.")

# =====================================
# PATHS AND CALIBRATIONS
# =====================================
FRONT_MODEL_PATH = "/home/team03/best(1).pt"
BACK_MODEL_PATH = "best (4).pt"
VECTOR_FILE_PATH = "pcb_vector.csv"

# Physical button pin mapping (GPIO26 corresponds to Physical Pin 37)
# Connect the other terminal of the push button to a GND pin (e.g. Physical Pin 39)
START_BUTTON_PIN = 26

# Camera pixel to millimeter calibration factor at 26cm distance
# Adjust this to match your physical field of view.
# E.g., if a 1280px wide image covers 192mm of width, PIXEL_TO_MM = 192 / 1280 = 0.15
PIXEL_TO_MM = 0.15

# Physical offsets (in mm) from camera center axis to soldering tip axis
CAMERA_OFFSET_X = 40.0
CAMERA_OFFSET_Y = 100.0

# Camera alignment reference position (where camera center points when inspecting the board)
# The coordinates of X and Y axes when the camera is positioned directly above the PCB origin
X_CAM_REF = 100.0
Y_CAM_REF = 100.0

# Expected component counts for front PCB inspection
EXPECTED = {
    "push_button": 7,
    "resistor": 10,
    "IC": 1,
    "terminal": 1
}

# YOLO Class index filters (exclude good_pcb and bad_pcb tags)
DETECT_CLASSES = [1, 5, 6, 7]

# =====================================
# SERIAL AUTO-DISCOVERY HANDSHAKING
# =====================================
def discover_serial_ports():
    print("Scanning USB ports by hardware descriptors...")
    ports = serial.tools.list_ports.comports()
    esp32_port = None
    uno_port = None
    
    all_filtered_ports = []
    
    for port in ports:
        desc = port.description.lower()
        hwid = port.hwid.lower()
        dev = port.device.lower()
        
        # Filter out Pi's internal Bluetooth/UART serial ports (ttyAMA*)
        if "ama" in dev:
            continue
            
        all_filtered_ports.append(port.device)
        print(f"Detected port: {port.device} (Desc: {port.description}, HWID: {port.hwid})")
        
        # Identify Arduino Uno explicitly (standard USB VID:PID is 2341:0043 or similar, name contains arduino/uno)
        if "arduino" in desc or "uno" in desc or "2341" in hwid or "acm" in dev:
            uno_port = port.device
            print(f"-> Matched Arduino Uno on {port.device}")
            
    # Identify ESP32: Standard ESP32 dev boards use CP210x, CH340, or PL2303 chips.
    # Fallback: if we found an Arduino, the ESP32 is the other USB port.
    for p in all_filtered_ports:
        # Check if this port is the ESP32 based on hardware description
        port_info = next(item for item in ports if item.device == p)
        p_desc = port_info.description.lower()
        p_hwid = port_info.hwid.lower()
        
        if p != uno_port:
            if "cp210" in p_desc or "ch340" in p_desc or "usb" in p_desc or "usb" in p:
                esp32_port = p
                print(f"-> Matched ESP32 on {p}")
                break
                
    # Direct fallback if one matches but the other doesn't
    if uno_port and not esp32_port:
        for p in all_filtered_ports:
            if p != uno_port:
                esp32_port = p
                print(f"-> Fallback: Assuming ESP32 on {p}")
                break
    elif esp32_port and not uno_port:
        for p in all_filtered_ports:
            if p != esp32_port:
                uno_port = p
                print(f"-> Fallback: Assuming Arduino Uno on {p}")
                break
                
    return esp32_port, uno_port

# =====================================
# CAMERA PROFILES
# =====================================
def set_front_camera(camera_instance):
    print("Switching Camera to FRONT profile...")
    camera_instance.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 4.0,
        "Sharpness": 2.0,
        "Contrast": 1.0,
        "Saturation": 1.0
    })

def set_back_camera(camera_instance):
    print("Switching Camera to BACK profile...")
    camera_instance.set_controls({"AeEnable": False, "AwbEnable": True})
    time.sleep(0.1) 
    camera_instance.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 3.8,
        "AeEnable": False,
        "ExposureTime": 8000,
        "AnalogueGain": 2.0,
        "AwbEnable": True,
        "Sharpness": 4.5,
        "Contrast": 1.2,
        "Saturation": 1.0
    })

# =====================================
# LOAD NOMINAL VECTOR FILE COORDINATES
# =====================================
def load_pcb_vectors(filepath):
    vectors = []
    if not os.path.exists(filepath):
        print(f"Warning: Vector file {filepath} not found. Solder coordinates will rely on raw camera frames.")
        return vectors
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vectors.append({
                    "component": row["Component"],
                    "pin": row["Pin"],
                    "x": float(row["X"]),
                    "y": float(row["Y"])
                })
        print(f"Loaded {len(vectors)} nominal coordinates from {filepath}")
    except Exception as e:
        print(f"Error reading vector file: {e}")
    return vectors

# =====================================
# COORDINATE REGISTRATION (AFFINE ALIGNMENT)
# =====================================
def register_coordinates(detected_holes, nominal_vectors):
    """
    Computes an Affine Transform between detected hole pixel coordinates 
    and the design coordinates from the vector file.
    """
    if len(detected_holes) < 3 or len(nominal_vectors) < 3:
        print("Insufficient points for coordinate registration. Falling back to direct pixel mapping.")
        return None

    # Extract X, Y design coords
    nom_pts = np.array([[v['x'], v['y']] for v in nominal_vectors], dtype=np.float32)
    nom_pts_sorted = nom_pts[np.lexsort((nom_pts[:, 1], nom_pts[:, 0]))]
    
    # Sort detected pixel coordinates
    det_pts = np.array(detected_holes, dtype=np.float32)
    det_pts_sorted = det_pts[np.lexsort((det_pts[:, 1], det_pts[:, 0]))]
    
    min_len = min(len(nom_pts_sorted), len(det_pts_sorted))
    pts_nom = nom_pts_sorted[:min_len]
    pts_det = det_pts_sorted[:min_len]
    
    # Estimate partial affine transform
    matrix, inliers = cv2.estimateAffinePartial2D(pts_nom, pts_det, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    
    if matrix is not None:
        print("Registration Successful. Calibration Matrix Computed.")
        return matrix
    else:
        print("Registration Estimation Failed.")
        return None

# =====================================
# MAIN SYSTEM EXECUTION
# =====================================
def main():
    # 1. Discover USB Serial interfaces
    esp_port, uno_port = discover_serial_ports()
    if not esp_port or not uno_port:
        print("Error: Could not identify both microcontrollers. Check wiring/USB links.")
        return

# 2. Establish persistent Serial connections
    print(f"Opening Serial Ports: ESP32={esp_port}, Uno={uno_port}...")
    try:
        # Optimized ESP32 connection with hardware flow control overrides
        ser_esp = serial.Serial(
            port=esp_port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2.0,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        # Force toggle DTR and RTS lines to wake up the CP2102 chip
        ser_esp.setDTR(False)
        ser_esp.setRTS(False)
        time.sleep(0.1)
        ser_esp.setDTR(True)
        ser_esp.setRTS(True)

        # Standard Arduino Uno connection
        ser_uno = serial.Serial(uno_port, 115200, timeout=2.0)
    except serial.SerialException as e:
    
        # Clear buffers
        ser_esp.reset_input_buffer()
        ser_uno.reset_input_buffer()

    # 3. Configure Hardware Start Button
    start_button = None
    if GPIO_AVAILABLE:
        try:
            # Active LOW button (internal pull-up enabled)
            start_button = Button(START_BUTTON_PIN, pull_up=True)
            print(f"Physical start button active on GPIO {START_BUTTON_PIN} (Physical pin 37).")
        except Exception as e:
            print(f"Could not initialize GPIO Button: {e}. Falling back to keyboard.")

    # Home Arduino Uno once at boot (to establish coordinates)
   # Wait for Arduino Uno to finish booting up completely
    print("Waiting for Arduino Uno readiness signal...")
    while True:
        line = ser_uno.readline().decode('utf-8', errors='ignore').strip()
        if "UNO_CONTROLLER_READY" in line:
            print("-> Arduino Uno is ready.")
            break
        time.sleep(0.1)

    # Home Arduino Uno once at boot (to establish coordinates)
    print("Homing CNC Cartesian axes...")
    ser_uno.write(b"HOME\n")
    while True:
        line = ser_uno.readline().decode('utf-8', errors='ignore').strip()
        if "HOMED_SUCCESS" in line:
            print("CNC Cartesian axes homed successfully.")
            break
        elif "HOMING_STARTED" in line:
            continue
        elif "ERROR" in line:
            print("Error: Arduino Uno homing failed.")
            return
        time.sleep(0.1)
    # Now it is safe to home
    print("Homing CNC Cartesian axes...")
    ser_uno.write(b"HOME\n")
    while True:
        line = ser_uno.readline().decode('utf-8', errors='ignore').strip()
        if "HOMED_SUCCESS" in line:
            print("CNC Cartesian axes homed successfully.")
            break
        elif "HOMING_STARTED" in line:
            continue # Skip processing message
        elif "ERROR" in line:
            print("Error: Arduino Uno homing failed.")
            return
        time.sleep(0.1)

    # 4. Load Models & Nominal Vectors
    print("Loading AI YOLOv8 models...")
    front_model = YOLO(FRONT_MODEL_PATH)
    back_model = YOLO(BACK_MODEL_PATH)
    nominal_vectors = load_pcb_vectors(VECTOR_FILE_PATH)

    # 5. Initialize Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720)}, buffer_count=4)
    picam2.configure(config)
    picam2.start()

    # Dynamic Cycle Loop
    while True:
        # Wait for user input to start next PCB cycle
        print("\n=======================================================")
        print("SYSTEM IDLE. Ready for next cycle.")
        if start_button is not None:
            print(f"--> Press the physical PUSH BUTTON on GPIO {START_BUTTON_PIN} to start.")
        else:
            print("--> Press ENTER in this terminal to start.")
        print("=======================================================")

        # Wait loop (non-blocking window to keep GUI thread responsive if window is open)
        if start_button is not None:
            while not start_button.is_pressed:
                # Show an idle screen
                idle_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(idle_frame, "SYSTEM IDLE - PRESS BUTTON TO START", (200, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("PCB Inspection", idle_frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    print("Exiting controller...")
                    picam2.stop()
                    cv2.destroyAllWindows()
                    ser_esp.close()
                    ser_uno.close()
                    return
        else:
            # Block on keyboard input
            input("Press Enter to continue...")

        print(">>> Cycle Triggered! Initializing hardware & heating... <<<")

        # 6. Turn ON Relays (SMPS & Soldering Iron) via ESP32
        print("Powering ON 24V SMPS & Soldering Iron...")
        ser_esp.write(b"SYSTEM_ON\n")
        resp_esp = ser_esp.readline().decode('utf-8', errors='ignore').strip()
        if "SYSTEM_OK" not in resp_esp:
            print(f"Warning: ESP32 response unexpected: {resp_esp}")

        # 7. Heating Delay (20 Seconds)
        heat_delay = 20.0
        start_heat_time = time.time()
        while time.time() - start_heat_time < heat_delay:
            elapsed = time.time() - start_heat_time
            remaining = max(0.0, heat_delay - elapsed)
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            cv2.putText(frame, f"WARMING UP SOLDERING IRON: {remaining:.1f}s", (100, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            cv2.imshow("PCB Inspection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Safe shut off
                ser_esp.write(b"SYSTEM_OFF\n")
                picam2.stop()
                cv2.destroyAllWindows()
                return

        print("Soldering iron ready. Starting inspection cycle.")
        
        # State control variables for this cycle
        mode = "FRONT"
        set_front_camera(picam2)
        
        DETECTION_INTERVAL = 2
        HOLD_TIME = 5.0
        frame_count = 0
        last_front_results = None
        last_front_detection_time = time.time()
        good_detected_started = False
        good_start_timestamp = None
        prev_time = time.time()
        cycle_active = True

        # Process active PCB
        while cycle_active:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_count += 1
            current_time = time.time()
            annotated = frame.copy()

            # -------------------------------------------------
            # PROCESSING STATE: FRONT SIDE INSPECTION
            # -------------------------------------------------
            if mode == "FRONT":
                if frame_count % DETECTION_INTERVAL == 0 or last_front_results is None:
                    detections = front_model.predict(
                        source=frame,
                        imgsz=960,
                        conf=0.15,
                        iou=0.40,
                        classes=DETECT_CLASSES,
                        verbose=False
                    )
                    if detections is not None:
                        last_front_results = detections
                        last_front_detection_time = current_time

                if last_front_results is not None and (current_time - last_front_detection_time) < HOLD_TIME:
                    results = last_front_results
                else:
                    results = None

                counts = {"push_button": 0, "resistor": 0, "IC": 0, "terminal": 0}

                if results is not None:
                    annotated = results[0].plot()
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            cls = int(box.cls.item())
                            name = front_model.names[cls]
                            if name in counts:
                                counts[name] += 1

                # Validate component numbers
                pcb_status = "GOOD PCB"
                for component, expected in EXPECTED.items():
                    if counts[component] < expected:
                        pcb_status = "BAD PCB"
                        break
                        
                # Render Dashboard
                panel_x = 900
                cv2.rectangle(annotated, (panel_x, 0), (1280, 720), (40, 40, 40), -1)
                cv2.putText(annotated, "FRONT DASHBOARD", (panel_x + 10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                y_offset = 90
                dashboard_data = [
                    ("PushBtn", counts["push_button"], EXPECTED["push_button"]),
                    ("Resistor", counts["resistor"], EXPECTED["resistor"]),
                    ("IC", counts["IC"], EXPECTED["IC"]),
                    ("Terminal", counts["terminal"], EXPECTED["terminal"])
                ]

                for name, actual, expected in dashboard_data:
                    text_color = (0, 255, 0) if actual >= expected else (0, 0, 255)
                    cv2.putText(annotated, f"{name}: {actual}/{expected}", (panel_x + 10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    y_offset += 50

                if pcb_status == "GOOD PCB":
                    status_color = (0, 255, 0)
                    if not good_detected_started:
                        good_detected_started = True
                        good_start_timestamp = current_time

                    elapsed_hold = current_time - good_start_timestamp
                    remaining_hold = max(0.0, HOLD_TIME - elapsed_hold)
                    
                    cv2.putText(annotated, f"Flipping in: {remaining_hold:.1f}s", (panel_x + 10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if elapsed_hold >= HOLD_TIME:
                        print(">>> PCB VERIFIED. FLIPPING TO BACK SIDE... <<<")
                        cv2.putText(annotated, "FLIPPING PCB...", (200, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 4)
                        cv2.imshow("PCB Inspection", annotated)
                        cv2.waitKey(1)
                        
                        # Trigger Servo Flip on Uno
                        ser_uno.write(b"FLIP\n")
                        while True:
                            line = ser_uno.readline().decode('utf-8', errors='ignore').strip()
                            if "FLIPPED" in line:
                                break
                            time.sleep(0.05)
                        
                        mode = "BACK"
                        set_back_camera(picam2)
                        time.sleep(1.5)  # Wait for mechanical settlement
                        prev_time = time.time()
                        continue
                else:
                    status_color = (0, 0, 255)
                    good_detected_started = False
                    good_start_timestamp = None

                cv2.putText(annotated, pcb_status, (panel_x + 20, y_offset + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

            # -------------------------------------------------
            # PROCESSING STATE: BACK SIDE HOLE INSPECTION
            # -------------------------------------------------
            else:
                results = back_model.predict(
                    source=frame,
                    conf=0.05,
                    iou=0.25,
                    imgsz=1024,
                    verbose=False
                )

                detected_holes = []
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1

                        if w > 45 or h > 45:
                            continue

                        conf = float(box.conf[0])
                        center_u = int((x1 + x2) / 2)
                        center_v = int((y1 + y2) / 2)
                        detected_holes.append([center_u, center_v])

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(annotated, (center_u, center_v), 3, (0, 0, 255), -1)
                        cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(annotated, "BACK PCB INSPECTION", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(annotated, f"True Holes: {len(detected_holes)}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Align and solder
                if len(detected_holes) >= 3 and len(nominal_vectors) >= 3:
                    cv2.putText(annotated, "Holes detected! Running alignment registration...", (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("PCB Inspection", annotated)
                    cv2.waitKey(1)
                    
                    affine_mat = register_coordinates(detected_holes, nominal_vectors)
                    
                    if affine_mat is not None:
                        print("Starting soldering path sequence...")
                        for vec in nominal_vectors:
                            nominal_pos = np.array([vec['x'], vec['y'], 1.0], dtype=np.float32)
                            pixel_pos = np.dot(affine_mat, nominal_pos)
                            u_mapped, v_mapped = pixel_pos[0], pixel_pos[1]
                            
                            pixel_x_offset = (u_mapped - 640.0) * PIXEL_TO_MM
                            pixel_y_offset = (v_mapped - 360.0) * PIXEL_TO_MM
                            
                            physical_hole_x = X_CAM_REF + pixel_x_offset
                            physical_hole_y = Y_CAM_REF + pixel_y_offset
                            
                            # Correct for offset
                            target_solder_x = physical_hole_x + CAMERA_OFFSET_X
                            target_solder_y = physical_hole_y + CAMERA_OFFSET_Y
                            
                            cv2.circle(annotated, (int(u_mapped), int(v_mapped)), 6, (255, 0, 0), 2)
                            print(f"Soldering Pin {vec['component']}-{vec['pin']}: X:{target_solder_x:.2f} Y:{target_solder_y:.2f}")
                            
                            cmd = f"SOLDER:X{target_solder_x:.4f}:Y{target_solder_y:.4f}\n"
                            ser_uno.write(cmd.encode('utf-8'))
                            
                            while True:
                                uno_line = ser_uno.readline().decode('utf-8', errors='ignore').strip()
                                if "DONE" in uno_line:
                                    break
                                elif "ERROR" in uno_line:
                                    print(f"Uno motion error: {uno_line}")
                                    break
                                time.sleep(0.01)
                        
                        print("PCB Soldering Run Completed!")
                        
                        # Unflip PCB back to inspection state
                        print("Unflipping PCB...")
                        ser_uno.write(b"UNFLIP\n")
                        while True:
                            line = ser_uno.readline().decode('utf-8', errors='ignore').strip()
                            if "UNFLIPPED" in line:
                                break
                            time.sleep(0.05)
                            
                        # Turn off heating and SMPS at end of cycle
                        print("Deactivating heater and SMPS relays...")
                        ser_esp.write(b"SYSTEM_OFF\n")
                        while True:
                            line = ser_esp.readline().decode('utf-8', errors='ignore').strip()
                            if "SYSTEM_SHUTDOWN" in line:
                                break
                            time.sleep(0.05)
                        
                        cycle_active = False # Break cycle loop to wait for start button press
                        continue

            # Performance FPS Overlay
            current_fps_time = time.time()
            fps = 1 / (current_fps_time - prev_time)
            prev_time = current_fps_time

            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("PCB Inspection", annotated)

            # Check for escape
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Aborting cycle...")
                ser_esp.write(b"SYSTEM_OFF\n")
                ser_uno.write(b"RELEASE\n")
                picam2.stop()
                cv2.destroyAllWindows()
                ser_esp.close()
                ser_uno.close()
                return

if __name__ == "__main__":
    main()
