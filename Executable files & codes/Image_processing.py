from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls
import cv2
import time

# =====================================
# LOAD MODELS
# =====================================
front_model = YOLO("/home/team03/best(1).pt")
back_model = YOLO("best (4).pt")

# =====================================
# EXPECTED COMPONENTS (FRONT)
# =====================================
EXPECTED = {
    "push_button": 7,
    "resistor": 8,
    "IC": 1,
    "terminal": 1
}

# Ignore good_pcb and bad_pcb tags
DETECT_CLASSES = [ 1, 5, 6, 7]

# =====================================
# CAMERA PROFILES
# =====================================
def set_front_camera(camera_instance):
    print("Switching Camera to FRONT profile...")
    # Fully restore automatic controls to clear manual overrides
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
    # Step 1: Turn off AE first to let libcamera catch the state change
    camera_instance.set_controls({"AeEnable": False, "AwbEnable": False})
    time.sleep(0.1) 
    
    # Step 2: Apply calibrated manual exposure with lower initial gains
    camera_instance.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 5.0,
        "Sharpness": 3.0,
        "Contrast": 1.2,
        "Saturation": 1.2
    })
# =====================================
# INITIALIZE HARDWARE
# =====================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    buffer_count=4
)
picam2.configure(config)
picam2.start()

print("Camera warming up...")
time.sleep(2)
# Start in FRONT inspection state
mode = "FRONT"
set_front_camera(picam2)

# =====================================
# STATE CONTROL SETTINGS
# =====================================
DETECTION_INTERVAL = 2
HOLD_TIME = 5  # Seconds a good PCB must stay present to trigger switch

frame_count = 0
last_front_results = None
last_front_detection_time = time.time()

# Smooth state transition anchors
good_detected_started = False
good_start_timestamp = None

prev_time = time.time()

print("=========================================")
print("PCB Inspection System Online")
print("Press Q on display to quit")
print("=========================================")

# =====================================
# MAIN PROCESSING LOOP
# =====================================
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    frame_count += 1
    current_time = time.time()
    annotated = frame.copy()

    # -------------------------------------------------
    # PROCESSING STATE: FRONT SIDE INSPECTION
    # -------------------------------------------------
    if mode == "FRONT":
        # Periodic model evaluation logic
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

        # Retain last inferred window properties inside the hold parameters
        if last_front_results is not None and (current_time - last_front_detection_time) < HOLD_TIME:
            results = last_front_results
        else:
            results = None

        counts = {
            "push_button": 0,
            "resistor": 0,
            "IC": 0,
            "terminal": 0
        }

        if results is not None:
            annotated = results[0].plot()
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls.item())
                    name = front_model.names[cls]
                    if name in counts:
                        counts[name] += 1

        # Check structural completeness
        pcb_status = "GOOD PCB"
        for component, expected in EXPECTED.items():
            if counts[component] < expected:
                pcb_status = "BAD PCB"
                break
                
        # Render Information Center Panel Dashboard
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

        # Handle State Progression Timers
        if pcb_status == "GOOD PCB":
            status_color = (0, 255, 0)
            if not good_detected_started:
                good_detected_started = True
                good_start_timestamp = current_time

            elapsed_hold = current_time - good_start_timestamp
            remaining_hold = max(0.0, HOLD_TIME - elapsed_hold)
            
            # Show Countdown Timer text
            cv2.putText(annotated, f"Flipping in: {remaining_hold:.1f}s", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if elapsed_hold >= HOLD_TIME:
                print(">>> FRONT INSPECTION VERIFIED. TRANSITIONING TO BACK SIDE... <<<")
                mode = "BACK"
                set_back_camera(picam2)
                time.sleep(1.5)  # Allow mechanical frame shift and settings sync
                prev_time = time.time()  # Prevent extreme frametime spikes from sleep
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
            conf=0.09,
            iou=0.30,
            imgsz=640,
            verbose=False
        )

        hole_count = 0

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # Physical artifact clipping filter
                if w > 100 or h > 100:
                    continue

                hole_count += 1
                conf = float(box.conf[0])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # UI Overlay Panel for BACK Side Verification Output
        cv2.putText(annotated, "BACK PCB INSPECTION", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated, f"True Holes: {hole_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # =====================================
    # PERFORMANCE METRICS OVERLAY
    # =====================================
    current_fps_time = time.time()
    fps = 1 / (current_fps_time - prev_time)
    prev_time = current_fps_time

    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # =====================================
    # CORE RENDER ENGINE FRAME EXIT
    # =====================================
    cv2.imshow("PCB Inspection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup handlers
picam2.stop()
cv2.destroyAllWindows()
print("System Shutdown Complete.")
