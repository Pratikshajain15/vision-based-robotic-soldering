from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls
import cv2
import time

# =====================================
# MODEL
# =====================================
model = YOLO("/home/team03/best(1).pt")

# =====================================
# EXPECTED COMPONENTS
# =====================================
EXPECTED = {
    "push_button": 7,
    "resistor": 8,
    "electrolytic_capacitor": 1,
    "Ceremic capacitors": 1,
    "IC": 1,
    "terminal": 1
}

# Ignore good_pcb and bad_pcb
DETECT_CLASSES = [0, 1, 3, 5, 6, 7]

# =====================================
# CAMERA SETUP
# =====================================
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    buffer_count=4
)

picam2.configure(config)
picam2.start()

time.sleep(2)

picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 4.0,

    "AeEnable": True,
    "AwbEnable": True,

    "Sharpness": 2.0,
    "Contrast": 1.0
})

# =====================================
# DETECTION SETTINGS
# =====================================
DETECTION_INTERVAL = 2
HOLD_TIME = 5

frame_count = 0
last_results = None
last_detection_time = time.time()

prev_time = time.time()

print("PCB Inspection Started")
print("Press Q to Quit")

# =====================================
# MAIN LOOP
# =====================================
while True:

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_count += 1
    current_time = time.time()

    # =====================================
    # RUN YOLO
    # =====================================
    if frame_count % DETECTION_INTERVAL == 0 or last_results is None:

        detections = model.predict(
            source=frame,
            imgsz=960,
            conf=0.15,
            iou=0.40,
            classes=DETECT_CLASSES,
            verbose=False
        )

        if detections is not None:
            last_results = detections
            last_detection_time = current_time

    # =====================================
    # KEEP LAST DETECTIONS
    # =====================================
    if (
        last_results is not None and
        (current_time - last_detection_time) < HOLD_TIME
    ):
        results = last_results
    else:
        results = None

    # =====================================
    # DISPLAY FRAME
    # =====================================
    annotated = frame.copy()

    counts = {
        "push_button": 0,
        "resistor": 0,
        "electrolytic_capacitor": 0,
        "Ceremic capacitors": 0,
        "IC": 0,
        "terminal": 0
    }

    # =====================================
    # COUNT COMPONENTS
    # =====================================
    if results is not None:

        annotated = results[0].plot()

        if results[0].boxes is not None:

            for box in results[0].boxes:

                cls = int(box.cls.item())
                name = model.names[cls]

                if name in counts:
                    counts[name] += 1

    # =====================================
    # PCB STATUS
    # =====================================
    pcb_status = "GOOD PCB"

    for component, expected in EXPECTED.items():

        actual = counts[component]

        if actual < expected:
            pcb_status = "BAD PCB"
            break

    # =====================================
    # DASHBOARD
    # =====================================
    panel_x = 900

    cv2.rectangle(
        annotated,
        (panel_x, 0),
        (1280, 720),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        annotated,
        "PCB DASHBOARD",
        (panel_x + 10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y = 90

    dashboard = [
        ("PushBtn", counts["push_button"], 7),
        ("Resistor", counts["resistor"], 8),
        ("Electro", counts["electrolytic_capacitor"], 1),
        ("Ceramic", counts["Ceremic capacitors"], 1),
        ("IC", counts["IC"], 1),
        ("Terminal", counts["terminal"], 1)
    ]

    for name, actual, expected in dashboard:

        color = (0, 255, 0)

        if actual < expected:
            color = (0, 0, 255)

        cv2.putText(
            annotated,
            f"{name}: {actual}/{expected}",
            (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        y += 50

    # =====================================
    # PCB STATUS DISPLAY
    # =====================================
    if pcb_status == "GOOD PCB":
        status_color = (0, 255, 0)
    else:
        status_color = (0, 0, 255)

    cv2.putText(
        annotated,
        pcb_status,
        (panel_x + 20, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )

    # =====================================
    # FPS
    # =====================================
    current = time.time()

    fps = 1 / (current - prev_time)

    prev_time = current

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # =====================================
    # SHOW WINDOW
    # =====================================
    cv2.imshow("PCB Inspection", annotated)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =====================================
# CLEANUP
# =====================================
picam2.stop()
cv2.destroyAllWindows()