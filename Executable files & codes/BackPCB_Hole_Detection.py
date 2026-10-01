from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls
import cv2
import time

# =====================================
# Load Hole Detection Model
# =====================================
model = YOLO("best (4).pt")

# =====================================
# Camera Setup
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

# =====================================
# IMX708 Image Quality Settings
# =====================================
picam2.set_controls({

        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 0.1,
        "ExposureTime": 8000,       # Slightly lowered to prevent over-exposure
        "AnalogueGain": 1.5,        # Dropped from 5.0 to stop rapid whiteouts
        "ColourGains": (1.5, 1.5),
        "Sharpness": 2.5,
        "Contrast": 1.1,
        "Saturation": 1.0
})

print("PCB Hole Detection Started")
print("Press Q to quit")

# =====================================
# FPS Counter
# =====================================
prev_time = time.time()

while True:

    frame = picam2.capture_array()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # =====================================
    # YOLO Detection
    # =====================================
    results = model.predict(
        source=frame,
        conf=0.09,
        iou=0.30,
        imgsz=640,      # Faster than 1280
        verbose=False
    )

    display_frame = frame.copy()

    hole_count = 0

    if len(results) > 0 and len(results[0].boxes) > 0:

        for box in results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            # Your original hole filter
            if w > 100 or h > 100:
                continue

            hole_count += 1

            conf = float(box.conf[0])

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    # =====================================
    # FPS
    # =====================================
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"True Holes: {hole_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "PCB Hole Detection - IMX708",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
