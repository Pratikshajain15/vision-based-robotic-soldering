# Vision-Assisted Robotic Soldering

A vision-assisted robotic system developed for **automated PCB inspection, solder-hole localization, coordinate extraction, and selective soldering**.

The system combines **YOLOv8-based computer vision, image processing, coordinate mapping, A* path planning, embedded control, and a 3-DOF Cartesian robotic platform** to automate the PCB soldering workflow.

---

## 🔍 System Workflow

```text
PCB
 │
 ▼
Front-Side Inspection
 │
 ▼
YOLOv8 Component Detection
 │
 ▼
Component Verification
 │
 ├── BAD PCB ──► Stop
 │
 ▼ GOOD PCB
PCB Flipping
 │
 ▼
Back-Side Hole Detection
 │
 ▼
Hole Centroid Extraction
 │
 ▼
Coordinate Transformation
 │
 ▼
A* Path Optimization
 │
 ▼
3-DOF Cartesian Robot
 │
 ▼
Selective Soldering
```

The system uses a dual-stage vision approach: the front side is inspected for component placement, while the backside is processed to locate soldering holes and generate robotic coordinates.

---

## ⚙️ Key Features

- YOLOv8-based PCB component detection
- Front-side PCB component verification
- Back-side solder-hole localization
- Image-based coordinate extraction
- Pixel-to-workspace coordinate transformation
- A* path optimization for soldering points
- 3-DOF Cartesian robotic motion
- Raspberry Pi-based vision processing
- Arduino-based motion control
- ESP32-based power and heating control

---

## 🧠 Technologies

| Category | Technologies |
|---|---|
| Computer Vision | YOLOv8, OpenCV, Picamera2 |
| Programming | Python, C/C++ |
| Robotics | Cartesian Robot, A* Path Planning |
| Embedded | Raspberry Pi 5, Arduino UNO, ESP32 |
| Simulation | MATLAB, Simulink, System Composer, CoppeliaSim |
| Mechanical Design | SolidWorks |
| Communication | Serial Communication, GPIO |

---

## 🏗️ Hardware

| Component | Function |
|---|---|
| Raspberry Pi 5 | Vision processing and system coordination |
| Arducam Camera | PCB image acquisition |
| Arduino UNO | Cartesian motion control |
| ESP32 | Power and heater control |
| NEMA 17 + A4988 | X/Y-axis motion |
| NEMA 23 + TB6600 | Z-axis motion |
| Proximity Sensors | Homing and travel-limit protection |
| 24 V SMPS | System power |
| Soldering Module | Selective soldering |

The project architecture integrates the vision, control, motion, sensing, and power subsystems into a unified robotic soldering system.

---

## 📊 Results

The project was evaluated using custom PCB datasets and a simulation/physical prototype workflow.

### Front PCB Detection

The project report reports a front-side YOLOv8 evaluation of:

- **99.4% mAP@50**
- Detection of PCB components including resistors, push buttons, IC, terminal and capacitors

### Robotic Path Planning

An **A\***-based traversal sequence was used to optimize the soldering path.

The project report reports a cycle time of:

**180.2 seconds for a 60-joint PCB**

---

## 📁 Repository Structure

```text
vision-based-robotic-soldering/
│
├── Back_PCB/
├── Front PCB/
├── Executable files & codes/
│   ├── BackPCB_Hole_Detection.py
│   ├── FrontPCB_Component_Detection.py
│   ├── Image_processing.py
│   ├── Integrated_code.py
│   └── esp32_controller.ino
│
├── Mathworks files/
│   ├── Logical_Architecture.slx
│   ├── MainController_Behavior.slx
│   ├── MotionSystem_Behavior.slx
│   ├── Robotic_Soldering_Physical.slx
│   ├── SolderingSystem_Behavior.slx
│   └── VisionSystem_Behavior.slx
│
├── Output Images/
├── Physical Prototype/
├── Simulation/
├── SolidWorks/
├── .gitignore
└── README.md
```

Large datasets, trained model weights, videos, reports, presentations, team photographs, and selected large files are excluded from the public repository.

---

## 📷 Physical Prototype

![Physical Prototype](Physical%20Prototype/Final%20Prototype/Img%202.jpg)

---

## 🔬 Vision Results

### Front PCB Component Detection

![Front PCB Detection](Output%20Images/FrontPCB%20detection.jpeg)

### Back PCB Hole Detection

![Back PCB Hole Detection](Output%20Images/BackPCB_Holes%20detection.png)

---

## 👥 Team

**Team 03 — KLE Technological University**

Pratiksha Salecha · Shahista Khatib · Shreyas Kubsad · Rishita Kadam · Suchit Patil

**Guide:** Mr. Girish Karikatti
```
