# Traffic Surveillance Using YOLOv8

A Python-based traffic-surveillance application that detects traffic
participants in uploaded videos using a trained YOLOv8 model. It uses
Gradio for the interface and OpenCV for video processing.

## Features

- Upload traffic videos for object detection.
- Detect objects frame by frame.
- Adjust the detection confidence threshold.
- Display bounding boxes and class labels.
- Preview and download annotated videos.
- Use the trained model bundled with the repository.

## Technology Stack

- Python 3.10 or newer
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Gradio

## Project Structure

```text
surveillance_v4_10k/
├── app.py
├── run.py
├── run_linux.sh
├── run_windows.bat
├── best.pt
├── yolov8n.pt
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data.yaml
├── index.html
└── metadata/
    ├── CLASS_STATISTICS.csv
    ├── SELECTION_MANIFEST.csv
    ├── SELECTION_REPORT.txt
    ├── SOURCE_STATISTICS.csv
    └── SPLIT_STATISTICS.csv
```

The dataset directories, training outputs, and virtual environment are
excluded from the repository. They are not needed for inference with
the included trained checkpoint.

## Requirements

- Python 3.10 or newer
- Internet access for initial dependency installation
- Sufficient disk space for dependencies and the model

## Download and Run

1. On GitHub, select **Code → Download ZIP**.
2. Extract the ZIP and keep the project files together.
3. Install Python 3.10 or newer if it is not already installed.
4. Run the appropriate launcher for your operating system.

### Windows

Double-click `run_windows.bat`.

Alternatively, open Command Prompt or PowerShell in the project
folder and run:

```bat
py -3 run.py
```

If the Python launcher is unavailable, try:

```bat
python run.py
```

### Linux

Install Python 3.10 or newer and your distribution's virtual-environment
support package if it is provided separately.

Open a terminal in the project folder and run:

```bash
chmod +x run_linux.sh
./run_linux.sh
```

Alternatively:

```bash
python3 run.py
```

### macOS

Install Python 3.10 or newer. Open Terminal in the extracted folder and run:

```bash
python3 run.py
```

The launcher creates a local `.venv` environment and installs the
packages listed in `requirements.txt`. Initial setup requires internet
access.

## Using the Application

1. Wait for Gradio to start.
2. Open the local URL printed in the terminal.
3. Upload a video.
4. Select a confidence threshold.
5. Run detection and preview or download the annotated output.

Press Ctrl+C in the terminal to stop the application.

## Model

`best.pt` is the trained checkpoint used by the application. The
launcher configures `MODEL_PATH` to point to this file, avoiding
dependence on the original developer's filesystem paths.

`yolov8n.pt` is the base checkpoint and is not a substitute for the
trained checkpoint.

## Dataset

The dataset-selection report describes 10,000 selected images:

- Training: 8,000 images
- Validation: 1,000 images
- Testing: 1,000 images

The six classes are:

| ID | Class |
|---:|---|
| 0 | `threew` |
| 1 | `car` |
| 2 | `h_vehicle` |
| 3 | `person` |
| 4 | `two` |
| 5 | `unclassified` |

The report records 112,770 bounding boxes. Dataset statistics and
selection information are available in `metadata/`.

The actual training images and labels are not included. They are
required for retraining, but not for inference with `best.pt`.

## CPU and GPU

CPU inference may be used. GPU acceleration depends on compatible
hardware, drivers, and the installed PyTorch configuration. The
launcher does not install GPU drivers or guarantee GPU acceleration.

## Troubleshooting

- **Python not found:** Install Python and ensure it is available in
  your terminal.
- **Environment creation fails:** Install virtual-environment support
  for your Python installation.
- **Dependency installation fails:** Check internet connectivity and
  Python/package compatibility.
- **Model loading fails:** Confirm that `best.pt` exists and is valid.
- **Video processing fails:** Try a standard MP4 file.
- **Gradio does not start:** Check the terminal output for errors or
  a port conflict.

## Redistribution

Check the applicable redistribution terms for the trained model,
base weights, source datasets, and third-party dependencies before
redistributing them. No independent test-performance or accuracy
claim is made in this README.
