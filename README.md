# Facial Recognition Project

This project provides two main functionalities using face recognition:

1. **Compare two images to check if they contain the same person** (`main.py`)
2. **Mark attendance using webcam face recognition** (`Attendance.py`)

## Requirements

- `python`
- `opencv-python`
- `numpy`
- `face_recognition`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 1. Comparing Two Images (`main.py`)

`main.py` allows you to compare two images and determine if they contain the same person. It prints the match result and face distance, and can optionally display the images with detected faces.

### Usage

```bash
python main.py <image1_path> <image2_path>
```

- `<image1_path>`: Path to the first image file
- `<image2_path>`: Path to the second image file
- Add `--no-show` if you do not want to display the images in a window

#### Example

```bash
python main.py images/AJ.png images/shaw.JPG
```

This will print whether the faces match and the face distance, and show both images with rectangles around detected faces.

## 2. Attendance System (`Attendance.py`)

`Attendance.py` uses your webcam to recognize faces in real time and mark attendance in `Attendance.csv`.

- Place reference images of known people in the `images/` folder. 
- When a known face is detected by the webcam, their name and the current time will be recorded in `Attendance.csv`.

### Usage

```bash
python Attendance.py
```

- The webcam window will open and start recognizing faces.
- Press `Ctrl+C` or close the window to stop.

## Files

- `main.py`: Script to compare two images for face similarity.
- `Attendance.py`: Script for real-time face recognition attendance using webcam.
- `images/`: Folder containing reference images for attendance.
- `Attendance.csv`: Output file for attendance records.
- `requirements.txt`: Python dependencies for the project.

## Notes
- Make sure your images are clear and contain only one face for best results.
- The attendance system assumes each image in `images/` contains a single face.
- If you encounter issues with dlib or face_recognition installation, ensure you are using a compatible Python version (3.8–3.11).