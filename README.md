# 3D-Cube-AR
##Authors
Andres Mauricio Avilan
Juan David Meza

A minimal Augmented Reality (AR) project that performs camera calibration, pose estimation, and 3D cube projection using OpenCV and a 9×6 inner-corner chessboard pattern.

This project implements a complete computer vision pipeline:
1. Camera calibration from real chessboard images  
2. Pose estimation using `solvePnP`  
3. Projection of a virtual 3D cube aligned with the chessboard  
4. Real-time augmented-reality visualization

The code is based on the calibration script provided by Professor Flavio Augusto Prieto Ortiz and is expanded with the 3D projection module.
---

## Features
- Chessboard detection (9×6 inner corners)
- Estimation of intrinsic parameters and distortion coefficients
- Recovery of rotation and translation vectors (R, t)
- Construction of the projection matrix `P = K [R | t]`
- Overlay of a 3D cube onto the real image
- Undistortion utilities for testing and analysis

---

## Requirements
- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib  

Install dependencies:

```bash
pip install opencv-python numpy matplotlib

## How to Use

### 1. Capture chessboard images
Use a **9×6 inner-corner chessboard**.  
Take several photos from different angles and distances, then place all images in:
- calibracion\calib_images


Make sure the chessboard is fully visible and fills a reasonable portion of the frame.

---

### 2. Run camera calibration
Execute the calibration script:

```bash
python calibrate_camera.py

### 3. Run the AR cube demo

```bash
python ar_cube.py

