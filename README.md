# Pose Analysis
A Python-based human pose analysis and annotation project using OpenCV and MediaPipe.

# Project Overview
This project processes video recordings to extract, normalize, and visualize human body pose landmarks.

## Processing Pipeline
Pose Extraction (script1_extract_pose.py)
Processes video frames using MediaPipe Pose Landmarker.
Extracts 13 body landmarks.
Exports coordinates to pose_raw_data.csv.
Pose Normalization (script2_normalize_pose.py)
Centers coordinates around the shoulder midpoint.
Normalizes coordinates using shoulder width.
Exports pose_normalized.csv.
Pose Annotation (script3_annotate_frames.py)
Visualizes the normalized human skeleton.
Allows manual frame selection.
Exports selected frames to selected_frames.csv.
Requirements
Python
OpenCV
MediaPipe
NumPy

## Install dependencies:
bash
pip install -r requirements.txt

## Model
The project uses the MediaPipe Pose Landmarker model:
pose_landmarker_full.task

## Project Status
Proof of concept. The processing pipeline is under active development.

## Future Development
Improve pose extraction and normalization.
Improve error handling and data validation.
Develop automated testing.
Introduce browser-based visualization and annotation.
Develop a React frontend and Python backend.
