import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import csv

# load the trained model
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")

# Run it as video; look for one person in the frame
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1
)

# Create the model
landmarker = vision.PoseLandmarker.create_from_options(options)

# Tell opencv to load the video
video_path = "20250723_200057.mp4"  # replace with actual name of the video
cap = cv2.VideoCapture(video_path)

# Give the starting point of the video a frame of 0
frame_index = 0

# Create csv
csv_file = open("pose_raw_data.csv", "w", newline="")
writer = csv.writer(csv_file)

# Populate csv header
writer.writerow(
    [
        "frame",
        "nose_x",
        "nose_y",
        "nose_z",
        "left_shoulder_x",
        "left_shoulder_y",
        "left_shoulder_z",
        "right_shoulder_x",
        "right_shoulder_y",
        "right_shoulder_z",
        "left_elbow_x",
        "left_elbow_y",
        "left_elbow_z",
        "right_elbow_x",
        "right_elbow_y",
        "right_elbow_z",
        "left_wrist_x",
        "left_wrist_y",
        "left_wrist_z",
        "right_wrist_x",
        "right_wrist_y",
        "right_wrist_z",
        "left_hip_x",
        "left_hip_y",
        "left_hip_z",
        "right_hip_x",
        "right_hip_y",
        "right_hip_z",
        "left_knee_x",
        "left_knee_y",
        "left_knee_z",
        "right_knee_x",
        "right_knee_y",
        "right_knee_z",
        "left_ankle_x",
        "left_ankle_y",
        "left_ankle_z",
        "right_ankle_x",
        "right_ankle_y",
        "right_ankle_z",
    ]
)

# Process the frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert to RGB (opencv format) to MediaPipe expected format = RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert image into format MediaPipe expects
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run the model (landmarker) against the image for each frame
    result = landmarker.detect_for_video(mp_image, frame_index)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Print key landmarks
        key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

        # Extract the coordinates and store them in a list
        coords = []
        for i in key_points:
            lm = landmarks[i]
            coords.append((lm.x, lm.y, lm.z))

        # Flatten coords
        row = [frame_index]
        for x, y, z in coords:
            row.extend([x, y, z])

    # For frames without pose detected
    else:
        row = [frame_index] + [None] * (13 * 3)

    writer.writerow(row)

    # Frame progress
    if frame_index % 100 == 0:
        print(f"Processed frame {frame_index}")

    frame_index += 1

    csv_file.flush()

csv_file.close()
cap.release()
