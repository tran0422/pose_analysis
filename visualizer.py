import csv
import cv2
import numpy as np

# CONFIG
CSV_FILE = "pose_normalized.csv"
FPS_DELAY = 30  # ms
ZOOM_FACTOR = 1.2

# ----------------------------
# JOINTS
# ----------------------------
JOINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# ----------------------------
# EDGES (skeleton)
# ----------------------------
EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# ----------------------------
# HELPERS
# ----------------------------
def get_joint(row, name):
    x = float(row[f"{name}_x"])
    y = float(row[f"{name}_y"])
    return np.array([x, y])


def to_pixel(pt, center, torso_height):
    p = pt - center
    px = int((p[0] / torso_height) * (100 * ZOOM_FACTOR) + 320)
    py = int((p[1] / torso_height) * (100 * ZOOM_FACTOR) + 240)
    return px, py


# ----------------------------
# LOAD CSV
# ----------------------------
with open(CSV_FILE, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("ROWS LOADED:", len(rows))

# ----------------------------
# VISUAL LOOP (ANNOTATION MODE)
# ----------------------------
frame_idx = 0
selected_frames = set()
paused = False

print("\nControls:")
print("SPACE = mark/unmark frame")
print("P = pause/play")
print("A = previous frame")
print("D = next frame")
print("Q = quit + save\n")

while frame_idx < len(rows):

    row = rows[frame_idx]

    # Load joints
    joints = {j: get_joint(row, j) for j in JOINTS}

    l_shoulder = joints["left_shoulder"]
    r_shoulder = joints["right_shoulder"]
    l_hip = joints["left_hip"]
    r_hip = joints["right_hip"]

    # Stable torso center
    torso_center = (l_shoulder + r_shoulder + l_hip + r_hip) / 4.0

    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0

    torso_height = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_height < 1e-5:
        torso_height = 1.0

    # Canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw bones
    for a, b in EDGES:
        ax, ay = to_pixel(joints[a], torso_center, torso_height)
        bx, by = to_pixel(joints[b], torso_center, torso_height)
        cv2.line(canvas, (ax, ay), (bx, by), (255, 0, 0), 2)

    # Draw joints
    for name, pt in joints.items():
        px, py = to_pixel(pt, torso_center, torso_height)
        cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)

    # Frame label
    cv2.putText(
        canvas,
        f"Frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Selected indicator
    if frame_idx in selected_frames:
        cv2.putText(
            canvas,
            "SELECTED",
            (450, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Pose Visualizer (Annotation Mode)", canvas)

    key = cv2.waitKey(0 if paused else FPS_DELAY) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Mark / unmark frame
    elif key == ord(" "):
        if frame_idx in selected_frames:
            selected_frames.remove(frame_idx)
            print("Unmarked frame:", frame_idx)
        else:
            selected_frames.add(frame_idx)
            print("Marked frame:", frame_idx)

    # Pause/play
    elif key == ord("p"):
        paused = not paused

    # Step backward
    elif key == ord("a"):
        frame_idx = max(0, frame_idx - 1)
        continue

    # Step forward
    elif key == ord("d"):
        frame_idx = min(len(rows) - 1, frame_idx + 1)
        continue

    # Auto advance if playing
    if not paused:
        frame_idx += 1

cv2.destroyAllWindows()

# ----------------------------
# SAVE ANNOTATIONS
# ----------------------------
with open("selected_frames.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame"])
    for idx in sorted(selected_frames):
        writer.writerow([idx])

print(f"\nSaved {len(selected_frames)} selected frames → selected_frames.csv")