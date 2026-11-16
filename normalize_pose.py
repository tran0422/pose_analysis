import csv
import math

input_file = "pose_raw_data.csv"
output_file = "pose_normalized.csv"

# Landmark names and order
LANDMARK_NAMES = [
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


def get_triplet(row, idx):
    # Extract x,y,z from flattened CSV row
    base = 1 + idx * 3
    return float(row[base]), float(row[base + 1]), float(row[base + 2])


with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Write header
    header = ["frame"]

    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_z"]

    writer.writerow(header)

    next(reader)  # skip input header

    for row in reader:

        frame_index = row[0]

        # Extract shoulders
        lsx, lsy, lsz = get_triplet(row, 1)
        rsx, rsy, rsz = get_triplet(row, 2)

        cx = (lsx + rsx) / 2
        cy = (lsy + rsy) / 2
        cz = (lsz + rsz) / 2

        # Scale shoulder width
        scale = math.sqrt((lsx - rsx) ** 2 + (lsy - rsy) ** 2)

        # FIX 1: prevent tiny scale explosion
        scale = max(scale, 1e-6)

        # Normalize all landmarks
        normalized_row = [frame_index]

        for i in range(13):
            x, y, z = get_triplet(row, i)

            # center
            x = x - cx
            y = y - cy
            z = z - cz

            # scale ONLY x and y (IMPORTANT FIX)
            nx = x / scale
            ny = y / scale

            # ✅ FIX 2: do NOT over-scale z
            nz = z / scale

            normalized_row.extend([nx, ny, nz])

        writer.writerow(normalized_row)
