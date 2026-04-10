import cv2
import numpy as np
import os
from tqdm import tqdm


def numerical_sort_key(filename):
    name = os.path.splitext(filename)[0].replace("frame_", "")
    try:
        return int(name)
    except ValueError:
        return 0


def is_frame_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))


def is_gt_folder(folder_name):
    return folder_name.endswith('_gt')


def normalize_frames(input_dir, output_dir, resize=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)

    video_folders = [
        f for f in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, f)) and not is_gt_folder(f)
    ]

    skipped = 0
    for video_name in tqdm(video_folders, desc=f"Normalizing: {input_dir}"):
        save_path = os.path.join(output_dir, f"{video_name}.npy")

        # RESUME CHECK — skip if .npy already exists
        if os.path.exists(save_path):
            skipped += 1
            continue

        video_frame_dir = os.path.join(input_dir, video_name)
        frame_files = sorted(
            [f for f in os.listdir(video_frame_dir) if is_frame_file(f)],
            key=numerical_sort_key
        )

        if not frame_files:
            continue

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_frame_dir, frame_file)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, resize)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        frames_array = np.array(frames)
        np.save(save_path, frames_array)

    print(f"Done. Normalized frames saved to {output_dir}")
    print(f"Skipped {skipped} already processed videos")


if __name__ == "__main__":
    jobs = [
        (
            "data/processed/frames/shanghaitech/training",
            "data/processed/normalized/shanghaitech/training"
        ),
        (
            "data/raw/shanghaitech/testing/frames",
            "data/processed/normalized/shanghaitech/testing"
        ),
        ("data/raw/ucsd/UCSDped1/Train", "data/processed/normalized/ucsd/ped1/training"),
        ("data/raw/ucsd/UCSDped1/Test",  "data/processed/normalized/ucsd/ped1/testing"),
        ("data/raw/ucsd/UCSDped2/Train", "data/processed/normalized/ucsd/ped2/training"),
        ("data/raw/ucsd/UCSDped2/Test",  "data/processed/normalized/ucsd/ped2/testing"),
    ]

    for input_dir, output_dir in jobs:
        if os.path.exists(input_dir):
            print(f"\nProcessing: {input_dir}")
            normalize_frames(input_dir, output_dir)
        else:
            print(f"Skipping (not found): {input_dir}")