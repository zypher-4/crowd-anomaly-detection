import cv2
import numpy as np
import os
from tqdm import tqdm


def numerical_sort_key(filename):
    name = os.path.splitext(filename)[0]
    # For ShanghaiTech training frames named 'frame_00001', strip the prefix
    name = name.replace("frame_", "")
    try:
        return int(name)
    except ValueError:
        return 0


def is_frame_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))


def is_gt_folder(folder_name):
    return folder_name.endswith('_gt')


def extract_optical_flow(frames_dir, output_dir, resize=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)

    video_folders = [
        f for f in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, f)) and not is_gt_folder(f)
    ]

    if not video_folders:
        print(f"No valid folders found in {frames_dir}")
        return

    for video_name in tqdm(video_folders, desc=f"Optical flow: {frames_dir}"):
        video_frame_dir = os.path.join(frames_dir, video_name)
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        # Get frames, sorted numerically
        frame_files = sorted(
            [f for f in os.listdir(video_frame_dir) if is_frame_file(f)],
            key=numerical_sort_key
        )

        if len(frame_files) < 2:
            print(f"Skipping {video_name} — not enough frames")
            continue

        flow_magnitudes = []
        flow_directions = []
        flow_uv_list = []

        # Read first frame
        first_path = os.path.join(video_frame_dir, frame_files[0])
        prev_frame = cv2.imread(first_path)
        prev_frame = cv2.resize(prev_frame, resize)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        for frame_file in frame_files[1:]:
            frame_path = os.path.join(video_frame_dir, frame_file)
            curr_frame = cv2.imread(frame_path)
            curr_frame = cv2.resize(curr_frame, resize)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            flow_magnitudes.append(magnitude)
            flow_directions.append(angle)
            flow_uv_list.append(flow)

            prev_gray = curr_gray

        np.save(os.path.join(video_out_dir, "flow_magnitude.npy"), np.array(flow_magnitudes))
        np.save(os.path.join(video_out_dir, "flow_direction.npy"), np.array(flow_directions))
        np.save(os.path.join(video_out_dir, "flow_uv.npy"), np.array(flow_uv_list))

    print(f"Done. Optical flow saved to {output_dir}")


if __name__ == "__main__":
    jobs = [
        # ShanghaiTech training — from your extracted frames
        (
            "data/processed/frames/shanghaitech/training",
            "data/processed/optical_flow/shanghaitech/training"
        ),
        # ShanghaiTech testing — frames already exist in raw
        (
            "data/raw/shanghaitech/testing/frames",
            "data/processed/optical_flow/shanghaitech/testing"
        ),
        # UCSD Ped1 training
        (
            "data/raw/ucsd/UCSDped1/Train",
            "data/processed/optical_flow/ucsd/ped1/training"
        ),
        # UCSD Ped1 testing (skips _gt folders automatically)
        (
            "data/raw/ucsd/UCSDped1/Test",
            "data/processed/optical_flow/ucsd/ped1/testing"
        ),
        # UCSD Ped2 training
        (
            "data/raw/ucsd/UCSDped2/Train",
            "data/processed/optical_flow/ucsd/ped2/training"
        ),
        # UCSD Ped2 testing
        (
            "data/raw/ucsd/UCSDped2/Test",
            "data/processed/optical_flow/ucsd/ped2/testing"
        ),
    ]

    for frames_dir, output_dir in jobs:
        if os.path.exists(frames_dir):
            print(f"\nProcessing: {frames_dir}")
            extract_optical_flow(frames_dir, output_dir)
        else:
            print(f"Skipping (not found): {frames_dir}")