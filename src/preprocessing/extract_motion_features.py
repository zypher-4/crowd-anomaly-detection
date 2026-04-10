import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def extract_motion_features(flow_dir, output_csv):
    records = []

    for video_name in tqdm(os.listdir(flow_dir), desc=f"Motion features: {flow_dir}"):
        video_flow_dir = os.path.join(flow_dir, video_name)
        if not os.path.isdir(video_flow_dir):
            continue

        mag_path = os.path.join(video_flow_dir, "flow_magnitude.npy")
        dir_path = os.path.join(video_flow_dir, "flow_direction.npy")

        if not os.path.exists(mag_path):
            continue

        magnitudes = np.load(mag_path)   # (N-1, H, W)
        directions = np.load(dir_path)   # (N-1, H, W)

        for frame_idx in range(len(magnitudes)):
            mag = magnitudes[frame_idx]
            dirn = directions[frame_idx]

            records.append({
                "video": video_name,
                "frame": frame_idx,
                "mean_speed": float(np.mean(mag)),
                "max_speed": float(np.max(mag)),
                "direction_variance": float(np.var(dirn)),
                "motion_coverage": float(np.mean(mag > 1.0))
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    print(df.describe())


if __name__ == "__main__":
    jobs = [
        (
            "data/processed/optical_flow/shanghaitech/training",
            "data/processed/motion_features/shanghaitech_train_features.csv"
        ),
        (
            "data/processed/optical_flow/shanghaitech/testing",
            "data/processed/motion_features/shanghaitech_test_features.csv"
        ),
        (
            "data/processed/optical_flow/ucsd/ped1/training",
            "data/processed/motion_features/ucsd_ped1_train_features.csv"
        ),
        (
            "data/processed/optical_flow/ucsd/ped1/testing",
            "data/processed/motion_features/ucsd_ped1_test_features.csv"
        ),
        (
            "data/processed/optical_flow/ucsd/ped2/training",
            "data/processed/motion_features/ucsd_ped2_train_features.csv"
        ),
        (
            "data/processed/optical_flow/ucsd/ped2/testing",
            "data/processed/motion_features/ucsd_ped2_test_features.csv"
        ),
    ]

    for flow_dir, output_csv in jobs:
        if os.path.exists(flow_dir):
            print(f"\nProcessing: {flow_dir}")
            extract_motion_features(flow_dir, output_csv)
        else:
            print(f"Skipping (not found): {flow_dir}")