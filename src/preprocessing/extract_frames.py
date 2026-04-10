import cv2
import os
from tqdm import tqdm

def extract_frames(video_dir, output_dir, resize=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.avi', '.mp4'))]

    for video_file in tqdm(video_files, desc="Extracting frames"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        frame_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(frame_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, resize)
            frame_path = os.path.join(frame_output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame_resized)
            frame_idx += 1

        cap.release()

    print(f"Done. Frames saved to {output_dir}")


if __name__ == "__main__":
    extract_frames(
        video_dir="data/raw/shanghaitech/training/videos",
        output_dir="data/processed/frames/shanghaitech/training"
    )