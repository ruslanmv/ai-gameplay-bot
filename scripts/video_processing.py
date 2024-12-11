import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a gameplay video at the specified frame rate.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        frame_rate (int): Number of frames per second to extract.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frame_rate
    
    count = 0
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        count += 1

    video.release()
    print(f"Extracted {frame_count} frames from {video_path}")

if __name__ == "__main__":
    video_path = "data/raw/gameplay_videos/sample_video.mp4"
    output_dir = "data/processed/frames"
    extract_frames(video_path, output_dir, frame_rate=1)
