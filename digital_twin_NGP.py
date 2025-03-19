import os
import cv2
import torch
import numpy as np
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import sys

# Add the path to the Instant-NGP Python bindings
sys.path.append('/content/instant-ngp/build')

# Import the Instant-NGP Python bindings
import pyngp as ngp

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sliding window size for recent frames
SLIDING_WINDOW_SIZE = 10

# Function to prepare data for Instant-NGP
def prepare_instant_ngp_data(frames, workspace_folder):
    # Create the required directory structure
    data_folder = os.path.join(workspace_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)

    # Save frames to the data folder
    for i, frame in enumerate(frames):
        frame_path = os.path.join(data_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)

    # Generate a dummy transforms.json file (replace with actual camera poses if available)
    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": [
            {
                "file_path": f"./frame_{i:04d}.jpg",
                "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            }
            for i in range(len(frames))
        ]
    }

    with open(os.path.join(data_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

# Function to run Instant-NGP for real-time reconstruction
def run_instant_ngp(frames, workspace_folder):
    os.makedirs(workspace_folder, exist_ok=True)

    # Prepare data for Instant-NGP
    prepare_instant_ngp_data(frames, workspace_folder)

    # Load the scene and train the model on recent frames
    scene = ngp.load_scene(f"{workspace_folder}/data")
    ngp.train(scene)

# Main Execution
if __name__ == "__main__":
    # Use live camera feed instead of video file
    cap = cv2.VideoCapture(0)  # Replace '0' with your camera index if multiple cameras are connected

    sliding_window = []
    frame_skip = 5  # Process every 5th frame to reduce load
    start_time = time.time()
    frame_count = 0

    workspace_folder = '/content/workspace'
    os.makedirs(workspace_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Downsample the frame to reduce computational load
                frame = cv2.resize(frame, (640, 480))

                # Add the frame to sliding window buffer
                sliding_window.append(frame)
                if len(sliding_window) > SLIDING_WINDOW_SIZE:
                    sliding_window.pop(0)  # Maintain sliding window size

                # Process the current sliding window using Instant-NGP
                executor.submit(run_instant_ngp, sliding_window.copy(), workspace_folder)

            frame_count += 1

            # Monitor system and GPU memory usage (optional)
            system_memory = psutil.virtual_memory()
            gpu_memory = torch.cuda.memory_allocated()
            print(f"System Memory Usage: {system_memory.percent}%")
            print(f"GPU Memory Usage: {gpu_memory / (1024 ** 2)} MB")

            torch.cuda.empty_cache()

        cap.release()

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")