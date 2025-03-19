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

# Function to prepare data for Instant-NGP
def prepare_instant_ngp_data(image_folder, workspace_folder):
    # Create the required directory structure
    data_folder = os.path.join(workspace_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Copy images to the data folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            shutil.copy(os.path.join(image_folder, filename), data_folder)
    
    # Generate a dummy transforms.json file (you need to replace this with actual camera poses)
    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": [
            {"file_path": f"./{filename}", "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
            for filename in os.listdir(data_folder) if filename.endswith('.jpg')
        ]
    }
    
    with open(os.path.join(data_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

# Function to run Instant-NGP for 3D reconstruction
def run_instant_ngp(image_folder, workspace_folder):
    os.makedirs(workspace_folder, exist_ok=True)
    
    # Prepare data for Instant-NGP
    prepare_instant_ngp_data(image_folder, workspace_folder)
    
    # Load the scene
    scene = ngp.load_scene(f"{workspace_folder}/data")
    
    # Train the model
    ngp.train(scene)

# Main Execution
if __name__ == "__main__":
    video_path = '/content/large.mp4'  # Update this path to your video file
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_skip = 5  # Process every 5th frame to reduce load

    start_time = time.time()
    frame_count = 0

    image_folder = '/content/datafiles'
    workspace_folder = '/content/workspace'
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(workspace_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Downsample the frame to reduce computational load
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
                frame_path = os.path.join(image_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)

            frame_count += 1

            # Monitor system and GPU memory usage
            system_memory = psutil.virtual_memory()
            gpu_memory = torch.cuda.memory_allocated()
            print(f"System Memory Usage: {system_memory.percent}%")
            print(f"GPU Memory Usage: {gpu_memory / (1024 ** 2)} MB")

            torch.cuda.empty_cache()

    cap.release()

    # Save frames to image folder for Instant-NGP
    for i, frame in enumerate(frames):
        frame_path = os.path.join(image_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)

    # Run Instant-NGP to generate 3D model
    run_instant_ngp(image_folder, workspace_folder)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")