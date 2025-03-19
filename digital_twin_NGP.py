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
from transformers import AutoProcessor, AutoModelForDepthEstimation
from huggingface_hub import login

# Add the path to the Instant-NGP Python bindings
sys.path.append('/content/instant-ngp/build')

# Import the Instant-NGP Python bindings
import pyngp as ngp

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hardcoded Hugging Face Token (replace with your actual token)
HF_TOKEN = "your_hugging_face_token_here"
login(HF_TOKEN)

# Load Depth Anything V2 (apple/DepthPro-hf)
processor = AutoProcessor.from_pretrained("apple/DepthPro-hf")
model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf")

# Sliding window size for recent frames
SLIDING_WINDOW_SIZE = 10

# Function to generate depth maps
def generate_depth_map(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    depth_map = predicted_depth.squeeze().cpu().numpy()
    return depth_map

# Function to generate camera pose from depth map
def generate_camera_pose_from_depth(depth_map):
    # Placeholder for actual depth map processing to generate camera pose
    # This is a simplified example, replace with actual pose estimation logic
    height, width = depth_map.shape
    focal_length = 0.5 * width / np.tan(0.5 * 0.6911112070083618)  # Example focal length calculation

    # Example camera pose (identity matrix)
    pose = np.eye(4)
    return pose.tolist()

# Function to prepare data for Instant-NGP using depth maps
def prepare_instant_ngp_data(frames, workspace_folder):
    # Create the required directory structure
    data_folder = os.path.join(workspace_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)

    # Save frames and generate depth maps
    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": []
    }

    for i, frame in enumerate(frames):
        frame_path = os.path.join(data_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Generate depth map
        depth_map = generate_depth_map(frame)
        depth_path = os.path.join(data_folder, f"depth_{i:04d}.npy")
        np.save(depth_path, depth_map)

        # Generate camera pose from depth map
        pose = generate_camera_pose_from_depth(depth_map)

        transforms["frames"].append({
            "file_path": frame_path,
            "depth_file_path": depth_path,
            "transform_matrix": pose,
            "w": frame.shape[1],
            "h": frame.shape[0]
        })

    with open(os.path.join(data_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

# Function to run Instant-NGP for real-time reconstruction
def run_instant_ngp(frames, workspace_folder):
    os.makedirs(workspace_folder, exist_ok=True)

    # Prepare data for Instant-NGP
    prepare_instant_ngp_data(frames, workspace_folder)

    # Initialize the NGP model
    ngp_model = ngp.Testbed(ngp.TestbedMode.Nerf)

    # Load the scene
    ngp_model.load_training_data(f"{workspace_folder}/data")

    # Train the model
    ngp_model.train()

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