# Install specific version of timm
!pip install timm==0.4.12

import os
import cv2
import torch
import numpy as np
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import sys
from transformers import AutoProcessor, AutoModelForDepthEstimation
from huggingface_hub import login
from scipy.spatial.transform import Rotation as R
from IPython.display import Image, display, clear_output

# Add the path to the Instant-NGP Python bindings
sys.path.append('/content/instant-ngp/build')

# Import the Instant-NGP Python bindings
import pyngp as ngp

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hardcoded Hugging Face Token (replace with actual token)
HF_TOKEN = "your_hugging_face_token_here"
login(HF_TOKEN)

# Load Depth Anything V2 (apple/DepthPro-hf)
processor = AutoProcessor.from_pretrained("apple/DepthPro-hf")
model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

# Sliding window size for recent frames
SLIDING_WINDOW_SIZE = 10
FRAME_SKIP = 5  # Process every 5th frame

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
    height, width = depth_map.shape
    focal_length = 0.5 * width / np.tan(0.5 * 0.6911112070083618)  # Example focal length calculation

    # Generate 3D points from depth map
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    z = depth_map
    x = (i - width / 2) * z / focal_length
    y = (j - height / 2) * z / focal_length

    points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Generate corresponding 2D points
    points_2d = np.stack((i, j), axis=-1).reshape(-1, 2)

    # Camera intrinsic matrix
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])

    # Use PnP to estimate camera pose
    try:
        _, rvec, tvec = cv2.solvePnP(points_3d.astype(np.float32), points_2d.astype(np.float32), K, None)
        # Convert rotation vector to rotation matrix
        R_mat, _ = cv2.Rodrigues(rvec)

        # Create 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R_mat
        pose[:3, 3] = tvec.squeeze()
    except Exception as e:
        print(f"Error in solvePnP: {e}")
        return np.eye(4).tolist()  # Return identity matrix on error
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
            "file_path": f"./frame_{i:04d}.jpg",  # Relative path
            "depth_file_path": f"./depth_{i:04d}.npy",  # Relative path
            "transform_matrix": pose,
            "w": frame.shape[1],
            "h": frame.shape[0]
        })

    # Write transforms.json file in a controlled way
    transforms_path = os.path.join(data_folder, 'transforms.json')
    try:
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=4)
        print(f"transforms.json created at {transforms_path}")
    except IOError as e:
        print(f"IOError writing transforms.json: {e}")
    except TypeError as e:
        print(f"TypeError writing transforms.json: {e}")
    except Exception as e:
        print(f"Error writing transforms.json: {e}")

# Function to run Instant-NGP for real-time reconstruction
def run_instant_ngp(frames, workspace_folder, ngp_model):
    os.makedirs(workspace_folder, exist_ok=True)

    # Prepare data for Instant-NGP
    prepare_instant_ngp_data(frames, workspace_folder)

    # Load the scene
    try:
        ngp_model.load_training_data(f"{workspace_folder}/data")
        print("Loaded training data successfully.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Train the model
    try:
        ngp_model.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Render the output image
    output_folder = os.path.join(workspace_folder, 'output')
    os.makedirs(output_folder, exist_ok=True)

    try:
        output_image_path = os.path.join(output_folder, 'render.png')
        ngp_model.render_to_file(output_image_path)
        print(f"Render saved to {output_image_path}")

        # Display the output render
        if os.path.exists(output_image_path):
            clear_output(wait=True)
            display(Image(filename=output_image_path))
        else:
            print("Render image not found.")

    except Exception as e:
        print(f"Error during rendering or display: {e}")

# Main Execution
if __name__ == "__main__":
    # Use pre-recorded video file instead of live camera feed
    video_path = '/content/large.mp4'  # Update this path to your video file
    cap = cv2.VideoCapture(video_path)

    sliding_window = []
    frame_skip = FRAME_SKIP  # Process every 5th frame
    start_time = time.time()
    frame_count = 0

    workspace_folder = '/content/workspace'
    os.makedirs(workspace_folder, exist_ok=True)

    # Initialize the NGP model
    ngp_model = ngp.Testbed(ngp.TestbedMode.Nerf)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []  # Store futures

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

                # Debug: Print the number of frames in the sliding window
                print(f"Sliding window size: {len(sliding_window)}")

                # Submit the task to the executor
                future = executor.submit(run_instant_ngp, sliding_window.copy(), workspace_folder, ngp_model)
                futures.append(future)  # Append future to the list

            frame_count += 1

            # Monitor system and GPU memory usage (optional)
            system_memory = psutil.virtual_memory()
            gpu_memory = torch.cuda.memory_allocated()
            print(f"System Memory Usage: {system_memory.percent}%")
            print(f"GPU Memory Usage: {gpu_memory / (1024 ** 2)} MB")
            torch.cuda.empty_cache()

        cap.release()

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # Get the result (or exception if any)
            except Exception as e:
                print(f"Exception in thread: {e}")

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time} seconds")
