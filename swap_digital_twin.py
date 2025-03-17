# Import required libraries
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
import time
from transformers import DPTImageProcessor, DPTForDepthEstimation
import psutil
import pycolmap
from concurrent.futures import ThreadPoolExecutor

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load advanced depth estimation model
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

# Function to run PyCOLMAP for camera pose and intrinsic estimation
def run_pycolmap(image_folder, workspace_folder):
    # Create database
    database_path = f"{workspace_folder}/database.db"
    
    # Feature extraction
    pycolmap.extract_features(database_path=database_path, image_path=image_folder)
    
    # Exhaustive matching 
    pycolmap.match_exhaustive(database_path=database_path)
    
    # Sparse reconstruction
    sparse_folder = f"{workspace_folder}/sparse"
    os.makedirs(sparse_folder, exist_ok=True)
    
    # Run incremental mapping as a standalone function
    maps = pycolmap.incremental_mapping(database_path=database_path, 
                                        image_path=image_folder, 
                                        output_path=sparse_folder)
    
    # Dense reconstruction
    dense_folder = f"{workspace_folder}/dense"
    os.makedirs(dense_folder, exist_ok=True)
    
    # Use the first reconstruction for undistorting (typically sparse/0)
    sparse_model_path = os.path.join(sparse_folder, "0")
    
    pycolmap.undistort_images(image_path=image_folder, 
                             input_path=sparse_model_path, 
                             output_path=dense_folder)
    
    pycolmap.patch_match_stereo(workspace_path=dense_folder)
    pycolmap.stereo_fusion(workspace_path=dense_folder, 
                          output_path=f"{dense_folder}/fused.ply")
    
    # Export model
    pycolmap.convert_model(input_path=sparse_model_path, 
                          output_path=workspace_folder, 
                          output_type="TXT")
    
# Video to Point Cloud Converter
class VideoToTwinConverter:
    def __init__(self, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_model = depth_model
        self.image_processor = image_processor
        self.batch_size = batch_size

    def _generate_pointcloud(self, rgb, depth):
        h, w = depth.shape
        fx, fy = 0.8*w, 0.8*h  # Simplified focal length

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        z = depth * 100  # Scale factor
        x = (xx - w/2) * z / fx
        y = (yy - h/2) * z / fy

        points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        colors = rgb.reshape(-1, 3)/255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def process_frame(self, frame):
        # Preprocess frame
        inputs = self.image_processor(images=[frame], return_tensors="pt").to(self.device)

        # Depth prediction
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Generate point cloud
        pointcloud = self._generate_pointcloud(frame, depth)
        return pointcloud

# Function to align and merge point clouds
def align_and_merge_pointclouds(pointclouds):
    pcd_combined = pointclouds[0]
    
    for i in range(1, len(pointclouds)):
        # Perform point-to-point ICP registration
        icp_result = o3d.pipelines.registration.registration_icp(
            pointclouds[i], pcd_combined, max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        # Transform the point cloud and merge it
        pointclouds[i].transform(icp_result.transformation)
        pcd_combined += pointclouds[i]
    
    # Downsample the merged point cloud
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.01)
    return pcd_combined

# Main Execution
if __name__ == "__main__":
    # Initialize components
    converter = VideoToTwinConverter(batch_size=32)

    # Open the video file (use 0 for live camera feed)
    video_path = '/home/sk/work/ajinkya/digital_twin/large.mp4'  # Update this path to your video file
    cap = cv2.VideoCapture(video_path)

    frames = []
    image_folder = './datafiles/custom/JPEGImages/640p/custom'
    workspace_folder = './datafiles/custom/triangulation'
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(workspace_folder, exist_ok=True)

    start_time = time.time()
    frame_count = 0
    pointclouds = []
    frame_skip = 5  # Process every 5th frame to reduce load

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
                
                # Process frame asynchronously
                future = executor.submit(converter.process_frame, frame)
                pointclouds.append(future)

            frame_count += 1

            # Monitor system and GPU memory usage
            system_memory = psutil.virtual_memory()
            gpu_memory = torch.cuda.memory_allocated()
            print(f"System Memory Usage: {system_memory.percent}%")
            print(f"GPU Memory Usage: {gpu_memory / (1024 ** 2)} MB")

    cap.release()

    # Wait for all point cloud processing to complete
    pointclouds = [future.result() for future in pointclouds]

    # Run PyCOLMAP to estimate camera pose and intrinsic parameters
    run_pycolmap(image_folder, workspace_folder)

    # Align and merge point clouds
    pcd_combined = align_and_merge_pointclouds(pointclouds)

    # Save the final 3D model
    o3d.io.write_point_cloud("final_3d_model.ply", pcd_combined)

    # Visualize the final 3D model
    o3d.visualization.draw_geometries([pcd_combined])

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")