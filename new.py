import os
import cv2
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import json
import datetime
import argparse

class VideoToDigitalTwinPipeline:
    def __init__(self, video_path, output_dir, model_type="dpt_large"):
        """
        Initialize the pipeline for creating digital twins from video
        
        Args:
            video_path: Path to the input video file
            output_dir: Main output directory for all processing
            model_type: DPT model type ('dpt_large', 'dpt_hybrid', etc.)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.model_type = model_type
        
        # Create directory structure
        self.point_cloud_dir = os.path.join(output_dir, "point_clouds")
        self.mesh_dir = os.path.join(output_dir, "meshes")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        for directory in [self.point_cloud_dir, self.mesh_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create metadata structure
        self.metadata = {
            "creation_date": datetime.datetime.now().isoformat(),
            "source_video": os.path.abspath(video_path),
            "components": [],
            "version": "1.0"
        }
    
    def extract_point_clouds_from_video(self, sample_rate=10):
        """Extract point clouds from video frames using Intel DPT from Hugging Face"""
        print(f"Extracting point clouds from video: {self.video_path}")
        
        # Import Hugging Face transformers
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
        except ImportError:
            print("Hugging Face transformers not found. Installing...")
            import subprocess
            subprocess.call(["pip", "install", "transformers"])
            from transformers import DPTForDepthEstimation, DPTImageProcessor
        
        # Map model_type to Hugging Face model ID
        model_map = {
            "dpt_large": "Intel/dpt-large",
            "dpt_hybrid": "Intel/dpt-hybrid",
            "midas_v21_small": "Intel/dpt-large" # Fallback to large as small isn't directly available
        }
        
        model_id = model_map.get(self.model_type, "Intel/dpt-large")
        
        # Hardcoded Hugging Face token for authentication
        # WARNING: For production code, use environment variables or secure storage instead
        hf_token = "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"
        
        # Load DPT model from Hugging Face with authentication
        print(f"Loading Intel DPT model from Hugging Face ({model_id})...")
        processor = DPTImageProcessor.from_pretrained(model_id, token=hf_token)
        model = DPTForDepthEstimation.from_pretrained(model_id, token=hf_token)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Load video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video FPS: {fps}, Total frames: {frame_count}")
        
        # Process frames
        frame_idx = 0
        point_cloud_paths = []
        
        with tqdm(total=frame_count//sample_rate) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_idx % sample_rate == 0:
                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Prepare image for model input using the processor
                    inputs = processor(images=rgb, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Predict depth
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predicted_depth = outputs.predicted_depth
                        
                        # Interpolate to original image size if needed
                        if predicted_depth.shape[-2:] != (rgb.shape[0], rgb.shape[1]):
                            predicted_depth = torch.nn.functional.interpolate(
                                predicted_depth.unsqueeze(1),
                                size=rgb.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze()
                        
                        # Convert to numpy array
                        depth = predicted_depth.cpu().numpy()
                    
                    # Normalize depth values to a reasonable range (important for point cloud creation)
                    depth_min = depth.min()
                    depth_max = depth.max()
                    depth = (depth - depth_min) / (depth_max - depth_min) * 10.0  # Scale to 0-10 range
                    
                    # Create point cloud
                    point_cloud = self.create_point_cloud_from_depth(rgb, depth)
                    
                    # Save point cloud
                    output_path = os.path.join(self.point_cloud_dir, f"frame_{frame_idx:06d}.ply")
                    o3d.io.write_point_cloud(output_path, point_cloud)
                    point_cloud_paths.append(output_path)
                    pbar.update(1)
                
                frame_idx += 1
        
        cap.release()
        print(f"Extracted {len(point_cloud_paths)} point clouds")
        
        # Combine all point clouds
        combined_pcd = self.combine_point_clouds(point_cloud_paths)
        combined_pcd_path = os.path.join(self.point_cloud_dir, "combined_pointcloud.ply")
        o3d.io.write_point_cloud(combined_pcd_path, combined_pcd)
        
        return combined_pcd
    
    def create_point_cloud_from_depth(self, rgb, depth, fx=None, fy=None, cx=None, cy=None):
        """Convert depth map and RGB image to colored point cloud"""
        height, width = depth.shape
        
        # If no camera intrinsics are provided, use default values
        if fx is None:
            fx = width * 0.8  # Approximate focal length
        if fy is None:
            fy = width * 0.8
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2
        
        # Create meshgrid for pixel coordinates
        y, x = np.meshgrid(range(height), range(width), indexing='ij')
        
        # Calculate 3D coordinates
        z = depth.flatten()
        x = ((x.flatten() - cx) * z / fx)
        y = ((y.flatten() - cy) * z / fy)
        
        # Stack to create point cloud
        points = np.stack((x, y, z), axis=1)
        
        # Get colors from RGB image
        colors = rgb.reshape(-1, 3) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def combine_point_clouds(self, point_cloud_paths, downsample_voxel_size=0.01):
        """Combine multiple point cloud files into a single point cloud"""
        print("Combining point clouds...")
        combined = o3d.geometry.PointCloud()
        
        for path in tqdm(point_cloud_paths):
            pc = o3d.io.read_point_cloud(path)
            combined += pc
        
        # Downsample the combined point cloud if it's too large
        if downsample_voxel_size > 0:
            print(f"Downsampling combined point cloud (voxel size: {downsample_voxel_size})...")
            combined = combined.voxel_down_sample(voxel_size=downsample_voxel_size)
        
        # Remove outliers
        print("Removing outliers...")
        cl, ind = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined = combined.select_by_index(ind)
        
        # Estimate normals for better mesh reconstruction
        print("Estimating normals...")
        combined.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        combined.orient_normals_consistent_tangent_plane(100)
        
        return combined
    
    def create_mesh(self, pcd, method="poisson"):
        """
        Create a mesh from the point cloud
        
        Args:
            pcd: Point cloud to convert to mesh
            method: Mesh reconstruction method ('poisson', 'ball_pivoting', or 'alpha_shape')
        
        Returns:
            The reconstructed mesh
        """
        print(f"Creating mesh using {method} reconstruction...")
        
        if method == "poisson":
            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, scale=1.1, linear_fit=False)
            
            # Filter low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == "ball_pivoting":
            # Ball pivoting algorithm
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            
        elif method == "alpha_shape":
            # Alpha shape reconstruction
            alpha = 0.03
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
        else:
            raise ValueError(f"Unknown mesh reconstruction method: {method}")
        
        # Clean up the mesh
        print("Cleaning and optimizing mesh...")
        mesh.compute_vertex_normals()
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    
    def texture_mesh(self, mesh, pcd):
        """Apply colors from point cloud to mesh"""
        print("Texturing mesh...")
        
        # Create a KD Tree from point cloud points
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        # For each vertex in the mesh, find the nearest point in the point cloud
        vertices = np.asarray(mesh.vertices)
        vertex_colors = []
        
        for vertex in tqdm(vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
            nearest_point_color = np.asarray(pcd.colors)[idx[0]]
            vertex_colors.append(nearest_point_color)
        
        # Apply colors to mesh vertices
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
        
        return mesh
    
    def segment_model(self, mesh):
        """
        Segment the model into meaningful components
        This is a simplified implementation - real segmentation would be more complex
        """
        print("Segmenting model into components...")
        
        # For this example, we'll just use connected components
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_area = np.asarray(cluster_area)
        
        # Remove small clusters
        min_area = 0.01 * sum(cluster_area)
        triangles_to_remove = cluster_area < min_area
        remove_triangles = np.zeros(len(triangle_clusters), dtype=bool)
        
        for i in range(len(triangle_clusters)):
            if triangles_to_remove[triangle_clusters[i]]:
                remove_triangles[i] = True
        
        mesh.remove_triangles_by_mask(remove_triangles)
        
        # Get components
        components = []
        for i in range(len(cluster_n_triangles)):
            if cluster_area[i] >= min_area:
                component_mesh = o3d.geometry.TriangleMesh()
                component_triangles = triangle_clusters == i
                component_mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(mesh.triangles)[component_triangles])
                component_mesh.vertices = mesh.vertices
                if len(mesh.vertex_colors) > 0:
                    component_mesh.vertex_colors = mesh.vertex_colors
                
                components.append({
                    "id": f"component_{i}",
                    "mesh": component_mesh,
                    "area": float(cluster_area[i])
                })
        
        print(f"Model segmented into {len(components)} components")
        return components
    
    def create_digital_twin(self, reconstruction_method="poisson"):
        """Create the complete digital twin from video"""
        print(f"Creating digital twin from video: {self.video_path}")
        
        # Step 1: Extract point clouds from video
        pcd = self.extract_point_clouds_from_video()
        
        # Step 2: Create and texture mesh
        mesh = self.create_mesh(pcd, method=reconstruction_method)
        textured_mesh = self.texture_mesh(mesh, pcd)
        
        # Step 3: Save complete model
        model_path = os.path.join(self.mesh_dir, "complete_model.obj")
        o3d.io.write_triangle_mesh(model_path, textured_mesh)
        
        # Step 4: Segment into components
        components = self.segment_model(textured_mesh)
        
        # Step 5: Save components
        for component in components:
            component_path = os.path.join(self.mesh_dir, f"{component['id']}.obj")
            o3d.io.write_triangle_mesh(component_path, component["mesh"])
            
            # Add to metadata
            self.metadata["components"].append({
                "id": component["id"],
                "path": component_path,
                "area": component["area"]
            })
        
        # Step 6: Save metadata
        with open(os.path.join(self.metadata_dir, "digital_twin_metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Step 7: Also save as PLY format
        self.export_digital_twin_as_ply()
        
        print(f"Digital twin created successfully in {self.output_dir}")
        return self.metadata

    def export_digital_twin_as_ply(self, output_path=None):
        """Export the digital twin mesh as a PLY file with vertex colors"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "digital_twin.ply")
        
        print(f"Exporting digital twin as PLY file: {output_path}")
        
        # Load the complete model (which is stored as OBJ)
        model_path = os.path.join(self.mesh_dir, "complete_model.obj")
        mesh = o3d.io.read_triangle_mesh(model_path)
        
        # Make sure we have vertex normals
        if not mesh.has_vertex_normals():
            print("Computing vertex normals...")
            mesh.compute_vertex_normals()
        
        # If there are no vertex colors, add a default color
        if not mesh.has_vertex_colors():
            print("No vertex colors found, adding default colors...")
            vertices = np.asarray(mesh.vertices)
            vertex_colors = np.ones((len(vertices), 3)) * 0.7  # Light gray
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        # Save as PLY
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Digital twin exported as PLY: {output_path}")
        
        return output_path

    def create_web_viewer(self):
        """Create a simple web viewer for the digital twin"""
        print("Creating web viewer...")
        
        # Create a simple HTML viewer
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Digital Twin Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.137.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.137.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.137.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.137.0/examples/js/loaders/OBJLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.137.0/examples/js/loaders/MTLLoader.js"></script>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; }
        #canvas { width: 100%; height: 100vh; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            font-family: Arial, sans-serif;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="canvas"></div>
    <div id="info">Digital Twin Viewer<br>Use mouse to rotate, scroll to zoom</div>
    <script>
        // Basic Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x222222);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('canvas').appendChild(renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0x606060);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight2.position.set(-1, -1, -1);
        scene.add(directionalLight2);
        
        // Add controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        // Load the digital twin model
        const objLoader = new THREE.OBJLoader();
        objLoader.load('meshes/complete_model.obj', function(object) {
            // Add material if model doesn't have one
            object.traverse(function(child) {
                if (child instanceof THREE.Mesh) {
                    if (!child.material) {
                        child.material = new THREE.MeshPhongMaterial({
                            color: 0xaaaaaa,
                            wireframe: false
                        });
                    }
                }
            });
            
            scene.add(object);
            
            // Center the model
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            object.position.sub(center);
            
            // Adjust camera
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            camera.position.z = maxDim * 2;
        });
        
        // Handle window resize
        window.addEventListener('resize', function() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""
        
        with open(os.path.join(self.output_dir, "viewer.html"), "w") as f:
            f.write(html_content)
        
        print(f"Web viewer created at {os.path.join(self.output_dir, 'viewer.html')}")

    def visualize_digital_twin(self):
        """Visualize the digital twin using Open3D"""
        print("Visualizing digital twin...")
        
        # Load the complete model
        model_path = os.path.join(self.mesh_dir, "complete_model.obj")
        mesh = o3d.io.read_triangle_mesh(model_path)
        mesh.compute_vertex_normals()
        
        # Create a simple coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        
        # Visualize
        o3d.visualization.draw_geometries([mesh, coordinate_frame])

def main():
    
    # Create the digital twin pipeline
    pipeline = VideoToDigitalTwinPipeline("/content/large.mp4", '.', "dpt_large")
    
    # Run the pipeline
    pipeline.create_digital_twin("poisson")
    
    # Create web viewer if requested
    pipeline.create_web_viewer()
    

    pipeline.visualize_digital_twin()

if __name__ == "__main__":
    main()