import numpy as np
import pandas as pd


def _load_data_from_file(file_path: str) -> np.ndarray:
    """Helper function to load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df[["x", "y"]].values


def _generate_synthetic_data(dataset_name: str, n_samples: int = 300) -> np.ndarray:
    """Generate synthetic 2D data with well-separated, dense clusters and sparse outliers."""
    # Use dataset name as seed for reproducibility with variation
    seed = hash(dataset_name) % 2**32
    np.random.seed(seed)
    
    # Determine a random number of clusters in range [3, 8)
    n_clusters = np.random.randint(3, 8)
    
    # Create well-separated cluster centers
    # Use a grid-like approach to ensure good separation
    if n_clusters <= 4:
        # For 3-4 clusters, use corners of a rectangle
        base_positions = [
            [-8, -8.5], [8, -3.5], [-6, 6.5], [12, 2.5]
        ][:n_clusters]
    else:
        # For 5-7 clusters, use a more spread out pattern
        base_positions = [
            [-12, -8], [0, -4], [11.5, -9],
            [-8.5, 3], [16, 2],
            [7.5, 11.5], [-12.5, 12.5]
        ][:n_clusters]
    
    # Add some randomness to positions while maintaining separation
    cluster_centers = []
    for pos in base_positions:
        offset = np.random.normal(0, 0.2, 2)
        cluster_centers.append([pos[0] + offset[0], pos[1] + offset[1]])
    
    cluster_centers = np.array(cluster_centers)
    
    # Reserve 0%-1% for outliers
    outlier_ratio = np.random.uniform(0.0, 0.01)
    n_outliers = int(n_samples * outlier_ratio)
    n_cluster_points = n_samples - n_outliers
    
    data_points = []
    
    # Generate dense, circular clusters
    points_per_cluster = n_cluster_points // n_clusters
    remaining_points = n_cluster_points % n_clusters
    
    for i, center in enumerate(cluster_centers):
        # Distribute remaining points to first few clusters
        cluster_size = points_per_cluster + (1 if i < remaining_points else 0)
        
        # Make clusters dense and circular with consistent size
        cluster_std = np.random.uniform(0.25, 0.4)  # Much smaller std for dense clusters
        
        # Generate points in a more circular pattern
        angles = np.random.uniform(0, 2*np.pi, cluster_size)
        radii = np.random.normal(0, cluster_std, cluster_size)
        radii = np.abs(radii)  # Ensure positive radii
        
        # Convert to cartesian coordinates
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        cluster_points = np.column_stack([x, y])
        
        data_points.append(cluster_points)
    
    # Generate sparse outliers in the spaces between clusters
    if n_outliers > 0:
        outliers = []
        max_attempts = n_outliers * 10  # Prevent infinite loops, try at most 5 times per outlier-to-add
        attempts = 0
        
        while len(outliers) < n_outliers and attempts < max_attempts:
            # Generate candidate outlier point
            candidate = np.random.uniform(-15, 15, 2)
            
            # Check if it's far enough from all cluster centers
            min_distance = np.min([np.linalg.norm(candidate - center) for center in cluster_centers])
            
            # Only accept if it's far enough from clusters (creates sparse outliers)
            if min_distance > 5: 
                outliers.append(candidate)
            
            attempts += 1
        
        if outliers:
            data_points.append(np.array(outliers))
        print(f"{len(outliers)} outliers in total")
    
    # Combine all data
    data = np.vstack(data_points)
    
    # Ensure exact sample count (trim if necessary)
    if len(data) > n_samples:
        data = data[:n_samples]
    
    # Shuffle the data
    np.random.shuffle(data)
    
    return data
