import numpy as np
import pandas as pd
import os
from typing import Dict, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from tools.data_loading.data_loading_utils import _load_data_from_file


def register_clustering_tools(mcp):
    """Register clustering tools for the given MCP server instance"""

    @mcp.tool()
    def kmeans_clustering(file_path: str, n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
        """
        Perform K-means clustering on data from a file.

        K-means is a centroid-based clustering algorithm that partitions data into k clusters.

        PROS:
        - Computationally efficient: O(n*k*i) where n=samples, k=clusters, i=iterations
        - Simple and well-understood algorithm
        - Works well with spherical, well-separated clusters
        - Guaranteed convergence
        - Good performance on large datasets
        - Produces compact, balanced clusters

        CONS:
        - Requires pre-specifying number of clusters (k)
        - Sensitive to the random initialization of cluster centers
        - Assumes spherical clusters of similar size
        - Sensitive to outliers (outliers can skew centroids)
        - Struggles with non-convex cluster shapes
        - Poor performance on clusters with different densities

        BEST USE CASES:
        - When you know the approximate number of clusters
        - Data with spherical, well-separated clusters
        - Large datasets where efficiency is important
        - When clusters are of similar size and density
        - As a preprocessing step for other algorithms using a k number larger than the actual

        AVOID WHEN:
        - Clusters have very different sizes or densities
        - Data contains many outliers
        - Clusters have non-spherical shapes
        - Number of clusters is completely unknown

        Args:
            file_path: Path to CSV file containing the dataset
            n_clusters: Number of clusters to find
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with cluster labels, centroids, and algorithm metadata
        """
        try:
            data_array = _load_data_from_file(file_path)
            
            # Standardize the data for better clustering
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            
            # Transform centroids back to original scale
            centroids_scaled = kmeans.cluster_centers_
            centroids = scaler.inverse_transform(centroids_scaled)
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # Calculate cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "centroid": centroids[label].tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Save results to CSV file to prevent context overflow
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_kmeans_results.csv"
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'K-means'
            })
            
            results_df.to_csv(results_file, index=False)
            
            # Save metadata separately
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Algorithm: K-means\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Inertia: {inertia:.4f}\n")
                f.write(f"Algorithm complexity: O(n*k*i)\n")
                f.write(f"Efficiency: High\n")
                f.write(f"Scalability: Excellent\n")
                f.write(f"Outlier sensitivity: High\n")
                f.write(f"Cluster sizes: {[cluster_stats[i]['size'] for i in range(n_clusters)]}\n")
                for i in range(n_clusters):
                    f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, centroid={cluster_stats[i]['centroid']}\n")
            
            return {
                "success": True,
                "algorithm": "K-means",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": n_clusters,
                "inertia": float(inertia),
                "file_size_bytes": os.path.getsize(results_file),
                "message": f"K-means clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": n_clusters,
                    "cluster_sizes": [cluster_stats[i]["size"] for i in range(n_clusters)],
                    "efficiency": "High"
                }
            }
        except Exception as e:
            return {"error": f"K-means clustering failed: {str(e)}"}

    @mcp.tool()
    def dbscan_clustering(file_path: str, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """
        Scan a database (db) that is full of clustering information. This is really poor documentation.
        No arguments at all.
        """
        try:
            data_array = _load_data_from_file(file_path)
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_scaled)
            
            # Identify outliers (label = -1)
            outlier_mask = labels == -1
            outliers = data_array[outlier_mask]
            
            # Get unique cluster labels (excluding -1 for noise)
            unique_labels = np.unique(labels[labels != -1])
            n_clusters = len(unique_labels)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for label in unique_labels:
                cluster_points = data_array[labels == label]
                cluster_stats[int(label)] = {
                    "size": len(cluster_points),
                    "center": cluster_points.mean(axis=0).tolist(),
                    "std_x": float(cluster_points[:, 0].std()),
                    "std_y": float(cluster_points[:, 1].std())
                }
            
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Save results to CSV file to prevent context overflow
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results_file = f"reports/{base_name}_dbscan_results.csv"
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'point_id': range(len(labels)),
                'cluster_label': labels,
                'x': data_array[:, 0],
                'y': data_array[:, 1],
                'source_file': file_path,
                'algorithm': 'DBSCAN'
            })
            
            results_df.to_csv(results_file, index=False)
            
            # Save metadata separately
            metadata_file = results_file.replace('.csv', '_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Algorithm: DBSCAN\n")
                f.write(f"Source file: {file_path}\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Number of outliers: {np.sum(outlier_mask)}\n")
                f.write(f"Parameters: eps={eps}, min_samples={min_samples}\n")
                f.write(f"Algorithm complexity: O(n log n)\n")
                f.write(f"Efficiency: Medium\n")
                f.write(f"Scalability: Good\n")
                f.write(f"Outlier sensitivity: Low (robust)\n")
                if n_clusters > 0:
                    cluster_sizes = [cluster_stats[i]['size'] for i in range(n_clusters)]
                    f.write(f"Cluster sizes: {cluster_sizes}\n")
                    for i in range(n_clusters):
                        f.write(f"Cluster {i}: size={cluster_stats[i]['size']}, center={cluster_stats[i]['center']}\n")
            
            return {
                "success": True,
                "algorithm": "DBSCAN",
                "source_file": file_path,
                "results_file": os.path.abspath(results_file),
                "n_clusters": int(n_clusters),
                "n_outliers": int(np.sum(outlier_mask)),
                "file_size_bytes": os.path.getsize(results_file),
                "parameters": {"eps": eps, "min_samples": min_samples},
                "message": f"DBSCAN clustering completed. Results saved to {results_file}",
                "summary": {
                    "clusters_found": int(n_clusters),
                    "outliers_detected": int(np.sum(outlier_mask)),
                    "efficiency": "Medium"
                }
            }
        except Exception as e:
            return {"error": f"DBSCAN clustering failed: {str(e)}"}
