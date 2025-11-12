import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict, Optional

from environment import DATA_DIR, PLOTS_DIR
from tools.data_loading.data_loading_utils import _load_data_from_file
from tools.visualization.visualization_utils import (
    _plot_clusters_and_outliers,
    _plot_clusters_only,
    _plot_outliers_only
)

plt.switch_backend('Agg')

def register_visualization_tools(mcp):
    """Register visualization tools for the given MCP server instance"""
    
    @mcp.tool()
    def plot_dataset(file_path: str, title: Optional[str] = None, 
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a simple scatter plot of a dataset from a file and save the plot to a figure.
        
        IMPORTANT: Always provide ABSOLUTE PATHS for both file_path and save_path.
        Use the get_workspace_paths() tool first to get the correct directory paths.
        
        Args:
            file_path: ABSOLUTE path to CSV file containing the dataset (e.g., /full/path/to/data.csv)
            title: Title for the plot
            save_path: ABSOLUTE path where to save the plot image (e.g., /full/path/to/plot.png)
        
        Returns:
            Dictionary with plot information and saved file location
        """
        try:
            # Validate input file exists
            if not os.path.exists(file_path):
                return {"error": f"Input file not found: {file_path}"}
            
            data_array = _load_data_from_file(file_path)
            
            # Ensure save directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # Auto-generate save path if not provided or invalid
            if save_path is None or save_path == "":
                abs_plots_dir = os.path.abspath(PLOTS_DIR)
                os.makedirs(abs_plots_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_path = os.path.join(abs_plots_dir, f"{base_name}_plot.png")
            
            # Create figure and axis
            plt.figure(figsize=(10, 8))
            
            # Simple scatter plot
            plt.scatter(data_array[:, 0], data_array[:, 1], 
                    alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)
            
            # Styling
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save image
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Clean up
            
            # Calculate plot statistics
            plot_stats = {
                "n_points": len(data_array),
                "plot_type": "simple_scatter",
                "x_range": [float(data_array[:, 0].min()), float(data_array[:, 0].max())],
                "y_range": [float(data_array[:, 1].min()), float(data_array[:, 1].max())]
            }
            
            return {
                "plot_created": True,
                "title": title,
                "source_file": file_path,
                "saved_to": os.path.abspath(save_path),
                "file_size_bytes": os.path.getsize(save_path),
                "statistics": plot_stats,
                "message": f"Plot successfully created and saved to: {os.path.abspath(save_path)}"
            }
            
        except Exception as e:
            return {"error": f"Plot creation failed: {str(e)}"}

    @mcp.tool()
    def plot_clustering_results(results_file_path: str, title: Optional[str] = None,
                            save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a visualization of clustering results from a results file.

        Args:
            results_file_path: Path to the clustering results CSV file
            title: Custom title for the plot
            save_path: Path to save the plot image (auto-generated if not provided)

        Returns:
            Dictionary with plot information and saved file location
        """
        try:
            if not os.path.exists(results_file_path):
                return {"error": f"Results file not found: {results_file_path}"}
            
            # Load clustering results from CSV
            results_df = pd.read_csv(results_file_path)
            
            # Extract metadata from first row (stored as comments or separate columns)
            source_file = results_df.iloc[0]['source_file'] if 'source_file' in results_df.columns else None
            algorithm = results_df.iloc[0]['algorithm'] if 'algorithm' in results_df.columns else "Clustering"
            
            if not source_file:
                return {"error": "Clustering result missing source_file information"}
            
            data_array = _load_data_from_file(source_file)
            cluster_labels = results_df['cluster_label'].tolist()
            
            # Auto-generate save path if not provided
            if save_path is None:
                os.makedirs('plots', exist_ok=True)
                base_name = os.path.splitext(os.path.basename(results_file_path))[0]
                save_path = f"plots/{base_name}_plot.png"
            
            # Create title
            if title is None:
                n_clusters = len(set(cluster_labels))
                title = f"{algorithm} Results - {n_clusters} Clusters"
            
            # Create figure and axis
            plt.figure(figsize=(10, 8))
            
            # Plot clusters
            _plot_clusters_only(data_array, cluster_labels)
            
            # Styling
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save image
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Clean up
            
            # Calculate plot statistics
            plot_stats = {
                "n_points": len(data_array),
                "plot_type": "clusters_only",
                "n_clusters": len(set(cluster_labels)),
                "x_range": [float(data_array[:, 0].min()), float(data_array[:, 0].max())],
                "y_range": [float(data_array[:, 1].min()), float(data_array[:, 1].max())]
            }
            
            return {
                "plot_created": True,
                "title": title,
                "source_file": source_file,
                "saved_to": save_path,
                "statistics": plot_stats
            }
            
        except Exception as e:
            return {"error": f"Clustering plot creation failed: {str(e)}"}

    @mcp.tool()
    def plot_outliers(results_file_path: str, title: Optional[str] = None,
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a visualization of outlier detection results from a results file.

        Args:
            results_file_path: Path to the outlier detection results CSV file
            title: Custom title for the plot
            save_path: Path to save the plot image (auto-generated if not provided)

        Returns:
            Dictionary with plot information and saved file location
        """
        try:
            if not os.path.exists(results_file_path):
                return {"error": f"Results file not found: {results_file_path}"}
            
            # Load outlier results from CSV
            results_df = pd.read_csv(results_file_path)
            
            # Extract metadata
            source_file = results_df.iloc[0]['source_file'] if 'source_file' in results_df.columns else None
            method = results_df.iloc[0]['method'] if 'method' in results_df.columns else "Outlier Detection"
            
            if not source_file:
                return {"error": "Outlier result missing source_file information"}
            
            data_array = _load_data_from_file(source_file)
            outlier_flags = results_df['is_outlier'].astype(bool).tolist()
            
            # Auto-generate save path if not provided
            if save_path is None:
                os.makedirs('plots', exist_ok=True)
                base_name = os.path.splitext(os.path.basename(results_file_path))[0]
                save_path = f"plots/{base_name}_plot.png"
            
            # Create title
            if title is None:
                n_outliers = sum(outlier_flags)
                title = f"{method} - {n_outliers} Outliers Found"
            
            # Create figure and axis
            plt.figure(figsize=(10, 8))
            
            # Plot outliers
            _plot_outliers_only(data_array, outlier_flags)
            
            # Styling
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save image
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Clean up
            
            # Calculate plot statistics
            n_outliers = sum(outlier_flags)
            plot_stats = {
                "n_points": len(data_array),
                "plot_type": "outliers_only",
                "n_outliers": n_outliers,
                "outlier_percentage": (n_outliers / len(data_array)) * 100,
                "x_range": [float(data_array[:, 0].min()), float(data_array[:, 0].max())],
                "y_range": [float(data_array[:, 1].min()), float(data_array[:, 1].max())]
            }
            
            return {
                "plot_created": True,
                "title": title,
                "source_file": source_file,
                "saved_to": save_path,
                "statistics": plot_stats
            }
            
        except Exception as e:
            return {"error": f"Outlier plot creation failed: {str(e)}"}

    @mcp.tool()
    def plot_combined_analysis(clustering_results_file: str, outlier_results_file: str,
                            title: Optional[str] = None, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a combined visualization showing both clustering and outlier results.

        Args:
            clustering_results_file: Path to clustering results CSV file
            outlier_results_file: Path to outlier detection results CSV file
            title: Custom title for the plot (auto-generated if not provided)
            save_path: Path to save the plot image (auto-generated if not provided)

        Returns:
            Dictionary with plot information and saved file location
        """
        try:
            if not os.path.exists(clustering_results_file):
                return {"error": f"Clustering results file not found: {clustering_results_file}"}
            
            if not os.path.exists(outlier_results_file):
                return {"error": f"Outlier results file not found: {outlier_results_file}"}
            
            # Load results from CSV files
            clustering_df = pd.read_csv(clustering_results_file)
            outlier_df = pd.read_csv(outlier_results_file)
            
            # Extract metadata
            clustering_source = clustering_df.iloc[0]['source_file'] if 'source_file' in clustering_df.columns else None
            outlier_source = outlier_df.iloc[0]['source_file'] if 'source_file' in outlier_df.columns else None
            algorithm = clustering_df.iloc[0]['algorithm'] if 'algorithm' in clustering_df.columns else "Clustering"
            
            if clustering_source != outlier_source:
                return {"error": "Clustering and outlier results must be from the same dataset"}
            
            # Load data
            data_array = _load_data_from_file(clustering_source)
            cluster_labels = clustering_df['cluster_label'].tolist()
            outlier_flags = outlier_df['is_outlier'].astype(bool).tolist()
            
            # Auto-generate save path if not provided
            if save_path is None:
                os.makedirs('plots', exist_ok=True)
                base_name = os.path.splitext(os.path.basename(clustering_results_file))[0]
                save_path = f"plots/{base_name}_combined_plot.png"
            
            # Create title
            if title is None:
                n_clusters = len(set(cluster_labels))
                n_outliers = sum(outlier_flags)
                title = f"{algorithm} + Outlier Detection - {n_clusters} Clusters, {n_outliers} Outliers"
            
            # Create figure and axis
            plt.figure(figsize=(12, 8))
            
            # Plot combined results
            _plot_clusters_and_outliers(data_array, cluster_labels, outlier_flags)
            
            # Styling
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save image
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Clean up
            
            # Calculate plot statistics
            n_outliers = sum(outlier_flags)
            plot_stats = {
                "n_points": len(data_array),
                "plot_type": "clusters_and_outliers",
                "n_clusters": len(set(cluster_labels)),
                "n_outliers": n_outliers,
                "outlier_percentage": (n_outliers / len(data_array)) * 100,
                "x_range": [float(data_array[:, 0].min()), float(data_array[:, 0].max())],
                "y_range": [float(data_array[:, 1].min()), float(data_array[:, 1].max())]
            }
            
            return {
                "plot_created": True,
                "title": title,
                "source_file": clustering_source,
                "saved_to": save_path,
                "statistics": plot_stats
            }
            
        except Exception as e:
            return {"error": f"Combined plot creation failed: {str(e)}"}

