from typing import List
import matplotlib.pyplot as plt
import numpy as np


def _plot_clusters_only(data_array: np.ndarray, cluster_labels: List[int]):
    """Helper function to plot clustered data."""
    unique_labels = list(set(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = np.array(cluster_labels) == label
        cluster_data = data_array[mask]

        plt.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add centroid
        centroid = cluster_data.mean(axis=0)
        plt.scatter(centroid[0], centroid[1], c="black", marker="x", s=200, linewidth=3)

    plt.legend()


def _plot_outliers_only(data_array: np.ndarray, outlier_flags: List[bool]):
    """Helper function to plot data with outliers highlighted."""
    outlier_mask = np.array(outlier_flags)
    normal_data = data_array[~outlier_mask]
    outlier_data = data_array[outlier_mask]

    # Plot normal points
    plt.scatter(
        normal_data[:, 0],
        normal_data[:, 1],
        c="blue",
        alpha=0.7,
        s=50,
        label="Normal Points",
        edgecolors="black",
        linewidth=0.5,
    )

    # Plot outliers
    if len(outlier_data) > 0:
        plt.scatter(
            outlier_data[:, 0],
            outlier_data[:, 1],
            c="red",
            marker="X",
            s=100,
            label="Outliers",
            edgecolors="darkred",
            linewidth=1,
        )

    plt.legend()


def _plot_clusters_and_outliers(
    data_array: np.ndarray, cluster_labels: List[int], outlier_flags: List[bool]
):
    """Helper function to plot clustered data with outliers highlighted."""
    outlier_mask = np.array(outlier_flags)

    # Plot clusters for non-outlier points
    normal_mask = ~outlier_mask
    normal_data = data_array[normal_mask]
    normal_labels = [
        cluster_labels[i] for i in range(len(cluster_labels)) if not outlier_mask[i]
    ]

    if len(normal_data) > 0:
        unique_labels = list(set(normal_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = np.array(normal_labels) == label
            cluster_data = normal_data[mask]

            plt.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                c=[colors[i]],
                label=f"Cluster {label}",
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

    # Plot outliers
    outlier_data = data_array[outlier_mask]
    if len(outlier_data) > 0:
        plt.scatter(
            outlier_data[:, 0],
            outlier_data[:, 1],
            c="red",
            marker="X",
            s=100,
            label="Outliers",
            edgecolors="darkred",
            linewidth=1,
        )

    plt.legend()
