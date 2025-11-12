from .clustering.clustering_tools import register_clustering_tools
from .data_loading.data_loading_tools import register_data_loading_tools
from .math.math_tools import register_math_tools
from .visualization.visualization_tools import register_visualization_tools


def register_all_tools(mcp):
    register_clustering_tools(mcp)
    register_data_loading_tools(mcp)
    register_math_tools(mcp)
    register_visualization_tools(mcp)
