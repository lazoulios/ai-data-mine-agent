import os
from typing import Any, Dict
from datetime import datetime
import pandas as pd


from environment import DATA_DIR, PLOTS_DIR, DATABASE_TABLES
from tools.data_loading.data_loading_utils import _generate_synthetic_data


def register_data_loading_tools(mcp):
    """Register data loading tools for the given MCP server instance"""

    @mcp.tool()
    def get_workspace_paths() -> Dict[str, Any]:
        """
        Get the absolute paths for data and plots directories, ensuring they exist.
        Use this tool when you need to know where to save files or access existing files.

        Returns:
            Dictionary with absolute paths for data and plots directories
        """
        try:
            # Get current working directory
            current_dir = os.getcwd()

            # Create absolute paths
            abs_data_dir = os.path.abspath(DATA_DIR)
            abs_plots_dir = os.path.abspath(PLOTS_DIR)

            # Ensure directories exist
            os.makedirs(abs_data_dir, exist_ok=True)
            os.makedirs(abs_plots_dir, exist_ok=True)

            return {
                "success": True,
                "current_directory": current_dir,
                "data_directory": abs_data_dir,
                "plots_directory": abs_plots_dir,
                "data_dir_exists": os.path.exists(abs_data_dir),
                "plots_dir_exists": os.path.exists(abs_plots_dir),
                "usage_examples": {
                    "data_file": f"{abs_data_dir}/my_dataset.csv",
                    "plot_file": f"{abs_plots_dir}/my_analysis.png",
                },
            }
        except Exception as e:
            return {"error": f"Failed to get workspace paths: {str(e)}"}

    @mcp.tool()
    def list_database_tables() -> Dict[str, Any]:
        """
        List all available tables in the database. Good to use when the user asks for some
        data-related task on a dataset that has never been mentioned in the chat before. The
        tool connects to the user's database.

        Returns:
            Dictionary with available tables and their descriptions
        """
        try:
            tables_info = []

            for table_name, info in DATABASE_TABLES.items():
                tables_info.append(
                    {
                        "table_name": table_name,
                        "description": info["description"],
                        "columns": info["columns"],
                        "estimated_rows": f"{info['sample_size_range'][0]}-{info['sample_size_range'][1]}",
                        "cluster_potential": info["cluster_characteristics"],
                    }
                )

            return {
                "success": True,
                "database_name": "analytics_db",
                "total_tables": len(tables_info),
                "tables": tables_info,
                "connection_status": "Connected",
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": f"Failed to list database tables: {str(e)}"}

    @mcp.tool()
    def fetch_table_data(table_name: str, limit: int, file_path: str) -> Dict[str, Any]:
        """
        Fetch data from a specified database table and save to a csv file.

        IMPORTANT: Provide an ABSOLUTE path for file_path parameter.
        Use the get_workspace_paths() tool first to get the correct data directory path.

        Args:
            table_name: Name of the database table to fetch
            limit: The number of rows to fetch
            file_path: ABSOLUTE path where to save the data in CSV format (e.g., /full/path/to/data.csv)

        Returns:
            Dictionary with query metadata and file information (no raw data)
        """
        try:
            if table_name not in DATABASE_TABLES:
                return {
                    "error": f"Table '{table_name}' not found. See available tables using the `list_database_tables` tool."
                }

            table_info = DATABASE_TABLES[table_name]

            # Execute database query
            min_size, max_size = table_info["sample_size_range"]
            n_samples = min(limit, max_size)

            # Fetch data from database table
            data = _generate_synthetic_data(table_name, n_samples)

            # Create DataFrame with metadata
            df = pd.DataFrame(data, columns=["x", "y"])

            # Ensure data directory exists
            abs_data_dir = os.path.abspath(DATA_DIR)
            os.makedirs(abs_data_dir, exist_ok=True)

            # Handle file path
            if file_path is None or file_path == "":
                file_path = os.path.join(abs_data_dir, f"{table_name}_data.csv")
            else:
                # Ensure directory exists for the provided path
                file_dir = os.path.dirname(file_path)
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)

                # Ensure .csv extension
                if not file_path.endswith(".csv"):
                    file_path += ".csv"

            # Save to CSV file
            df.to_csv(file_path, index=False)

            # Save metadata separately as a comment file
            metadata_file = file_path.replace(".csv", "_metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Table: {table_name}\n")
                f.write(f"Description: {table_info['description']}\n")
                f.write(f"Columns: {', '.join(table_info['columns'])}\n")
                f.write(f"Fetched at: {datetime.now().isoformat()}\n")
                f.write(f"Query limit: {limit}\n")
                f.write(
                    f"Cluster characteristics: {table_info['cluster_characteristics']}\n"
                )
                f.write(f"Rows: {len(data)}\n")
                f.write(f"X range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]\n")
                f.write(f"Y range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]\n")

            # Get file size
            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "table_name": table_name,
                "query_executed": f"SELECT * FROM {table_name}"
                + (f" LIMIT {limit}" if limit else ""),
                "rows_fetched": len(data),
                "file_path": os.path.abspath(file_path),
                "file_size_bytes": file_size,
                "columns": table_info["columns"],
                "description": table_info["description"],
                "cluster_potential": table_info["cluster_characteristics"],
                "message": f"Fetched {len(data)} rows from {table_name} table and saved to {file_path}",
            }
        except Exception as e:
            return {"error": f"Failed to fetch table data: {str(e)}"}

    @mcp.tool()
    def get_table_info(table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific database table.

        This tool simulates querying database metadata for a specific table.
        In production, this would query information_schema or similar system tables.

        Args:
            table_name: Name of the database table to inspect

        Returns:
            Dictionary with detailed table information
        """
        try:
            if table_name not in DATABASE_TABLES:
                available_tables = list(DATABASE_TABLES.keys())
                return {
                    "error": f"Table '{table_name}' not found. Available tables: {available_tables}"
                }

            table_info = DATABASE_TABLES[table_name]

            return {
                "success": True,
                "table_name": table_name,
                "description": table_info["description"],
                "columns": table_info["columns"],
                "column_count": len(table_info["columns"]),
                "estimated_row_range": table_info["sample_size_range"],
                "cluster_characteristics": table_info["cluster_characteristics"],
                "data_types": "Numeric",
            }
        except Exception as e:
            return {"error": f"Failed to get table info: {str(e)}"}

    @mcp.tool()
    def list_csv_files_in(directory: str) -> Dict[str, Any]:
        """
        List all CSV files in a directory.

        Args:
            directory: Directory to search for CSV files
        Returns:
            Dictionary with list of CSV files and their metadata
        """
        try:
            if not os.path.exists(directory):
                return {"error": f"Directory not found: {directory}"}
            
            dataset_files = []
            
            for filename in os.listdir(directory):
                if filename.endswith('.csv') and not filename.endswith('_results.csv'):
                    file_path = os.path.join(directory, filename)
                    try:
                        # Load CSV to validate it's a dataset
                        df = pd.read_csv(file_path)
                        
                        # Check if it's a valid dataset
                        if 'x' in df.columns and 'y' in df.columns:
                            file_size = os.path.getsize(file_path)
                            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                            
                            # Try to get dataset name from metadata file
                            metadata_file = file_path.replace('.csv', '_metadata.txt')
                            dataset_name = "Unknown"
                            if os.path.exists(metadata_file):
                                with open(metadata_file, 'r') as f:
                                    for line in f:
                                        if line.startswith('Dataset:') or line.startswith('Table:'):
                                            dataset_name = line.split(':', 1)[1].strip()
                                            break
                            
                            dataset_files.append({
                                "filename": filename,
                                "file_path": os.path.abspath(file_path),
                                "file_size_bytes": file_size,
                                "file_modified": file_modified,
                                "n_samples": len(df)
                            })
                    except Exception:
                        # Skip invalid CSV or non-dataset files
                        continue
            
            return {
                "success": True,
                "directory": os.path.abspath(directory),
                "csv_files_found": len(dataset_files),
                "files": sorted(dataset_files, key=lambda x: x["filename"])
            }
        except Exception as e:
            return {"error": f"Failed to list CSV files: {str(e)}"}
