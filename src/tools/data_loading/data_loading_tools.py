import os
import shutil
from typing import Any, Dict
from datetime import datetime
import pandas as pd


from environment import DATA_DIR, PLOTS_DIR, DATABASE_TABLES

# Get the project root directory (parent of src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.dirname(current_dir)  # src/tools
src_dir = os.path.dirname(tools_dir)  # src
PROJECT_ROOT = os.path.dirname(src_dir)  # project root

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
                # Check if dataset file exists
                source_file = os.path.join(PROJECT_ROOT, "data", f"{table_name}.csv")
                dataset_exists = os.path.exists(source_file)

                # Get row count from the actual dataset if it exists
                estimated_rows = "Unknown"
                if dataset_exists:
                    try:
                        df = pd.read_csv(source_file)
                        estimated_rows = len(df)
                    except Exception:
                        estimated_rows = "Error reading file"

                tables_info.append(
                    {
                        "table_name": table_name,
                        "description": info["description"],
                        "columns": info["columns"],
                        "estimated_rows": estimated_rows,
                        "cluster_potential": info["cluster_characteristics"],
                        "dataset_available": dataset_exists,
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
    def fetch_table_data(table_name: str) -> Dict[str, Any]:
        """
        Fetch data from a specified database table.
        This copies the pre-generated dataset from the data/ folder to the .data_cache/ folder
        and returns the absolute path where the data was saved.

        Args:
            table_name: Name of the database table to fetch

        Returns:
            Dictionary with query metadata and file information (no raw data)
        """
        try:
            if table_name not in DATABASE_TABLES:
                return {
                    "error": f"Table '{table_name}' not found. See available tables using the `list_database_tables` tool."
                }

            # Check if source dataset exists
            source_file = os.path.join(PROJECT_ROOT, "data", f"{table_name}.csv")
            if not os.path.exists(source_file):
                return {
                    "error": f"Dataset file not found: {source_file}. Please run scripts/generate_datasets.py first."
                }

            table_info = DATABASE_TABLES[table_name]

            # Ensure cache directory exists
            abs_data_dir = os.path.abspath(DATA_DIR)
            os.makedirs(abs_data_dir, exist_ok=True)

            # Create destination path in cache
            file_path = os.path.join(abs_data_dir, f"{table_name}.csv")

            # Copy the dataset from data/ to the target location
            shutil.copy2(source_file, file_path)

            # Read the dataset to get info
            df = pd.read_csv(file_path)

            # Get file size
            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "table_name": table_name,
                "query_executed": f"SELECT * FROM {table_name}",
                "rows_fetched": len(df),
                "file_path": os.path.abspath(file_path),
                "file_size_bytes": file_size,
                "columns": table_info["columns"],
                "description": table_info["description"],
                "cluster_potential": table_info["cluster_characteristics"],
                "message": f"Fetched {len(df)} rows from {table_name} table and saved to {file_path}",
            }
        except Exception as e:
            return {"error": f"Failed to fetch table data: {str(e)}"}

    @mcp.tool()
    def get_table_info(table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific database table.

        This tool provides metadata for a specific table including description,
        columns, and cluster characteristics.

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

            # Get actual row count from the dataset if it exists
            source_file = os.path.join(PROJECT_ROOT, "data", f"{table_name}.csv")
            actual_rows = "Unknown"
            dataset_available = False

            if os.path.exists(source_file):
                dataset_available = True
                try:
                    df = pd.read_csv(source_file)
                    actual_rows = len(df)
                except Exception:
                    actual_rows = "Error reading file"

            return {
                "success": True,
                "table_name": table_name,
                "description": table_info["description"],
                "columns": table_info["columns"],
                "column_count": len(table_info["columns"]),
                "actual_rows": actual_rows,
                "cluster_characteristics": table_info["cluster_characteristics"],
                "data_types": "Numeric",
                "dataset_available": dataset_available,
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
