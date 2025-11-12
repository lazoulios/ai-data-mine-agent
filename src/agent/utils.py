import os
from pathlib import Path
from environment import DATA_DIR, PLOTS_DIR


current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent
abs_data_dir = os.path.abspath(os.path.join(parent_dir, DATA_DIR))
abs_plots_dir = os.path.abspath(os.path.join(parent_dir, PLOTS_DIR))
