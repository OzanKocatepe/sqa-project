from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PLOTTING_DIR = PROJECT_ROOT / "plots"
DATA_DIR = PROJECT_ROOT / "data"
STYLESHEET = PROJECT_ROOT.parent / "stylesheet.mplstyle"