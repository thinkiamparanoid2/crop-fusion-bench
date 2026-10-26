from pathlib import Path

# ==============================================================================
# Thesis Repository Directory Configuration
# ==============================================================================

# Experiment and project roots
REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent

# Key Data Directories
DATA_ROOT = PROJECT_ROOT / "Data"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
VAL_DIR = DATA_ROOT / "val"
SPLIT_DIR = DATA_ROOT

# Metadata CSV Files
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
VAL_CSV = DATA_ROOT / "val.csv"
SPLIT_TRAIN_CSV = SPLIT_DIR / "train.csv"
SPLIT_VAL_CSV = SPLIT_DIR / "val.csv"

# Training configuration
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
TRAIN_IMAGE_SUFFIX = ''

# Project Directories
SPLITTER_DIR = REPO_ROOT / "legacy" / "Splitter"
LEARNING_DIR = REPO_ROOT / "learning"
MODEL_DIR = REPO_ROOT / "model"

# Configuration Directories
VSCODE_DIR = REPO_ROOT / ".vscode"

# ==============================================================================
# Quick Path Verification when run directly
# ==============================================================================
if __name__ == "__main__":
    print("-" * 60)
    print("Thesis Repository Directory Configuration")
    print("-" * 60)
    print(f"REPO_ROOT:      {REPO_ROOT}")
    print(f"PROJECT_ROOT:   {PROJECT_ROOT}")
    print(f"DATA_ROOT:      {DATA_ROOT}")
    print(f"TRAIN_DIR:      {TRAIN_DIR}")
    print(f"TEST_DIR:       {TEST_DIR}")
    print(f"VAL_DIR:        {VAL_DIR}")
    print(f"SPLIT_DIR:      {SPLIT_DIR}")
    print(f"TRAIN_CSV:      {TRAIN_CSV}")
    print(f"TEST_CSV:       {TEST_CSV}")
    print(f"VAL_CSV:        {VAL_CSV}")
    print(f"SPLIT_TRAIN_CSV:{SPLIT_TRAIN_CSV}")
    print(f"SPLIT_VAL_CSV:  {SPLIT_VAL_CSV}")
    print(f"SPLITTER_DIR:   {SPLITTER_DIR}")
    print(f"VSCODE_DIR:     {VSCODE_DIR}")
    print("-" * 60)
