
import os

# Project configuration

PROJECT_NAME = "MLP Project"

# Random seed
RANDOM_SEED = 42

# Dataset configuration
DATA_PATH = "data/dataset.csv"
TARGET_COLUMN = "value"

# Data preprocessing
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model configuration
INPUT_SIZE = 10
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Device configuration
DEVICE = "cpu"

# Output directories
MODEL_DIR = "saved_models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

