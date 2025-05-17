import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Data Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONSUMPTION_DATA_FILENAME = 'Actual_consumption_201604010000_202501010000_Hour.csv'
GENERATION_DATA_FILENAME = 'Actual_generation_201604010000_202504020000_Hour.csv'

CONSUMPTION_DATA_PATH = os.path.join(DATA_DIR, CONSUMPTION_DATA_FILENAME)
GENERATION_DATA_PATH = os.path.join(DATA_DIR, GENERATION_DATA_FILENAME)

# --- Model Config (Placeholders) ---
SEQUENCE_LENGTH = 24  
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'saved_model.pth')

# --- Preprocessing ---
TARGET_CONSUMPTION_COLUMN = 'grid load [MWh] Calculated resolutions'
# Add target generation columns if needed, e.g., sum of renewables, total generation etc.

# --- Anomaly Detection ---
ANOMALY_THRESHOLD = 3.0 