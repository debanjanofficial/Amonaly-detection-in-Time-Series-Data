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

# --- LSTM Autoencoder Model Config ---

LSTM_SEQUENCE_LENGTH = 24  # Use 24 hours of data to form one sequence
LSTM_INPUT_DIM = None      # To be set dynamically based on selected features
LSTM_EMBEDDING_DIM = 64    # Latent space dimension (bottleneck)
LSTM_HIDDEN_DIM = 128      # LSTM hidden state dimension
LSTM_N_LAYERS = 2          # Number of LSTM layers
LSTM_DROPOUT_PROB = 0.2    # Dropout probability

# --- Training Config ---
TRAIN_BATCH_SIZE = 64
TRAIN_N_EPOCHS = 100
TRAIN_LEARNING_RATE = 0.001
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_models') # Directory to save models
MODEL_FILENAME = 'lstm_autoencoder.pth'
SCALER_FILENAME = 'scaler.pkl'