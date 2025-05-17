import pandas as pd
import numpy as np # Ensure numpy is imported
import logging
import torch
import torch.nn as nn # For loss function
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler # For scaling
import joblib # For saving/loading scaler

# Import project modules
import config
from src import data_loader, preprocessing, utils
from src.models.deep_learning_models import LSTMAutoencoder # Import your model


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model_pytorch(model, data_loader, n_epochs, learning_rate, device, model_save_path):
    """Trains the PyTorch model."""
    model.train() # Set model to training mode
    criterion = nn.MSELoss(reduction='mean') # Use mean squared error, averaged over elements
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logging.info(f"Starting training for {n_epochs} epochs on device: {device}")
    
    best_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i, seq_true_batch_tuple in enumerate(data_loader): # DataLoader yields tuples
            seq_true = seq_true_batch_tuple[0].to(device) # Get the tensor from the tuple

            optimizer.zero_grad()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * seq_true.size(0) # loss.item() is avg loss for batch
                                                         # Multiply by batch size for total batch loss
        
        avg_epoch_loss = epoch_loss / len(data_loader.dataset) # Average loss per sequence in epoch
        
        logging.info(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.6f}')

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model improved and saved to {model_save_path}")
            
    logging.info("Training complete.")
    # Load best model state
    model.load_state_dict(torch.load(model_save_path))
    return model

def run_pipeline():
    """Runs the anomaly detection pipeline."""

    # 0. Setup Environment
    logging.info("--- Starting Anomaly Detection Pipeline ---")
    device = utils.get_device() # Check for MPS early on

    # 1. Load Data
    logging.info("--- Step 1: Loading Data ---")
    date_cols_to_parse = ['Start date', 'End date']
    consumption_df = data_loader.load_data(
        config.CONSUMPTION_DATA_PATH,
        delimiter=';',
        date_cols=date_cols_to_parse # Pass date_cols for read_csv's parse_dates
    )
    generation_df = data_loader.load_data(
        config.GENERATION_DATA_PATH,
        delimiter=';',
        date_cols=date_cols_to_parse
    )

    if consumption_df.empty or generation_df.empty:
        logging.error("Failed to load one or both datasets. Exiting.")
        return

    # Initial Cleanup
    consumption_df = data_loader.initial_column_cleanup(consumption_df)
    generation_df = data_loader.initial_column_cleanup(generation_df)

    # --- Choose Data for Analysis ---
    # For now, let's focus on consumption data's 'grid load'
    df_analysis = consumption_df.copy()
    target_column = config.TARGET_CONSUMPTION_COLUMN
    logging.info(f"Selected '{target_column}' from consumption data for analysis.")

    # 2. Preprocess Data
    logging.info("--- Step 2: Preprocessing Data ---")

    # Convert timestamps
    df_analysis = preprocessing.preprocess_timestamps(df_analysis, 'Start date', 'End date')

    # Set time index
    df_analysis = preprocessing.set_time_index(df_analysis, 'Start date',duplicate_handling='first' # Or your choice: 'mean', 'last'
)


    # Clean the target numeric column
    if target_column in df_analysis.columns:
         df_analysis[target_column] = preprocessing.clean_numeric_column(df_analysis[target_column])
    else:
        logging.error(f"Target column '{target_column}' not found after initial loading/cleanup.")
        return # Exit if target column isn't available

    # Select only numeric columns for further processing like scaling/modeling later
    numeric_cols = df_analysis.select_dtypes(include=np.number).columns.tolist()
    logging.info(f"Numeric columns identified: {numeric_cols}")

    # Handle Missing Values (Example: using forward fill for numeric columns)
    # Apply ONLY to numeric columns to avoid issues with object types
    df_analysis = preprocessing.handle_missing_values(df_analysis, strategy='ffill', columns=numeric_cols)

    # Check data integrity after preprocessing
    logging.info(f"Data shape after preprocessing: {df_analysis.shape}")
    if df_analysis[target_column].isnull().any():
        logging.warning(f"Target column '{target_column}' still contains NaN values after handling. Review handling strategy.")
    else:
        logging.info(f"Target column '{target_column}' preprocessing complete. No NaNs detected.")

    #3. Feature Engineering ---
    logging.info("--- Step 3: Feature Engineering ---")
    df_analysis = preprocessing.add_time_features(df_analysis)

    # Define lags and window sizes
    lags_to_add = [1, 2, 3, 24, 48, 168] # 1h, 2h, 3h, 1 day, 2 days, 1 week ago
    window_sizes_for_rolling = [3, 6, 12, 24] # 3h, 6h, 12h, 24h windows

    # Ensure target_column is valid before creating lag/rolling features
    if target_column in df_analysis.columns:
        df_analysis = preprocessing.add_lag_features(df_analysis, target_column, lags_to_add)
        df_analysis = preprocessing.add_rolling_features(df_analysis, target_column, window_sizes_for_rolling, aggs=['mean', 'std', 'min', 'max'])
    else:
        logging.error(f"Target column '{target_column}' not found. Skipping lag and rolling features.")

    # After adding lag/rolling features, there will be NaNs at the beginning.
    # We need to decide how to handle them. Options:
    # 1. Drop rows with any NaNs: df_analysis.dropna(inplace=True)
    #    This reduces dataset size but ensures no NaNs for models that can't handle them.
    # 2. Fill with a value (e.g., 0, mean, median - but be careful with look-ahead bias if using overall mean/median)
    # 3. Some models (like tree-based) might handle NaNs, or specific imputation strategies can be used.
    # For now, let's drop them as many NNs require complete data.
    # The number of rows to drop will depend on the largest lag or window.
    initial_rows = len(df_analysis)
    df_analysis.dropna(inplace=True)
    rows_dropped = initial_rows - len(df_analysis)
    logging.info(f"Dropped {rows_dropped} rows due to NaNs introduced by lag/rolling features.")
    logging.info(f"Data shape after feature engineering and NaN handling: {df_analysis.shape}")

    if df_analysis.empty:
        logging.error("DataFrame is empty after feature engineering and NaN removal. Check lag/window parameters or data length.")
        return

    # ... (Rest of the pipeline: Model Training, Anomaly Detection, etc.) ...
    logging.info("--- Step 4: Model Training (Placeholder) ---")
    feature_columns = df_analysis.select_dtypes(include=np.number).columns.tolist()
    logging.info(f"Selected features for LSTM model ({len(feature_columns)} features): {feature_columns}")

    if not feature_columns:
        logging.error("No feature columns selected for the model. Exiting.")
        return

    # Scale features
    data_to_scale = df_analysis[feature_columns].copy()
    scaled_data_df, scaler = preprocessing.scale_features(data_to_scale) # Use the specific function
    
    if scaler is None:
        logging.error("Scaler was not fitted. Cannot proceed. Exiting.")
        return

    # Save the scaler
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, config.SCALER_FILENAME)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    scaled_data_values = scaled_data_df.values # Convert to NumPy array

    # Create sequences
    sequences = preprocessing.create_sequences(scaled_data_values, config.LSTM_SEQUENCE_LENGTH)

    if sequences.shape[0] == 0:
        logging.error("No sequences were created. Check sequence_length and data size. Exiting.")
        return

    # Convert to PyTorch Tensors
    # Our LSTM expects (batch, seq_len, num_features)
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(sequences_tensor) # Autoencoder target is the input itself
    data_loader = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
                                        # drop_last=True is good practice for training if last batch is small

    # Initialize model
    current_input_dim = sequences_tensor.shape[2] # Number of features
    model = LSTMAutoencoder(
        input_dim=current_input_dim,
        embedding_dim=config.LSTM_EMBEDDING_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        n_layers=config.LSTM_N_LAYERS,
        dropout_prob=config.LSTM_DROPOUT_PROB,
        device=device
    )
    
    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_FILENAME)

    # Train the model
    trained_model = train_model_pytorch(
        model,
        data_loader,
        config.TRAIN_N_EPOCHS,
        config.TRAIN_LEARNING_RATE,
        device,
        model_path
    )
    logging.info(f"LSTM Autoencoder trained and final version loaded from {model_path}")

    # --- Step 5: Anomaly Detection (Placeholder) ---
    logging.info("--- Step 5: Anomaly Detection (Placeholder) ---")
    # Here you would:
    # 1. Put the model in eval mode: trained_model.eval()
    # 2. Prepare your full dataset (or new data) using the same scaling and sequencing.
    # 3. Pass sequences through the trained_model to get reconstructions.
    # 4. Calculate reconstruction error (e.g., MSE) for each sequence.
    # 5. Anomalies are sequences with high reconstruction error (above a threshold).

    logging.info("--- Step 6: Visualization & Reporting (Placeholder) ---")
    logging.info("--- Pipeline Finished ---")


if __name__ == "__main__":
    run_pipeline()