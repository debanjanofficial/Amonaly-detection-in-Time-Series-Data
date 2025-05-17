# main.py

import pandas as pd
import numpy as np
import logging
import torch # Import torch to trigger MPS check if available

# Import project modules
import config
from src import data_loader, preprocessing, utils # visualization will be added later
# from src.models import ... # Import model modules when ready
# from src.anomaly_detector import AnomalyDetector # Import detector when ready

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    """Runs the anomaly detection pipeline."""

    # 0. Setup Environment
    logging.info("--- Starting Anomaly Detection Pipeline ---")
    device = utils.get_device() # Check for MPS early on

    # 1. Load Data
    logging.info("--- Step 1: Loading Data ---")
    consumption_df = data_loader.load_data(
        config.CONSUMPTION_DATA_PATH,
        delimiter=';'
        # Specify date columns for direct parsing if known and reliable
        # date_cols=['Start date', 'End date']
    )
    generation_df = data_loader.load_data(
        config.GENERATION_DATA_PATH,
        delimiter=';'
        # date_cols=['Start date', 'End date']
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
    df_analysis = preprocessing.set_time_index(df_analysis, 'Start date')

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

    # --- Placeholder for next steps ---
    logging.info("--- Step 3: Feature Engineering (Placeholder) ---")
    # Add feature engineering calls here (e.g., time features, lags, rolling stats)
    # Example: df_analysis = preprocessing.add_time_features(df_analysis)

    logging.info("--- Step 4: Model Training (Placeholder) ---")
    # Instantiate and train your PyTorch model (e.g., LSTM Autoencoder)
    # scaler = StandardScaler() # Need scaler for many models
    # scaled_data = scaler.fit_transform(df_analysis[[target_column]])
    # sequences = create_sequences(scaled_data, config.SEQUENCE_LENGTH)
    # dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # model = YourLSTMAutoencoder(input_dim=1, latent_dim=32).to(device)
    # train_model(model, loader, device) # Your training loop

    logging.info("--- Step 5: Anomaly Detection (Placeholder) ---")
    # Use the trained model to get anomaly scores (e.g., reconstruction errors)
    # anomaly_scores = detect_anomalies(model, all_data_loader, device)
    # anomalies = df_analysis[anomaly_scores > config.ANOMALY_THRESHOLD]

    logging.info("--- Step 6: Visualization & Reporting (Placeholder) ---")
    # Use visualization.py to plot results
    # visualization.plot_anomalies(df_analysis, target_column, anomalies)

    logging.info("--- Pipeline Finished ---")


if __name__ == "__main__":
    run_pipeline()