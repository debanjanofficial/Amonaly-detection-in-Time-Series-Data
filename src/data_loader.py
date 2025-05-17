import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str, delimiter: str = ';', date_cols: list = None) -> pd.DataFrame:
    
    if not os.path.exists(filepath):
        logging.error(f"File not found at path: {filepath}")
        return pd.DataFrame()

    logging.info(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            low_memory=False, 
            parse_dates=date_cols,
        )
        logging.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def initial_column_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    
    df.columns = df.columns.str.strip()
    # Add any other generic column name cleaning needed
    logging.info("Performed initial column name cleanup.")
    return df