import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_timestamps(df: pd.DataFrame, start_col: str, end_col: str = None) -> pd.DataFrame:
    
    logging.info(f"Preprocessing timestamp columns: {start_col}" + (f", {end_col}" if end_col else ""))
    if start_col in df.columns:
        df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        # Check for parsing errors (NaT values)
        if df[start_col].isnull().any():
            logging.warning(f"Found NaT values in '{start_col}' after conversion. Check original data format.")
    else:
        logging.warning(f"Start column '{start_col}' not found in DataFrame.")

    if end_col and end_col in df.columns:
        df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
        if df[end_col].isnull().any():
            logging.warning(f"Found NaT values in '{end_col}' after conversion. Check original data format.")
    elif end_col:
        logging.warning(f"End column '{end_col}' not found in DataFrame.")

    return df

def set_time_index(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    
    if index_col not in df.columns:
        logging.error(f"Index column '{index_col}' not found.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[index_col]):
         logging.warning(f"Index column '{index_col}' is not datetime. Attempting conversion.")
         df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
         if df[index_col].isnull().any():
             logging.error(f"Failed to convert index column '{index_col}' to datetime. Cannot set index.")
             return df

    logging.info(f"Setting '{index_col}' as index.")
    df = df.set_index(index_col)
    df = df.sort_index()
    # Check for duplicate indices
    if df.index.duplicated().any():
        logging.warning(f"Duplicate timestamps found in index '{index_col}'. Consider aggregation or removal.")
        # Optional: Handle duplicates, e.g., keep first or average
        # df = df[~df.index.duplicated(keep='first')]
    return df

def clean_numeric_column(series: pd.Series) -> pd.Series:
    
    name = series.name
    logging.debug(f"Cleaning numeric column: {name}")
    if series.dtype == 'object':
        # Replace comma decimal separator with dot BEFORE removing other chars
        series = series.astype(str).str.replace(',', '.', regex=False)
        # Remove thousands separators (like spaces or dots if not decimal) - BE CAREFUL with locale
        # This example assumes '.' is only for decimal after comma replacement
        # You might need more robust parsing based on exact format
        series = series.str.replace(r'[^\d.]', '', regex=True)
        # Handle potential multiple dots after cleaning
        series = series.str.replace(r'(\.\d*)\.', r'\1', regex=True) # Keep only the first dot

    # Convert to numeric, coercing errors
    numeric_series = pd.to_numeric(series, errors='coerce')

    # Report issues
    original_non_na = series.notna().sum()
    converted_non_na = numeric_series.notna().sum()
    if original_non_na > converted_non_na:
        logging.warning(f"Column '{name}': {original_non_na - converted_non_na} values could not be converted to numeric.")

    return numeric_series


def handle_missing_values(df: pd.DataFrame, strategy: str = 'ffill', columns: list = None) -> pd.DataFrame:
    
    target_cols = columns
    if target_cols is None:
        # Apply only to numeric columns if no specific columns are given
        target_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not target_cols:
        logging.warning("No numeric columns found or specified for missing value handling.")
        return df

    logging.info(f"Handling missing values using strategy '{strategy}' for columns: {target_cols}")

    for col in target_cols:
        if col not in df.columns:
            logging.warning(f"Column '{col}' specified for missing value handling not found.")
            continue

        if df[col].isnull().any():
            logging.info(f"Handling NaNs in column '{col}'...")
            if strategy == 'ffill':
                df[col] = df[col].fillna(method='ffill')
            elif strategy == 'bfill':
                df[col] = df[col].fillna(method='bfill')
            elif strategy == 'mean':
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                logging.info(f"Filled NaNs in '{col}' with mean: {mean_val}")
            elif strategy == 'median':
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logging.info(f"Filled NaNs in '{col}' with median: {median_val}")
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
                logging.info(f"Filled NaNs in '{col}' with 0.")
            elif strategy == 'drop':
                # Drop rows with NaN in this specific column - use carefully!
                df = df.dropna(subset=[col])
                logging.info(f"Dropped rows with NaNs in column '{col}'.")
            else:
                logging.warning(f"Unknown missing value strategy: {strategy}. No action taken for column '{col}'.")
        else:
            logging.info(f"No NaNs found in column '{col}'.")

    # Handle potential NaNs remaining after ffill (at the start) or bfill (at the end)
    if strategy in ['ffill', 'bfill']:
         df.fillna(0, inplace=True) # Example: fill remaining NaNs with 0 after ffill/bfill

    return df