import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_timestamps(df: pd.DataFrame, start_col: str, end_col: str = None) -> pd.DataFrame:
    """
    Ensures timestamp columns are in datetime format using the specified format string.
    """
    logging.info(f"Preprocessing timestamp columns: {start_col}" + (f", {end_col}" if end_col else ""))

    # ***********************************************************************************
    # *** Set the confirmed date format string here ***
    actual_date_format_in_csv = '%b %d, %Y %I:%M %p'
    # ***********************************************************************************

    if actual_date_format_in_csv:
        logging.info(f"Attempting to parse dates with specified format: {actual_date_format_in_csv}")
    else:
        # This part should ideally not be reached if you've confirmed the format
        logging.warning("No specific date format provided. Date parsing might be slow or error-prone.")


    for col_name in [start_col, end_col]:
        if col_name and col_name in df.columns:
            # Ensure the column is of string type before attempting to_datetime with a format string,
            # especially if parse_dates in read_csv might have already converted it (or parts of it).
            if not pd.api.types.is_datetime64_any_dtype(df[col_name]): # Only process if not already datetime
                try:
                    df[col_name] = pd.to_datetime(df[col_name].astype(str), format=actual_date_format_in_csv, errors='raise')
                    logging.info(f"Successfully parsed '{col_name}' using format '{actual_date_format_in_csv}'.")
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not parse '{col_name}' with format '{actual_date_format_in_csv}' (Error: {e}). Falling back to inference.")
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce') # Fallback if explicit format fails
            else:
                logging.info(f"Column '{col_name}' is already in datetime format.")


            if df[col_name].isnull().any():
                num_nulls = df[col_name].isnull().sum()
                original_non_nulls = df[col_name].notnull().sum() + num_nulls # Count before potential new NaTs
                logging.warning(f"Found {num_nulls} NaT (Not a Time) values in '{col_name}' out of {original_non_nulls} original entries after conversion. Review data and format string.")
        elif col_name:
            logging.warning(f"Column '{col_name}' for date parsing not found in DataFrame.")
    return df

def set_time_index(df: pd.DataFrame, index_col: str, duplicate_handling: str = 'first') -> pd.DataFrame:
    if index_col not in df.columns:
        logging.error(f"Index column '{index_col}' not found.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[index_col]):
        logging.warning(f"Index column '{index_col}' is not datetime. It should have been converted by preprocess_timestamps.")
        # Attempt one last time or error out
        df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
        if df[index_col].isnull().any():
            logging.error(f"Failed to convert index column '{index_col}' to datetime. Cannot set index.")
            return df

    logging.info(f"Setting '{index_col}' as index.")
    # Ensure index_col is not already the index to avoid issues with reset_index if it's not set
    if df.index.name != index_col:
         # If index_col is a regular column, set it as index
        if index_col in df.columns:
            df = df.set_index(index_col)
        else: # This case should not happen if previous checks are fine
            logging.error(f"Column '{index_col}' cannot be set as index as it does not exist.")
            return df
    # else: index is already set to index_col, no action needed for setting


    # Check for duplicate indices
    if df.index.duplicated().any():
        logging.warning(f"Duplicate timestamps found in index '{df.index.name}'. Handling strategy: {duplicate_handling}")
        if duplicate_handling == 'first':
            df = df[~df.index.duplicated(keep='first')]
        elif duplicate_handling == 'last':
            df = df[~df.index.duplicated(keep='last')]
        elif duplicate_handling == 'mean':
            numeric_df = df.select_dtypes(include=np.number)
            non_numeric_df = df.select_dtypes(exclude=np.number)
            df_mean = numeric_df.groupby(numeric_df.index).mean()
            if not non_numeric_df.empty:
                df_non_numeric_agg = non_numeric_df.groupby(non_numeric_df.index).first() # Or another strategy
                df = pd.concat([df_mean, df_non_numeric_agg], axis=1)
            else:
                df = df_mean
            logging.info("Aggregated duplicate indices by mean for numeric columns and first for others.")
        # Else (None or unknown strategy), do nothing and leave duplicates
    else:
        logging.info(f"No duplicate timestamps found in index '{df.index.name}'.")

    df = df.sort_index()
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