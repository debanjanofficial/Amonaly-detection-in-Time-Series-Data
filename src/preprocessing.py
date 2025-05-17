import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import torch


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
         df.fillna(0, inplace=True)

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features to the DataFrame from its DatetimeIndex.
    Assumes df has a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("DataFrame does not have a DatetimeIndex. Cannot add time features.")
        return df

    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek # Monday=0, Sunday=6
    df_copy['dayofmonth'] = df_copy.index.day
    df_copy['dayofyear'] = df_copy.index.dayofyear
    df_copy['month'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['weekofyear'] = df_copy.index.isocalendar().week.astype(int)
    df_copy['is_weekend'] = (df_copy.index.dayofweek >= 5).astype(int) # 1 if weekend, 0 if weekday

    logging.info(f"Added time features: {['hour', 'dayofweek', 'dayofmonth', 'dayofyear', 'month', 'year', 'quarter', 'weekofyear', 'is_weekend']}")
    return df_copy

def add_lag_features(df: pd.DataFrame, target_column: str, lags: list) -> pd.DataFrame:
    """
    Adds lag features for a specific target column.

    Args:
        df (pd.DataFrame): Input DataFrame with DatetimeIndex.
        target_column (str): The column to create lag features from.
        lags (list of int): A list of lag periods (e.g., [1, 2, 24] for 1h, 2h, 24h ago).
    """
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found for lag feature creation.")
        return df

    df_copy = df.copy()
    for lag in lags:
        if lag <= 0:
            logging.warning(f"Lag value must be positive. Skipping lag: {lag}")
            continue
        lag_col_name = f'{target_column}_lag_{lag}'
        df_copy[lag_col_name] = df_copy[target_column].shift(lag)
        logging.info(f"Added lag feature: {lag_col_name}")

    # Lags will introduce NaNs at the beginning, which need to be handled
    return df_copy

def add_rolling_features(df: pd.DataFrame, target_column: str, window_sizes: list, aggs: list = None) -> pd.DataFrame:
    
    if aggs is None:
        aggs = ['mean', 'std']

    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found for rolling feature creation.")
        return df

    df_copy = df.copy()
    for window in window_sizes:
        if window <= 0:
            logging.warning(f"Window size must be positive. Skipping window: {window}")
            continue
        
        # min_periods=1 ensures that we get a value even if the window is not full (at the start)
        # 'closed="left"' aligns the window so it uses past data up to (but not including) the current observation.
        # For anomaly detection, using 'closed="neither"' or 'closed="right"' (default) might be more common
        # to include the current point or a centered window if appropriate.
        # Let's use default (closed='right') which includes the current point in the calculation if not shifted.
        # To calculate based on *past* data, we'd shift the target first or use a specific windowing lib.
        # For simplicity here, we'll use the standard rolling which includes the current point if not careful.
        # A common practice for features is to use past data:
        # df_copy[target_column].shift(1).rolling(window=window, min_periods=1)

        for agg in aggs:
            roll_col_name = f'{target_column}_roll_{agg}_{window}h'
            try:
                # Calculate rolling features on PAST data by shifting the target column by 1
                # This prevents data leakage from the current timestep if these features are used for prediction
                # For pure anomaly detection on current point, you might not shift.
                # Let's use past data for features.
                series_to_roll = df_copy[target_column].shift(1) # Shift by 1 to use only past data
                df_copy[roll_col_name] = series_to_roll.rolling(window=window, min_periods=1).agg(agg)
                logging.info(f"Added rolling feature: {roll_col_name}")
            except Exception as e:
                logging.error(f"Could not apply rolling {agg} for window {window} on {target_column}: {e}")
    return df_copy

from typing import Tuple

def scale_features(df: pd.DataFrame, exclude_cols: list = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales numeric features in the DataFrame using MinMaxScaler.
    Non-numeric columns and columns in exclude_cols are not scaled.

    Args:
        df (pd.DataFrame): DataFrame with features.
        exclude_cols (list, optional): List of column names to exclude from scaling.

    Returns:
        pd.DataFrame: DataFrame with scaled numeric features.
        MinMaxScaler: The fitted scaler object.
    """
    df_scaled = df.copy()
    scalers = {} # To store scalers for each column if needed, or one for all

    numeric_cols = df_scaled.select_dtypes(include=np.number).columns.tolist()
    if exclude_cols:
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    else:
        cols_to_scale = numeric_cols
    
    if not cols_to_scale:
        logging.warning("No columns found to scale.")
        return df_scaled, None # Return None if no scaler fitted

    scaler = MinMaxScaler()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    logging.info(f"Scaled columns: {cols_to_scale}")
    
    return df_scaled, scaler # Return the single scaler used for all these columns

def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Transforms a 2D array of time series data into 3D sequences.

    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        sequence_length (int): The length of each output sequence.

    Returns:
        np.ndarray: 3D array of shape (n_samples - sequence_length + 1, sequence_length, n_features).
    """
    xs = []
    for i in range(len(data) - sequence_length + 1):
        x = data[i:(i + sequence_length)]
        xs.append(x)
    logging.info(f"Created sequences with length {sequence_length}. Number of sequences: {len(xs)}")
    return np.array(xs)