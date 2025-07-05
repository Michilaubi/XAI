# model_prep.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def prepare_datetime_index(df, timestamp_col, datetime_format="%d.%m.%Y %H:%M:%S", drop_original=False):
    """
    Converts a timestamp column to datetime, sets it as index.
    
    Args:
        df (pd.DataFrame): Your input dataframe.
        timestamp_col (str): Name of the column containing timestamps.
        datetime_format (str): Format of the timestamp strings.
        drop_original (bool): Whether to drop the original timestamp column after setting index.
    
    Returns:
        pd.DataFrame: DataFrame with datetime index.
    """
    df = df.copy()
    df['parsed_datetime'] = pd.to_datetime(df[timestamp_col], format=datetime_format, errors='coerce')
    df = df.set_index('parsed_datetime')
    
    if drop_original:
        df = df.drop(columns=[timestamp_col])
    
    return df
    
def prepare_multistep_data(
    df: pd.DataFrame,
    input_columns: list,
    target_column: str,
    input_window: int,
    forecast_horizon: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    scale_data: bool = True
):
    """
    Prepares and optionally scales data for multistep time series forecasting.

    Returns:
        X_train_multi, y_train_multi,
        X_val_multi, y_val_multi,
        X_test_multi, y_test_multi,
        input_scaler, target_scaler
    """
    total_samples = len(df)
    num_features = len(input_columns)

    # Initialize input and output arrays
    X_multi = np.zeros((total_samples, input_window, num_features))
    y_multi = np.zeros((total_samples, forecast_horizon))

    # Create lagged input windows
    for i, col in enumerate(input_columns):
        for j in range(input_window):
            X_multi[:, j, i] = df[col].shift(input_window - j - 1).bfill()

    # Create multistep output targets
    for step in range(forecast_horizon):
        y_multi[:, step] = df[target_column].shift(-step - 1).ffill()

    # Remove samples that can't be used
    valid_range = slice(input_window, total_samples - forecast_horizon)
    X_multi = X_multi[valid_range]
    y_multi = y_multi[valid_range]

    # Split sizes
    train_size_multi = int(train_ratio * len(X_multi))
    val_size_multi = int(val_ratio * len(X_multi))

    # Split data
    X_train_multi = X_multi[:train_size_multi]
    y_train_multi = y_multi[:train_size_multi]

    X_val_multi = X_multi[train_size_multi:train_size_multi + val_size_multi]
    y_val_multi = y_multi[train_size_multi:train_size_multi + val_size_multi]

    X_test_multi = X_multi[train_size_multi + val_size_multi:]
    y_test_multi = y_multi[train_size_multi + val_size_multi:]

    input_scaler, target_scaler = None, None

    if scale_data:
        # Flatten X to 2D for scaling
        X_train_flat = X_train_multi.reshape(-1, num_features)
        input_scaler = StandardScaler().fit(X_train_flat)

        # Scale the inputs and reshape back to the original 3D shape
        X_train_multi = input_scaler.transform(X_train_flat).reshape(X_train_multi.shape)
        X_val_multi = input_scaler.transform(X_val_multi.reshape(-1, num_features)).reshape(X_val_multi.shape)
        X_test_multi = input_scaler.transform(X_test_multi.reshape(-1, num_features)).reshape(X_test_multi.shape)

        # Scale the target variable
        target_scaler = StandardScaler().fit(y_train_multi)
        y_train_multi = target_scaler.transform(y_train_multi)
        y_val_multi = target_scaler.transform(y_val_multi)
        y_test_multi = target_scaler.transform(y_test_multi)

    # Return the data and scalers
    return X_train_multi, y_train_multi, X_val_multi, y_val_multi, X_test_multi, y_test_multi, input_scaler, target_scaler

def inverse_scale_predictions(y_scaled, scaler):
    """
    Inverse scales the predictions using the fitted scaler.
    
    Args:
        y_scaled (numpy.ndarray or torch.Tensor): The scaled predictions.
        scaler (StandardScaler): The scaler object that was used for scaling.
        
    Returns:
        numpy.ndarray: The unscaled predictions.
    """
    if isinstance(y_scaled, torch.Tensor):
        y_scaled = y_scaled.detach().cpu().numpy()

    original_shape = y_scaled.shape
    print(f"Original shape: {original_shape}")  # Print the original shape

    # Squeeze only if the last dimension has size 1 (i.e., from (63081, 10, 1) to (63081, 10))
    if y_scaled.shape[-1] == 1:
        y_scaled_squeezed = y_scaled.squeeze(axis=-1)
        print(f"Shape after squeezing: {y_scaled_squeezed.shape}")  # Print the shape after squeezing
    else:
        y_scaled_squeezed = y_scaled
        print(f"No squeezing needed, shape is: {y_scaled_squeezed.shape}")

    # Reshape to 2D: (num_samples * forecast_horizon, 10) for inverse transformation
    # Flatten the first dimension (63081) and keep the 10 as is
    y_scaled_2d = y_scaled_squeezed.reshape(-1, y_scaled_squeezed.shape[1])
    print(f"Reshaped for inverse transform: {y_scaled_2d.shape}")  # Print the reshaped shape

    # Inverse transform the scaled data
    y_unscaled_2d = scaler.inverse_transform(y_scaled_2d)

    print(f"Shape after inverse transform: {y_unscaled_2d.shape}")  # Print the shape after inverse transform

    # Reshape back to the original shape (i.e., (num_samples, forecast_horizon))
    return y_unscaled_2d.reshape(original_shape[0], original_shape[1])
