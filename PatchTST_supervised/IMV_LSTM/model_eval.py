# forecasting_utils.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, data_loader, device, target_scaler=None, print_metrics=True):
    model.eval()
    predictions = []
    targets = []
    alphas_all = []
    betas_all = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred, alphas, betas = model(X_batch)

            predictions.append(y_pred.cpu())
            targets.append(y_batch.cpu())
            alphas_all.append(alphas.cpu())
            betas_all.append(betas.cpu())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    alphas_all = torch.cat(alphas_all).numpy()
    betas_all = torch.cat(betas_all).numpy()

    # Optional inverse scaling
    if target_scaler is not None:
        from copy import deepcopy

        def inverse_scale(y):
            if y.ndim == 3 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            y_flat = y.reshape(-1, target_scaler.mean_.shape[0])
            return target_scaler.inverse_transform(y_flat).reshape(y.shape)

        predictions = inverse_scale(predictions)
        targets = inverse_scale(targets)

    # Compute metrics
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = mean_squared_error(targets.flatten(), predictions.flatten(), squared=False)
    r2 = r2_score(targets.flatten(), predictions.flatten())

    if print_metrics:
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    return predictions, targets, alphas_all, betas_all, {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_conicity_model(model, data_loader, device, target_scaler=None, print_metrics=True):
    model.eval()
    predictions = []
    targets = []
    alphas_all = []
    betas_all = []
    conicity_scores = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred, alphas, betas, conicity = model(X_batch)

            predictions.append(y_pred.cpu())
            targets.append(y_batch.cpu())
            alphas_all.append(alphas.cpu())
            betas_all.append(betas.cpu())
            conicity_scores.append(conicity.item())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    alphas_all = torch.cat(alphas_all).numpy()
    betas_all = torch.cat(betas_all).numpy()

    # Optional inverse scaling
    if target_scaler is not None:
        from copy import deepcopy

        def inverse_scale(y):
            if y.ndim == 3 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            y_flat = y.reshape(-1, target_scaler.mean_.shape[0])
            return target_scaler.inverse_transform(y_flat).reshape(y.shape)

        predictions = inverse_scale(predictions)
        targets = inverse_scale(targets)

    # Compute metrics
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = mean_squared_error(targets.flatten(), predictions.flatten(), squared=False)
    r2 = r2_score(targets.flatten(), predictions.flatten())
    avg_conicity = np.mean(conicity_scores)

    if print_metrics:
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Avg Conicity: {avg_conicity:.4f}")

    return predictions, targets, alphas_all, betas_all, {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Conicity": avg_conicity
    }


def compute_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_attention_weights(alphas, betas, sample_idx=0, input_var_names=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 5))

    # Alpha: temporal attention
    plt.subplot(1, 2, 1)
    alpha_sample = alphas[sample_idx].squeeze().T
    img = plt.imshow(alpha_sample, aspect="auto", cmap="viridis")
    plt.title("Temporal Attention (Alpha)")
    plt.xlabel("Time Step")
    plt.ylabel("Input Variable")
    
    if input_var_names is not None:
        plt.yticks(ticks=np.arange(len(input_var_names)), labels=input_var_names)
    
    plt.colorbar(img, orientation='vertical', label="Attention Weight")

    # Beta: average over forecast steps
    beta_sample = betas[sample_idx].squeeze(-1)  # shape: (forecast_steps, context_steps)
    beta_avg = beta_sample.mean(axis=0)          # shape: (context_steps,)

    plt.subplot(1, 2, 2)
    plt.plot(beta_avg, marker='o')
    plt.title("Context Attention (Beta, Averaged)")
    plt.xlabel("Context Index")
    plt.ylabel("Weight")

    plt.tight_layout()
    plt.show()


def plot_beta_attention_over_forecast_steps(betas, sample_idx=0):
    """
    Plots Beta (context) attention weights for each forecast step in a multi-step prediction.

    Parameters:
        betas (np.ndarray or torch.Tensor): shape (batch_size, forecast_horizon, context_length)
        sample_idx (int): index of the sample in the batch to visualize
    """
    if isinstance(betas, torch.Tensor):
        betas = betas.detach().cpu().numpy()
    
    beta_sample = betas[sample_idx]  # shape: (forecast_horizon, context_length)
    
    plt.figure(figsize=(10, 6))
    for step in range(beta_sample.shape[0]):
        plt.plot(
            beta_sample[step], 
            label=f"Forecast Step {step+1}", 
            marker='o', 
            alpha=0.8
        )

    plt.title("Context Attention (Beta) Over Forecast Steps")
    plt.xlabel("Context Index")
    plt.ylabel("Attention Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_forecast_vs_actual_per_step(preds, targets, sample_idx=0):
    """
    Plots actual vs. predicted values for each forecast step.

    Parameters:
        preds (np.ndarray or torch.Tensor): shape (batch_size, forecast_horizon)
        targets (np.ndarray or torch.Tensor): shape (batch_size, forecast_horizon)
        sample_idx (int): which sample in the batch to visualize
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    pred_sample = preds[sample_idx]
    target_sample = targets[sample_idx]
    horizon = len(pred_sample)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, horizon + 1), target_sample, label="Actual", marker='o')
    plt.plot(range(1, horizon + 1), pred_sample, label="Forecast", marker='x')
    
    plt.title(f"Forecast vs Actual (Sample {sample_idx})")
    plt.xlabel("Forecast Step")
    plt.ylabel("Value")
    plt.xticks(range(1, horizon + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_average_forecast_vs_actual_per_step(preds, targets):
    """
    Plots the average forecast vs. actual values for each forecast step across all samples.

    Parameters:
        preds (np.ndarray or torch.Tensor): shape (batch_size, forecast_horizon)
        targets (np.ndarray or torch.Tensor): shape (batch_size, forecast_horizon)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Calculate the mean of predictions and targets per forecast step
    avg_preds = np.mean(preds, axis=0)
    avg_targets = np.mean(targets, axis=0)
    
    horizon = len(avg_preds)

    # Plot the average forecast vs actual for each forecast step
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, horizon + 1), avg_targets, label="Average Actual", marker='o', linestyle='--')
    plt.plot(range(1, horizon + 1), avg_preds, label="Average Forecast", marker='x', linestyle='-')
    
    plt.title("Average Forecast vs Actual (All Samples)")
    plt.xlabel("Forecast Step")
    plt.ylabel("Value")
    plt.xticks(range(1, horizon + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_large_errors(y_true, y_pred, threshold=0.5, forecast_step=0):
    errors = np.abs(y_true[:, forecast_step] - y_pred[:, forecast_step])
    indices = np.where(errors > threshold)[0]

    if len(indices) == 0:
        print(f"No predictions with error greater than {threshold} for t+{forecast_step+1}.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(y_true[indices, forecast_step], label="Actual", marker='o')
    plt.plot(y_pred[indices, forecast_step], label="Predicted", marker='x')
    plt.title(f"Large Errors (|Pred - Actual| > {threshold}) at t+{forecast_step+1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_forecast_steps(preds, targets, forecast_horizon=5):
    """
    Plots actual vs predicted values for each forecast step (t+1, t+2, ..., t+forecast_horizon).
    
    Parameters:
        preds (np.ndarray or torch.Tensor): Forecast predictions with shape (batch_size, forecast_horizon)
        targets (np.ndarray or torch.Tensor): Actual target values with shape (batch_size, forecast_horizon)
        forecast_horizon (int): The number of forecast steps (t+1, t+2, ..., t+horizon)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    horizon = forecast_horizon
    batch_size = preds.shape[0]

    # Create subplots for each forecast step
    plt.figure(figsize=(15, 3 * horizon))

    for step in range(horizon):
        plt.subplot(horizon, 1, step + 1)

        # Plot actual vs predicted for each forecast step (t+1, t+2, ..., t+horizon)
        plt.plot(targets[:, step], label=f"Actual (t+{step+1})", marker='o', linestyle='-', color='blue')
        plt.plot(preds[:, step], label=f"Predicted (t+{step+1})", marker='x', linestyle='-', color='red')

        # Labels and title
        plt.title(f"Forecast Step t+{step+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()



def plot_forecast_steps_with_error_bands(preds, targets, forecast_horizon=5, subset_size=10, random_subset=False):
    """
    Plots actual vs predicted values for each forecast step (t+1, t+2, ..., t+forecast_horizon) 
    with error bands and a subset of samples.
    
    Parameters:
        preds (np.ndarray or torch.Tensor): Forecast predictions with shape (batch_size, forecast_horizon)
        targets (np.ndarray or torch.Tensor): Actual target values with shape (batch_size, forecast_horizon)
        forecast_horizon (int): The number of forecast steps (t+1, t+2, ..., t+horizon)
        subset_size (int): Number of samples to display (subset of the batch)
        random_subset (bool): If True, selects a random subset of samples.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    horizon = forecast_horizon
    batch_size = preds.shape[0]

    # Select a random subset of samples if random_subset is True
    if random_subset:
        subset_idx = np.random.choice(batch_size, subset_size, replace=False)  # random sample indices
    else:
        subset_idx = np.arange(min(subset_size, batch_size))  # first `subset_size` samples

    # Create subplots for each forecast step
    plt.figure(figsize=(15, 3 * horizon))

    for step in range(horizon):
        plt.subplot(horizon, 1, step + 1)

        # Calculate the mean and standard deviation of the predictions at each forecast step
        preds_step = preds[subset_idx, step]
        targets_step = targets[subset_idx, step]
        pred_mean = preds_step
        pred_std = np.std(preds_step, axis=0)

        # Plot actual vs predicted for each forecast step (t+1, t+2, ..., t+horizon)
        plt.plot(targets_step, label=f"Actual (t+{step+1})", marker='o', linestyle='--', color='blue')
        plt.plot(pred_mean, label=f"Predicted (t+{step+1})", marker='x', linestyle='-', color='red')

        # Add error bands: +/- 1 standard deviation around the predicted mean
        plt.fill_between(range(len(subset_idx)), 
                         pred_mean - pred_std, 
                         pred_mean + pred_std, 
                         color='red', alpha=0.2, label="Prediction Error Band (±1 std)")

        # Labels and title
        plt.title(f"Forecast Step t+{step+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_forecast_steps_with_error_bands_deviation(preds, targets, forecast_horizon=5, subset_size=10, deviation_threshold=None):
    """
    Plots actual vs predicted values for each forecast step (t+1, t+2, ..., t+forecast_horizon) 
    with error bands and a subset of samples where the deviation exceeds a threshold.
    
    Parameters:
        preds (np.ndarray or torch.Tensor): Forecast predictions with shape (batch_size, forecast_horizon)
        targets (np.ndarray or torch.Tensor): Actual target values with shape (batch_size, forecast_horizon)
        forecast_horizon (int): The number of forecast steps (t+1, t+2, ..., t+horizon)
        subset_size (int): Number of samples to display (subset of the batch)
        deviation_threshold (float): Minimum deviation between predicted and actual values for inclusion.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    horizon = forecast_horizon
    batch_size = preds.shape[0]

    # Calculate the absolute deviation between predictions and targets
    deviation = np.abs(preds - targets)

    # If deviation_threshold is provided, select samples where deviation exceeds the threshold
    if deviation_threshold is not None:
        # Calculate the mean deviation across all forecast steps for each sample
        mean_deviation = np.mean(deviation, axis=1)
        subset_idx = np.where(mean_deviation > deviation_threshold)[0]  # Samples where deviation > threshold
    else:
        subset_idx = np.arange(min(subset_size, batch_size))  # first `subset_size` samples

    # Create subplots for each forecast step
    plt.figure(figsize=(15, 3 * horizon))

    for step in range(horizon):
        plt.subplot(horizon, 1, step + 1)

        # Calculate the mean and standard deviation of the predictions at each forecast step
        preds_step = preds[subset_idx, step]
        targets_step = targets[subset_idx, step]
        pred_mean = preds_step
        pred_std = np.std(preds_step, axis=0)

        # Plot actual vs predicted for each forecast step (t+1, t+2, ..., t+horizon)
        plt.plot(targets_step, label=f"Actual (t+{step+1})", marker='o', linestyle='--', color='blue')
        plt.plot(pred_mean, label=f"Predicted (t+{step+1})", marker='x', linestyle='-', color='red')

        # Add error bands: +/- 1 standard deviation around the predicted mean
        plt.fill_between(range(len(subset_idx)), 
                         pred_mean - pred_std, 
                         pred_mean + pred_std, 
                         color='red', alpha=0.2, label="Prediction Error Band (±1 std)")

        # Labels and title
        plt.title(f"Forecast Step t+{step+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, num_samples=5):
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.plot(y_true[i], label="True", marker='o')
        plt.plot(y_pred[i], label="Predicted", marker='x')
        plt.title(f"Sample {i}")
        plt.xlabel("Forecast Step")
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()


