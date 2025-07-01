import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch

def evaluate(model, loader, target_channel=-1, scaler=None):
    """
    Returns a dict with
       MSE, RMSE, MAE  (on the *scaled* data)
    and if `scaler` is provided,
       MSE_unscaled, RMSE_unscaled, MAE_unscaled
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device).float()

            # forward
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]   # Model may return (forecast, ...)
            # out.shape == [B, pred_len, C]

            # make sure y has a channel dimension
            if y.dim() == 2:
                # IMV-LSTM: y is [B, T] → [B, T, 1]
                y = y.unsqueeze(-1)
            y = y.to(device)

            pred_len = out.shape[1]

            # pull out only the *last* pred_len steps of y
            # (works whether y was [B, T_pred] or [B, T_enc+T_pred])
            y_true = y[:, -pred_len:, target_channel]      # [B, pred_len]
            y_pred = out[:, :, target_channel]             # [B, pred_len]

            all_true.append(y_true.cpu().numpy().ravel())
            all_pred.append(y_pred.cpu().numpy().ravel())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # scaled metrics
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)

    results = {
        "MSE":   mse,
        "RMSE":  rmse,
        "MAE":   mae
    }

    if scaler is not None:
        # scaler.inverse_transform expects 2D arrays
        yt_un = scaler.inverse_transform(y_true.reshape(-1,1)).ravel()
        yp_un = scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()
        mse_u  = mean_squared_error(yt_un, yp_un)
        rmse_u = np.sqrt(mse_u)
        mae_u  = mean_absolute_error(yt_un, yp_un)
        results.update({
            "MSE_unscaled":   mse_u,
            "RMSE_unscaled":  rmse_u,
            "MAE_unscaled":   mae_u
        })

    return results

def get_preds_truths(model, loader, device, target_channel=-1):
    """
    Runs the model over the loader and collects predictions & true values.
    Returns:
      preds: np.ndarray of shape [N_samples, H, 1]
      truths: np.ndarray of shape [N_samples, H, 1]
    """
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].float().to(device), batch[1].float().to(device)

            # ---- make sure y is [B, H, 1] even if loader gives [B, H] ----
            if y.dim() == 2:
                y = y.unsqueeze(-1)

            # forward
            out = model(x)
            forecasts = out[0] if isinstance(out, tuple) else out

            # slice out the channel and get shape [B, H, 1]
            p = forecasts[:, :, target_channel].unsqueeze(-1)
            t = y[:, :, target_channel].unsqueeze(-1)

            all_preds.append(p.cpu().numpy())
            all_trues.append(t.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)


def stepwise_errors(preds, truths):
    """
    preds, truths: np.ndarray [N, H, 1]
    Returns lists of length H: rmse_list, mae_list
    """
    H = preds.shape[1]
    rmse_list, mae_list = [], []
    for h in range(H):
        y_pred = preds[:, h, 0]
        y_true = truths[:, h, 0]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
    return rmse_list, mae_list