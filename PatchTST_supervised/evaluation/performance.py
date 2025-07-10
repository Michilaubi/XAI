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

def get_preds_truths(model, loader, device, args=None, target_channel=-1):
    """
    Runs the model over the loader and collects predictions & true values.
    If `args` is given, and y comes in with length = args.label_len + args.pred_len,
    it will automatically slice off the first args.label_len steps so you end up
    with exactly args.pred_len ground-truth steps to compare to your preds.
    Returns:
      preds: np.ndarray of shape [N_samples, H_pred, 1]
      truths: np.ndarray of shape [N_samples, H_pred, 1]
    """
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].float().to(device), batch[1].float().to(device)
            # ensure y is [B, T, C]
            if y.dim() == 2:
                y = y.unsqueeze(-1)

            # forward
            out = model(x)
            forecasts = out[0] if isinstance(out, tuple) else out  # [B, T_out, C]

            # if we passed in args, crop off the label-len part
            if args is not None:
                expected = args.label_len + args.pred_len
                if y.size(1) == expected:
                    y = y[:, -args.pred_len:, :]

            # select the channel
            p = forecasts[:, :, target_channel].unsqueeze(-1)  # [B, T_out, 1]
            t = y[:, :, target_channel].unsqueeze(-1)          # [B, T_true, 1]

            all_preds.append(p.cpu().numpy())
            all_trues.append(t.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    return preds, trues


def stepwise_errors(preds, truths, target_scaler=None):
    """
    preds, truths: np.ndarray, shape [N, H, 1]
    target_scaler: the fitted StandardScaler used on y (or None if you never scaled y)
    
    Returns:
        rmse_list, mae_list  each of length H, in the *original* units of y
    """
    # reshape to [N*H, 1] so we can inverse_transform, then back to [N, H]
    if target_scaler is not None:
        N, H, _ = preds.shape
        flat_pred = preds.reshape(-1, 1)
        flat_true = truths.reshape(-1, 1)
        
        unscaled_pred = target_scaler.inverse_transform(flat_pred).reshape(N, H)
        unscaled_true = target_scaler.inverse_transform(flat_true).reshape(N, H)
    else:
        # just drop that last dim
        unscaled_pred = preds[..., 0]
        unscaled_true = truths[..., 0]
    
    rmse_list, mae_list = [], []
    for h in range(unscaled_pred.shape[1]):
        y_pred = unscaled_pred[:, h]
        y_true = unscaled_true[:, h]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
    
    return rmse_list, mae_list
