# utils_imvlstm.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def imvlstm_attention_randomization_check(
    model,
    loader,
    device,
    target_channel: int = -1,
    metric=mean_squared_error
):
    """
    1) Do a normal forward→MSE on target_channel
    2) Shuffle alphas along the encoding-time axis (for each batch, feature)
       and rerun via forward_with_alphas_betas (keeping betas fixed)
    3) Measure MSE again.
    Returns (orig_avg_mse, rand_avg_mse)
    """
    model.eval()
    orig_losses, rand_losses = [], []
    # stash the original override-capable forward method
    orig_fw = model.forward_with_alphas_betas

    for xb, yb in tqdm(loader, desc="IMV-LSTM rand-check"):
        xb, yb = xb.to(device), yb.to(device)
        if yb.dim() == 2:
            yb = yb.unsqueeze(-1)

        # — baseline forward —
        with torch.no_grad():
            y_hat, alphas, betas_all, _ = model(xb)
        y_true = yb[:, :, target_channel].cpu().numpy().ravel()
        y_pred = y_hat[:, :, target_channel].cpu().numpy().ravel()
        orig_losses.append(metric(y_true, y_pred))

        # — shuffle alphas along time axis —
        B, enc_len, V, _ = alphas.shape
        a = alphas.squeeze(-1)           # [B, enc_len, V]
        a_shuf = a.clone()
        for b in range(B):
            for v in range(V):
                perm = torch.randperm(enc_len, device=device)
                a_shuf[b, :, v] = a[b, perm, v]
        a_shuf = a_shuf.unsqueeze(-1)    # back to [B, enc_len, V, 1]

        # — rerun with shuffled alphas, original betas —
        with torch.no_grad():
            y_hat_rand, _, _, _ = model.forward_with_alphas_betas(
                xb,
                alphas_override = a_shuf,
                betas_override  = betas_all
            )
        y_pred_rand = y_hat_rand[:, :, target_channel].cpu().numpy().ravel()
        rand_losses.append(metric(y_true, y_pred_rand))

    # restore original method
    model.forward_with_alphas_betas = orig_fw

    return float(np.mean(orig_losses)), float(np.mean(rand_losses))

def imvlstm_beta_randomization_check(
    model, loader, device,
    target_channel: int = -1,
    metric=mean_squared_error
):
    """
    1) Run a normal forward, record MSE on target_channel.
    2) Shuffle betas across the forecast-time axis for each (batch, variable).
    3) Re-run via forward_with_alphas_betas (keep alphas fixed), record MSE.
    Returns (orig_avg_mse, rand_avg_mse).
    """
    model.eval()
    orig_losses, rand_losses = [], []

    # stash the override-capable forward method
    orig_fw = model.forward_with_alphas_betas

    for xb, yb in tqdm(loader, desc="beta-rand-check"):
        xb, yb = xb.to(device), yb.to(device)
        if yb.dim() == 2:  
            yb = yb.unsqueeze(-1)

        # — baseline forward —
        with torch.no_grad():
            y_hat, alphas, betas_all, _ = model(xb)
        y_true = yb[:, :, target_channel].cpu().numpy().ravel()
        y_pred = y_hat[:, :, target_channel].cpu().numpy().ravel()
        orig_losses.append(metric(y_true, y_pred))

        # — shuffle the forecast-time axis of betas_all —
        B, H, V, _ = betas_all.shape
        b = betas_all.squeeze(-1)          # [B, H, V]
        b_shuf = b.clone()
        for i in range(B):
            for v in range(V):
                perm = torch.randperm(H, device=device)
                b_shuf[i, :, v] = b[i, perm, v]
        b_shuf = b_shuf.unsqueeze(-1)      # [B, H, V, 1]

        # — rerun with shuffled betas —
        with torch.no_grad():
            y_hat_rand, _, _, _ = model.forward_with_alphas_betas(
                xb,
                alphas_override = alphas,
                betas_override  = b_shuf
            )
        y_pred_rand = y_hat_rand[:, :, target_channel].cpu().numpy().ravel()
        rand_losses.append(metric(y_true, y_pred_rand))

    # restore original
    model.forward_with_alphas_betas = orig_fw

    return float(np.mean(orig_losses)), float(np.mean(rand_losses))