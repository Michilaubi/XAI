# model_training.py

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

def mse_loss(forecasts, targets):
    return F.mse_loss(forecasts, targets)

def compute_total_loss(forecasts, targets, alphas, mean_conicity=None,
                       lambda_entropy=0.01, lambda_conicity=0.0):
    loss_main = mse_loss(forecasts, targets)
    loss_reg = 0.0

    if lambda_entropy > 0:
        loss_reg += lambda_entropy * attention_entropy_regularization(alphas)

    if lambda_conicity > 0 and mean_conicity is not None:
        loss_reg += lambda_conicity * conicity_regularization(mean_conicity)

    return loss_main + loss_reg


def train_multistep_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    target_scaler=None
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, _, _ = model(X_batch)  # [batch, forecast_steps, 1]
            y_pred = y_pred.squeeze(-1)    # [batch, forecast_steps]
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_targets.append(y_batch.detach().cpu().numpy())

            loop.set_postfix(loss=loss.item())

        # Metrics
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)

        # Optional: inverse transform
        if target_scaler:
            all_train_preds = target_scaler.inverse_transform(all_train_preds)
            all_train_targets = target_scaler.inverse_transform(all_train_targets)

        train_mae = mean_absolute_error(all_train_targets, all_train_preds)
        train_rmse = np.sqrt(mean_squared_error(all_train_targets, all_train_preds))
        train_r2 = r2_score(all_train_targets, all_train_preds)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred, _, _ = model(X_batch)
                y_pred = y_pred.squeeze(-1)

                val_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)

                all_val_preds.append(y_pred.cpu().numpy())
                all_val_targets.append(y_batch.cpu().numpy())

        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_targets = np.concatenate(all_val_targets, axis=0)

        # Optional: inverse transform
        if target_scaler:
            all_val_preds = target_scaler.inverse_transform(all_val_preds)
            all_val_targets = target_scaler.inverse_transform(all_val_targets)

        val_mae = mean_absolute_error(all_val_targets, all_val_preds)
        val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_preds))
        val_r2 = r2_score(all_val_targets, all_val_preds)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
              f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
              f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")

    return model


def evaluate_IMV_model(
    model,
    data_loader: torch.utils.data.DataLoader,
    device,
    target_scaler=None,      # Ignored when using scaled data
    save_attention: bool = False,
    save_dir: str = "./attention_test",
    print_metrics: bool = True
):
    """
    Evaluate model on data_loader, returning predictions, ground truths, raw attentions,
    and a metrics dict using scaled data (no inverse transform).
    """
    model.eval()
    preds_list, trues_list = [], []
    alphas_list, betas_list = [], []

    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            y_pred, alphas, betas, _ = model(Xb)  # y_pred: [B, H, 1]
            preds_list.append(y_pred.cpu().numpy())
            trues_list.append(yb.cpu().numpy())
            alphas_list.append(alphas.cpu().numpy())
            betas_list.append(betas.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)  # (N, H, 1)
    trues = np.concatenate(trues_list, axis=0)  # (N, H)
    alphas = np.concatenate(alphas_list, axis=0)
    betas  = np.concatenate(betas_list, axis=0)

    # 1) Squeeze out last singleton dimension
    preds2 = preds.squeeze(-1)  # (N, H)
    # trues already (N, H)

    # 2) Skip inverse-scaling: use scaled data directly

    # 3) Save raw attention if requested
    if save_attention:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "alphas.npy"), alphas)
        np.save(os.path.join(save_dir, "betas.npy"),  betas)
        if print_metrics:
            print(f"✅ Saved attention maps to {save_dir}/{{alphas.npy,betas.npy}}")

    # 4) Compute metrics on scaled predictions
    y_pred_flat = preds2.reshape(-1)
    y_true_flat = trues.reshape(-1)

    mae  = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2   = r2_score(y_true_flat, y_pred_flat)

    if print_metrics:
        print(f"→ Eval: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    return preds2, trues, alphas, betas, {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_evaluate_IMV(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device,
    target_scaler,
    num_epochs: int = 100,
    lr: float = 3e-4,
    patience: int = 25,
    save_attention_dir: str = "./imv_attention"
):
    os.makedirs(save_attention_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_ctr = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        train_losses = []
        all_preds, all_trues = [], []

        print(f"\n=== Epoch {epoch}/{num_epochs} ===")

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            y_pred, _, _, _ = model(Xb)       # [B, H, 1]
            y_pred = y_pred.squeeze(-1)       # [B, H]

            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() * Xb.size(0))
            all_preds.append(y_pred.detach().cpu().numpy())
            all_trues.append(yb.detach().cpu().numpy())

        # Aggregate training metrics on scaled data
        avg_train_loss = sum(train_losses) / len(train_loader.dataset)
        preds_flat = np.concatenate(all_preds, axis=0).reshape(-1)
        trues_flat = np.concatenate(all_trues, axis=0).reshape(-1)

        train_mae  = mean_absolute_error(trues_flat, preds_flat)
        train_rmse = math.sqrt(mean_squared_error(trues_flat, preds_flat))
        train_r2   = r2_score(trues_flat, preds_flat)

        print(f"Train → Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, "
              f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

        # — Validation —
        _, _, _, _, val_metrics = evaluate_IMV_model(
            model, val_loader, device,
            target_scaler=None,  # use scaled metrics for consistency
            save_attention=False,
            print_metrics=True
        )
        val_rmse = val_metrics["RMSE"]

        # Early stopping + checkpoint
        if val_rmse < best_val_rmse * 0.995:
            best_val_rmse = val_rmse
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
            }, os.path.join(save_attention_dir, "checkpoint_latest.pth"))
            print("🎉 New best model saved.")
        else:
            patience_ctr += 1
            print(f"No improvement → patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print("⏱ Early stopping.")
                break

    # Load best and final test eval on scaled data
    best_path = os.path.join(save_attention_dir, "checkpoint_latest.pth")
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state'])
    return evaluate_IMV_model(
        model,
        test_loader,
        device,
        target_scaler=None,
        save_attention=True,
        save_dir=save_attention_dir,
        print_metrics=True
    )




