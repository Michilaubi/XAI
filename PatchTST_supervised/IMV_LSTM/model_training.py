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



def plot_training_diagnostics(train_entropy, val_entropy, train_conicity=None, val_conicity=None):
    plt.figure(figsize=(12, 4))

    # Entropy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_entropy, label='Train Entropy')
    plt.plot(val_entropy, label='Val Entropy')
    plt.title('Attention Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.legend()

    # Conicity Plot (optional)
    if train_conicity and val_conicity:
        plt.subplot(1, 2, 2)
        plt.plot(train_conicity, label='Train Conicity')
        plt.plot(val_conicity, label='Val Conicity')
        plt.title('Attention Conicity')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Conicity')
        plt.legend()

    plt.tight_layout()
    plt.show()

def train_multistep_model_reg(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    target_scaler=None,
    loss_fn=None,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    train_entropy_log = []
    val_entropy_log = []
  #  train_conicity_log = []
  #  val_conicity_log = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        epoch_entropy = []
       # epoch_conicity = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, alphas, betas_all, outputs = model(X_batch)  # y_pred: [batch, steps, 1], alphas: [batch, heads, seq_len]
            y_pred = y_pred.squeeze(-1)         # [batch, steps]

            # Use the loss_fn you passed, which should support attention regularization if you want
            loss = loss_fn(y_pred, y_batch, attn_weights=alphas) if 'attn_weights' in loss_fn.__code__.co_varnames else loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_targets.append(y_batch.detach().cpu().numpy())

            # Calculate attention stats without affecting gradients
            with torch.no_grad():
                entropy = attention_entropy_regularization(alphas).item()
                epoch_entropy.append(entropy)

       #         conicity = compute_mean_conicity(alphas.mean(dim=2))
       #         epoch_conicity.append(conicity)

            loop.set_postfix(loss=loss.item())

        train_entropy_log.append(np.mean(epoch_entropy))
       # train_conicity_log.append(np.mean(epoch_conicity))

        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)

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
        val_entropy = []
       # val_conicity = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred, alphas, betas_all, outputs = model(X_batch)  # new unpacking
                y_pred = y_pred.squeeze(-1)

                val_loss += loss_fn(y_pred, y_batch).item() * X_batch.size(0)
                all_val_preds.append(y_pred.cpu().numpy())
                all_val_targets.append(y_batch.cpu().numpy())

                entropy = attention_entropy_regularization(alphas).item()
                val_entropy.append(entropy)

       #         conicity = compute_mean_conicity(alphas)
       #         val_conicity.append(conicity)

        val_entropy_log.append(np.mean(val_entropy))
       # val_conicity_log.append(np.mean(val_conicity))

        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_targets = np.concatenate(all_val_targets, axis=0)

        if target_scaler:
            all_val_preds = target_scaler.inverse_transform(all_val_preds)
            all_val_targets = target_scaler.inverse_transform(all_val_targets)

        val_mae = mean_absolute_error(all_val_targets, all_val_preds)
        val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_preds))
        val_r2 = r2_score(all_val_targets, all_val_preds)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
            f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
            f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
            f"Train Entropy: {train_entropy_log[-1]:.4f} | Val Entropy: {val_entropy_log[-1]:.4f}"
        )

    plot_training_diagnostics(train_entropy_log, val_entropy_log)
                              #train_conicity_log, val_conicity_log

    return model

def train_direct_model(model, train_loader, val_loader, num_epochs, learning_rate, device, loss_fn):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, alphas, betas, _ = model(X_batch)
            loss = loss_fn(y_pred.squeeze(-1), y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                y_pred, _, _, _ = model(X_val)
                val_loss += loss_fn(y_pred.squeeze(-1), y_val.squeeze(-1)).item() * X_val.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader.dataset):.4f} - "
              f"Val Loss: {val_loss/len(val_loader.dataset):.4f}")

    return model
    
def train_all_direct_models(X_train, Y_train, X_val, Y_val, forecast_steps, seq_len, input_dim, output_dim,
                            n_units, batch_size, num_epochs, learning_rate, device):
    models = []

    for horizon in range(1, forecast_steps + 1):
        print(f"\n=== Training model for horizon t+{horizon} ===")
        train_loader = create_direct_dataloader(X_train, Y_train, horizon, seq_len, batch_size)
        val_loader = create_direct_dataloader(X_val, Y_val, horizon, seq_len, batch_size)

        model = DirectIMVTensorLSTM(input_dim, output_dim, n_units, forecast_horizon=horizon)
        model = train_direct_model(model, train_loader, val_loader, num_epochs, learning_rate, device, loss_fn=nn.MSELoss())
        models.append(model)

    return models
def predict_with_all_models(models, X_seq, device):
    """
    X_seq: [batch_size, seq_len, input_dim]
    Returns: [batch_size, forecast_steps, output_dim]
    """
    X_seq = X_seq.to(device)
    preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            y_pred, _, _, _ = model(X_seq)
            preds.append(y_pred.unsqueeze(1))  # [batch, 1, output_dim]

    forecasts = torch.cat(preds, dim=1)  # [batch_size, forecast_steps, output_dim]
    return forecasts


def train_dropout_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    mc_samples: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses, train_mae, train_rmse = [], [], []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            y_pred_mean, _, _, _ = model(xb)  # Single forward pass
            loss = criterion(y_pred_mean, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_mae.append(torch.mean(torch.abs(y_pred_mean - yb)).item())
            train_rmse.append(torch.sqrt(torch.mean((y_pred_mean - yb) ** 2)).item())

        # Validation (with MC Dropout)
        model.train()  # keep dropout active
        val_losses, val_mae, val_rmse, val_entropy = [], [], [], []

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]"):
                xb, yb = xb.to(device), yb.to(device)
                preds = []

                for _ in range(mc_samples):
                    pred_sample, _, _, _ = model(xb)
                    preds.append(pred_sample)

                preds = torch.stack(preds)  # [mc_samples, batch, steps, output_dim]
                mean_pred = preds.mean(0)
                std_pred = preds.std(0)  # Predictive std for uncertainty

                loss = criterion(mean_pred, yb)
                val_losses.append(loss.item())
                val_mae.append(torch.mean(torch.abs(mean_pred - yb)).item())
                val_rmse.append(torch.sqrt(torch.mean((mean_pred - yb) ** 2)).item())

                # Entropy approximation (log std as uncertainty proxy)
                entropy = torch.mean(torch.log(std_pred + 1e-6)).item()
                val_entropy.append(entropy)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f} | "
            f"Train MAE: {np.mean(train_mae):.4f} | Val MAE: {np.mean(val_mae):.4f} | "
            f"Train RMSE: {np.mean(train_rmse):.4f} | Val RMSE: {np.mean(val_rmse):.4f} | "
            f"Val Entropy: {np.mean(val_entropy):.4f}"
        )


def train_concocity_model(
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
        train_conicity_total = 0.0
        all_train_preds = []
        all_train_targets = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, _, _, conicity = model(X_batch)  # Get conicity from model output
            y_pred = y_pred.squeeze(-1)              # [batch, forecast_steps]
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_conicity_total += conicity.item() * X_batch.size(0)

            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_targets.append(y_batch.detach().cpu().numpy())

            loop.set_postfix(loss=loss.item(), conicity=conicity.item())

        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)

        if target_scaler:
            all_train_preds = target_scaler.inverse_transform(all_train_preds)
            all_train_targets = target_scaler.inverse_transform(all_train_targets)

        train_mae = mean_absolute_error(all_train_targets, all_train_preds)
        train_rmse = np.sqrt(mean_squared_error(all_train_targets, all_train_preds))
        train_r2 = r2_score(all_train_targets, all_train_preds)
        train_loss /= len(train_loader.dataset)
        avg_train_conicity = train_conicity_total / len(train_loader.dataset)

        # -------------------- Validation -------------------- #
        model.eval()
        val_loss = 0.0
        val_conicity_total = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred, _, _, conicity = model(X_batch)
                y_pred = y_pred.squeeze(-1)

                val_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)
                val_conicity_total += conicity.item() * X_batch.size(0)

                all_val_preds.append(y_pred.cpu().numpy())
                all_val_targets.append(y_batch.cpu().numpy())

        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_targets = np.concatenate(all_val_targets, axis=0)

        if target_scaler:
            all_val_preds = target_scaler.inverse_transform(all_val_preds)
            all_val_targets = target_scaler.inverse_transform(all_val_targets)

        val_mae = mean_absolute_error(all_val_targets, all_val_preds)
        val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_preds))
        val_r2 = r2_score(all_val_targets, all_val_preds)
        val_loss /= len(val_loader.dataset)
        avg_val_conicity = val_conicity_total / len(val_loader.dataset)

        # -------------------- Logging -------------------- #
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
              f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
              f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
              f"Train Conicity: {avg_train_conicity:.4f} | Val Conicity: {avg_val_conicity:.4f}"
        )

    return model

def train_model_single_step(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs,
    patience,
    save_path,
    device,
    y_min=None,
    y_max=None,
    X_train_len=None,
    X_val_len=None,
    plot_interval=10
):
    best_val_rmse = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        start_time = time.monotonic()
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred, _, _ = model(X_batch)
            y_pred = y_pred.squeeze(1)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        preds = []
        targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred, _, _ = model(X_batch)
                y_pred = y_pred.squeeze(1)

                preds.append(y_pred.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
                val_loss += loss_fn(y_pred, y_batch).item() * X_batch.size(0)

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        train_rmse = (train_loss / X_train_len)**0.5
        val_rmse = (val_loss / X_val_len)**0.5

        print(f"Epoch {epoch} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            print("Model saved.")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

        if epoch % plot_interval == 0 and y_min is not None and y_max is not None:
            preds_rescaled = preds * (y_max - y_min) + y_min
            targets_rescaled = targets * (y_max - y_min) + y_min

            mse = mean_squared_error(targets_rescaled, preds_rescaled)
            mae = mean_absolute_error(targets_rescaled, preds_rescaled)

            print(f"Epoch {epoch} | MSE: {mse:.4f} | MAE: {mae:.4f}")

            plt.figure(figsize=(20, 10))
            plt.plot(preds_rescaled, label="Predicted")
            plt.plot(targets_rescaled, label="Actual")
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.show()

        end_time = time.monotonic()
        print(f"Epoch Time: {end_time - start_time:.2f}s")
