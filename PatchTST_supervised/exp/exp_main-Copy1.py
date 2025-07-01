from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchTST_Attention
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

# You need to provide or import your EarlyStopping implementation:
# from utils import EarlyStopping, adjust_learning_rate, metric, visual, data_provider

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTST_Attention': PatchTST_Attention,
        }
        model = model_dict[self.args.model].Model(self.args).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _plot_feature_summary(self, attn_tensor, batch_idx, folder_path):
        """
        Plot average attention per feature across all heads and layers.
        attn_tensor shape: [L, B, H, F, Q, K]
        """
        L, B, H, F, Q, K = attn_tensor.shape

        if H != F:
            raise ValueError(f"Expected H == F (one head per feature), but got H={H}, F={F}")

        # Mean attention per feature over layers, batches, heads, query, key dims
        feature_importance = attn_tensor.mean(axis=(0, 1, 2, 4, 5))  # shape: (F,)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(F), feature_importance)
        ax.set_title(f'Feature Attention Summary - Batch {batch_idx}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Avg Attention Weight')
        ax.set_xticks(np.arange(F))
        plt.tight_layout()
        save_path = os.path.join(folder_path, f'feature_importance_batch_{batch_idx}.png')
        plt.savefig(save_path)
        plt.close()

    def _save_temporal_attention(self, attn_tensor, batch_idx, folder_path, max_features=5):
        """
        Save temporal attention (Q x K) for each feature.
        attn_tensor shape: [L, B, H, F, Q, K]
        Assumes one head per feature (H == F).
        """
        os.makedirs(folder_path, exist_ok=True)

        L, B, H, F, Q, K = attn_tensor.shape

        if H != F:
            print(f"[Warning] Expected H == F (1 head per feature), but got H={H}, F={F}")
            return

        if Q != K:
            print(f"[Warning] Q ({Q}) != K ({K}), heatmap may be non-square.")

        for f in range(min(F, max_features)):
            try:
                temporal_attn = attn_tensor[:, :, f, f, :, :]  # shape: (L, B, Q, K)
                mean_temporal_attn = temporal_attn.mean(axis=(0, 1))  # mean over layers and batch -> (Q, K)

                # Save raw numpy
                npy_path = os.path.join(folder_path, f"temporal_attn_feature_{f}_batch_{batch_idx}.npy")
                np.save(npy_path, mean_temporal_attn)

                # Save heatmap
                plt.figure(figsize=(6, 5))
                plt.imshow(mean_temporal_attn, aspect='auto', cmap='viridis')
                plt.title(f'Temporal Attention - Feature {f}')
                plt.xlabel('Key Time Step')
                plt.ylabel('Query Time Step')
                plt.colorbar(label='Attention Weight')
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, f"temporal_attn_feature_{f}_batch_{batch_idx}.png"))
                plt.close()

            except Exception as e:
                print(f"[Warning] Failed to save temporal attention for feature {f}: {e}")

    def _save_attention_weights(self, attns, batch_idx, folder_path, plot=True, summary=True):
        """
        Save attention weights to disk, optionally plot feature summaries.
        attns: dict of attention tensors or lists
        """
        os.makedirs(folder_path, exist_ok=True)
        attn_arrays = {}

        for name, value in attns.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    key = f"{name}_layer_{i}"
                    if torch.is_tensor(v):
                        attn_arrays[key] = v.detach().cpu().numpy()
                    else:
                        attn_arrays[key] = np.array(v)
            elif torch.is_tensor(value):
                attn_arrays[name] = value.detach().cpu().numpy()
            else:
                try:
                    attn_arrays[name] = np.array(value)
                except Exception as e:
                    print(f"[Warning] Could not convert {name} to NumPy array: {e}")

        save_path = os.path.join(folder_path, f'attention_batch_{batch_idx}.npz')
        np.savez_compressed(save_path, **attn_arrays)
        print(f"[Saved] Attention weights -> {save_path}")

        if plot and 'attn_layer_0' in attn_arrays and summary:
            try:
                stacked = np.stack([attn_arrays[k] for k in attn_arrays if k.startswith('attn_layer_')])
                attn_tensor = torch.tensor(stacked)
                self._plot_feature_summary(attn_tensor, batch_idx, folder_path)
                self._save_temporal_attention(attn_tensor=attn_tensor, batch_idx=batch_idx, folder_path=folder_path)
            except Exception as e:
                print(f"[Warning] Failed to process attention visualizations: {e}")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        def patchify(x, patch_len=16, stride=8):
            """
            Convert input tensor into patches.
            Input x: Tensor shape (B, T, F)
            Returns: (B, F, P, patch_len)
            """
            x_np = x.detach().cpu().numpy()
            B, T, F = x_np.shape
            patches_per_seq = (T - patch_len) // stride + 1
            x_patches = []
            for b in range(B):
                patches = []
                for start in range(0, T - patch_len + 1, stride):
                    patch = x_np[b, start:start + patch_len, :]
                    patches.append(patch)
                x_patches.append(np.stack(patches, axis=0))  # shape: (P, patch_len, F)
            x_patches = np.stack(x_patches, axis=0)  # shape: (B, P, patch_len, F)
            x_patches = np.transpose(x_patches, (0, 3, 1, 2))  # (B, F, P, patch_len)
            return x_patches

        feature_energy_accum = None
        feature_attn_accum = None
        all_epoch_scores = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    if isinstance(outputs, tuple):
                        pred, attns = outputs[0], outputs[1]
                    else:
                        pred, attns = outputs, None

                    pred = pred.float()
                    loss = criterion(pred, batch_y)
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if scheduler:
                    scheduler.step()

                train_loss.append(loss.item())

                # Save attention weights per batch for analysis (optional, toggle as needed)
                if attns is not None and i % 50 == 0:
                    save_folder = os.path.join(path, f"attention_epoch_{epoch}_batch_{i}")
                    self._save_attention_weights(attns, batch_idx=i, folder_path=save_folder)

                if iter_count % 10 == 0:
                    print(f"Epoch: {epoch+1}/{self.args.train_epochs} | Step: {iter_count}/{train_steps} | Loss: {loss.item():.6f}")

            print(f"Epoch {epoch+1} finished. Avg Loss: {np.mean(train_loss):.6f}, Time: {time.time() - epoch_time:.2f}s")

            vali_loss = self.vali(vali_loader, criterion)
            test_loss, test_metrics = self.test(test_loader, criterion)

            print(f"Validation Loss: {vali_loss:.6f}")
            print(f"Test Loss: {test_loss:.6f} | Test Metrics: {test_metrics}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print(f"Training complete in {(time.time() - time_now) / 60:.2f} minutes.")
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs

                loss = criterion(pred, batch_y)
                total_loss += loss.item()

        return total_loss / len(vali_loader)

    def test(self, test_loader, criterion):
        self.model.eval()
        preds = []
        trues = []

        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs

                loss = criterion(pred, batch_y)
                total_loss += loss.item()

                preds.append(pred.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Calculate evaluation metrics - provide your own metric function or module
        test_metrics = metric(preds, trues)

        avg_loss = total_loss / len(test_loader)
        return avg_loss, test_metrics

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.model == 'PatchTST_Attention' and self.args.output_attention:
                    outputs, attns = self.model(batch_x, return_attn=True)
                else:
                    outputs = self.model(batch_x)
                    attns = None

                preds.append(outputs.detach().cpu().numpy())

                if attns is not None:
                    self._save_attention_weights(attns, i, folder_path)

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])

        result_path = './results/' + setting + '/'
        os.makedirs(result_path, exist_ok=True)

        np.save(result_path + 'real_prediction.npy', preds)
        return
