from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchTST_Attention
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

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
        model = model_dict[self.args.model].Model(self.args).float()

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

    def plot_and_save_attention_map(self, attn, save_path, layer_idx, head_idx):
        if attn.dim() == 4:
            attn_map = attn[0, head_idx].detach().cpu().numpy()
        elif attn.dim() == 3:
            attn_map = attn[head_idx].detach().cpu().numpy()
        else:
            print("Unsupported attention shape.")
            return
    
        plt.figure(figsize=(6, 6))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'Layer {layer_idx} - Head {head_idx}')
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
 

    def save_attention(self, attn_tensor, folder_path, prefix, batch_idx):
        """
        Save attention weights and visualize for given batch.
        attn_tensor: torch.Tensor of shape [bs, n_heads, q_len, seq_len]
        folder_path: directory to save files
        prefix: string prefix for filenames
        batch_idx: batch index to name the files
        
        Saves .npy and .png files.
        """
        os.makedirs(folder_path, exist_ok=True)
        attn_np = attn_tensor.detach().cpu().numpy()  # [bs, n_heads, q_len, seq_len]
        
        # Save raw numpy array
        np.save(os.path.join(folder_path, f"{prefix}_attn_batch{batch_idx}.npy"), attn_np)
        
        bs, n_heads, q_len, seq_len = attn_np.shape
        for head_idx in range(n_heads):
            plt.figure(figsize=(8,6))
            # Visualize first sample in batch and this head
            plt.imshow(attn_np[0, head_idx], aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'{prefix} Attention - Batch {batch_idx} - Head {head_idx}')
            plt.xlabel('Key Sequence Length')
            plt.ylabel('Query Length')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"{prefix}_attn_batch{batch_idx}_head{head_idx}.png"))
            plt.close()




    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
    
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
    
        # Create folders for attention map saving
        folder_path = './train_results/' + setting + '/'
        attn_folder = os.path.join(folder_path, 'attn_maps/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(attn_folder):
            os.makedirs(attn_folder)
    
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
    
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
    
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
    
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
    
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
    
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
    
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
    
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    
                # forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
    
                # ATTENTION MAP SAVING (only in final epoch, every Nth batch)
                if (
                    'TST' in self.args.model
                    and epoch == self.args.train_epochs - 1   # Save only on final epoch
                    and i % 20 == 0                           # Every 20th batch (customize here)
                ):
                    attn_maps = None
                    try:
                        if isinstance(self.model, torch.nn.DataParallel):
                            attn_maps = self.model.module.get_attention_maps()
                        else:
                            attn_maps = self.model.get_attention_maps()
                    except Exception as e:
                        print(f"Warning: Cannot get attention maps: {e}")
                
                    if attn_maps is not None:
                        for key in ['attn', 'trend', 'res']:
                            if key in attn_maps:
                                for layer_idx, attn in enumerate(attn_maps[key]):
                                    if attn is not None:
                                        save_base = os.path.join(attn_folder, f'{key}_epoch_{epoch}_batch_{i}_layer_{layer_idx}')
                                        np.save(save_base + '.npy', attn.detach().cpu().numpy())
                
                                        # Optional: visualize first 2 heads
                                        num_heads = attn.shape[0] if attn.dim() == 3 else attn.shape[1]
                                        for head_idx in range(min(2, num_heads)):
                                            fig_path = save_base + f'_head_{head_idx}.png'
                                            self.plot_and_save_attention_map(attn, fig_path, layer_idx, head_idx)
                
                    # Also save corresponding input, target, and prediction for analysis
                    data_save_dir = os.path.join(attn_folder, f'data_epoch_{epoch}_batch_{i}')
                    os.makedirs(data_save_dir, exist_ok=True)
                
                    np.save(os.path.join(data_save_dir, 'batch_x.npy'), batch_x.detach().cpu().numpy())
                    np.save(os.path.join(data_save_dir, 'batch_y.npy'), batch_y.detach().cpu().numpy())
                    np.save(os.path.join(data_save_dir, 'outputs.npy'), outputs.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
    
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
    
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
    
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
    
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
    
        return self.model


    def test(self, setting, test=0):
            test_data, test_loader = self._get_data(flag='test')
            
            if test:
                print('loading model')
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    
            preds = []
            trues = []
            inputx = []
            folder_path = './test_results/' + setting + '/'
            attn_folder = os.path.join(folder_path, 'attn_maps/')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if not os.path.exists(attn_folder):
                os.makedirs(attn_folder)
    
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
    
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
    
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    # forward pass
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
                    # Save attention maps and data for interpretability (PatchTST only)
                    if 'TST' in self.args.model:
                        attn_maps = None
                        try:
                            if isinstance(self.model, torch.nn.DataParallel):
                                attn_maps = self.model.module.get_attention_maps()
                            else:
                                attn_maps = self.model.get_attention_maps()
                        except Exception as e:
                            print(f"Warning: Cannot get attention maps: {e}")
                    
                        if attn_maps is not None:
                            for key in ['attn', 'trend', 'res']:
                                if key in attn_maps:
                                    for layer_idx, attn in enumerate(attn_maps[key]):
                                        if attn is not None:
                                            try:
                                                save_base = os.path.join(attn_folder, f'{key}_batch_{i}_layer_{layer_idx}')
                                                np.save(save_base + '.npy', attn.detach().cpu().numpy())
                                            except Exception as e:
                                                print(f"Warning: Failed to save attention map for {save_base}: {e}")
                    
                                            # Optional: save visualizations
                                            num_heads = attn.shape[0] if attn.dim() == 3 else attn.shape[1]
                                            for head_idx in range(min(2, num_heads)):
                                                fig_path = save_base + f'_head_{head_idx}.png'
                                                self.plot_and_save_attention_map(attn, fig_path, layer_idx, head_idx)
                    
                            # Save corresponding inputs, outputs, and targets for this batch
                            data_save_dir = os.path.join(attn_folder, f'data_batch_{i}')
                            os.makedirs(data_save_dir, exist_ok=True)
                    
                            data_items = {
                                'batch_x.npy': batch_x,
                                'batch_y.npy': batch_y,
                                'outputs.npy': outputs,
                                'batch_x_mark.npy': batch_x_mark,
                                'batch_y_mark.npy': batch_y_mark
                            }
                    
                            for filename, tensor in data_items.items():
                                try:
                                    np.save(os.path.join(data_save_dir, filename), tensor.detach().cpu().numpy())
                                except Exception as e:
                                    print(f"Warning: Failed to save {filename} for batch {i}: {e}")

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
    
                    pred = outputs
                    true = batch_y
    
                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())
    
                    if i % 20 == 0:
                        input_arr = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input_arr[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_arr[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    
            # Other result saving and metrics unchanged
            # ...
            if self.args.test_flop:
                test_params_flop((batch_x.shape[1],batch_x.shape[2]))
                exit()
            preds = np.array(preds)
            trues = np.array(trues)
            inputx = np.array(inputx)
    
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
    
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
            f.close()
    
            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            # np.save(folder_path + 'x.npy', inputx)
            return
        # Helper function for attention map visualization
    def visualize_attention_map(self, attn_path, head=0):
        """
        Load and plot attention map from .npy file.
        
        Args:
            attn_path (str): path to saved attention numpy file
            head (int): which attention head to visualize
        """
        attn = np.load(attn_path)  # Expected shape: [num_heads, seq_len, seq_len]
        if head >= attn.shape[0]:
            print(f"Requested head {head} exceeds number of heads {attn.shape[0]}")
            return
        
        plt.figure(figsize=(6, 6))
        plt.imshow(attn[head], cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Map from {os.path.basename(attn_path)} Head {head}')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.show()
        
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        print("[Predict] Started prediction")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(best_model_path):
                print(f"Loading model from: {best_model_path}")
                assert os.path.exists(best_model_path), f"Checkpoint not found at {best_model_path}"
                self.model.load_state_dict(torch.load(best_model_path))
                print("[Predict] Model checkpoint loaded successfully")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")
    
        preds = []
        trues = []
    
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
    
                # Decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
    
                preds.append(pred)
                trues.append(true)
    
        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
    
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
    
        pred_path = os.path.join(folder_path, 'pred.npy')
        true_path = os.path.join(folder_path, 'true.npy')
    
        np.save(pred_path, preds)
        np.save(true_path, trues)
    
        print(f"Predictions saved to: {pred_path}")
        print(f"Ground truth saved to: {true_path}")
    
