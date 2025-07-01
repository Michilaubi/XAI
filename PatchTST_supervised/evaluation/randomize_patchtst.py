# utils_patchtst.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from layers.PatchTST_backbone import _ScaledDotProductAttention

# ─── stash original forward (when this module loads) ─────────────────────────
_orig_sdp_forward = _ScaledDotProductAttention.forward

def restore_original_attention():
    """Restore the un-patched SDP forward."""
    _ScaledDotProductAttention.forward = _orig_sdp_forward

def enable_attention_randomization():
    """Monkey-patch SDP.forward so it shuffles the scores before softmax."""
    def _randomized_sdp_forward(self, q, k, v, prev=None,
                                key_padding_mask=None,
                                attn_mask=None):
        scores = torch.matmul(q, k) * self.scale
        if self.res_attention and prev is not None:
            scores = scores + prev

        if attn_mask is not None:
            if attn_mask.dtype==torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask
        if key_padding_mask is not None:
            scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        B,H,Q_len,K_len = scores.shape
        flat = scores.view(-1, K_len)
        idx  = torch.randperm(K_len, device=flat.device)
        shuffled = flat[:, idx].view(B,H,Q_len,K_len)

        attn = F.softmax(shuffled, dim=-1)
        attn = self.attn_dropout(attn)
        out  = torch.matmul(attn, v)

        if self.res_attention:
            return out, attn, shuffled
        else:
            return out, attn

    _ScaledDotProductAttention.forward = _randomized_sdp_forward

def disable_attention_randomization():
    """Put back the original SDP.forward."""
    _ScaledDotProductAttention.forward = _orig_sdp_forward

def patchtst_randomization_check(model, loader, device,
                                 pred_len, temp_var_index=-1):
    """
    Runs two passes over `loader`:
     - original attention
     - randomized (shuffled) attention
    and returns (orig_mse, rand_mse).
    """
    model.eval()
    orig_losses, rand_losses = [], []

    # 1) original
    disable_attention_randomization()
    for xb, yb, *rest in tqdm(loader, desc="orig-attn"):
        Xb = xb.to(device).float()
        y_true = yb[:, -pred_len:, temp_var_index].cpu().numpy().ravel()
        with torch.no_grad():
            out = model(Xb)
        pred = out[:, :, temp_var_index].cpu().numpy().ravel()
        orig_losses.append(mean_squared_error(y_true, pred))

    # 2) randomized
    enable_attention_randomization()
    for xb, yb, *rest in tqdm(loader, desc="rand-attn"):
        Xb = xb.to(device).float()
        y_true = yb[:, -pred_len:, temp_var_index].cpu().numpy().ravel()
        with torch.no_grad():
            out = model(Xb)
        pred = out[:, :, temp_var_index].cpu().numpy().ravel()
        rand_losses.append(mean_squared_error(y_true, pred))

    disable_attention_randomization()
    return float(np.mean(orig_losses)), float(np.mean(rand_losses))
