__all__ = ['PatchTST_Attention']

from typing import Callable, Optional
import torch
from torch import nn, Tensor
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = True, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # model setup
        self.decomposition = decomposition
        self.var_names = getattr(configs, 'var_names', [f'var_{i}' for i in range(c_in)])

        backbone_args = dict(
            c_in=c_in, context_window=context_window, target_window=target_window,
            patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, n_layers=n_layers,
            d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask,
            padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
            store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
            head_dropout=head_dropout, padding_patch=padding_patch, pretrain_head=pretrain_head,
            head_type=head_type, individual=individual, revin=revin, affine=affine,
            subtract_last=subtract_last, verbose=verbose, **kwargs
        )

        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(**backbone_args)
            self.model_res = PatchTST_backbone(**backbone_args)
        else:
            self.model = PatchTST_backbone(**backbone_args)

    def forward(self, x, return_attn: bool = False):
        """
        Args:
            x: [Batch, Input length, Channel]
            return_attn (bool): if True, also return attention weights
        Returns:
            x: [Batch, Output length, Channel]
            (optional) attn_weights
        """
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
    
            if return_attn:
                res, res_attn = self.model_res(res_init, return_attn=True)
                trend, trend_attn = self.model_trend(trend_init, return_attn=True)
            else:
                res = self.model_res(res_init)
                trend = self.model_trend(trend_init)
    
            x_out = (res + trend).permute(0, 2, 1)
    
            if return_attn:
                return x_out, {'res': res_attn, 'trend': trend_attn}
        else:
            x_in = x.permute(0, 2, 1)
            if return_attn:
                x_out, attn = self.model(x_in, return_attn=True)
                x_out = x_out.permute(0, 2, 1)
                return x_out, {'attn': attn}
            else:
                out = self.model(x_in)
                if isinstance(out, tuple):  # Safe unpacking
                    x_out, _ = out
                else:
                    x_out = out
                x_out = x_out.permute(0, 2, 1)

    
        return x_out
        
    def get_attn_weights(self):
        """
        Returns attention weights stored during the forward pass.
        Make sure `store_attn=True` during model creation and a forward pass has occurred.
        """
        if self.decomposition:
            trend_attn = [attn for attn in self.model_trend.backbone.attn_layers()]
            res_attn = [attn for attn in self.model_res.backbone.attn_layers()]
            return {
                'trend': trend_attn,
                'res': res_attn,
                'vars': self.var_names  # Add this
            }
        else:
            attn = [attn for attn in self.model.backbone.attn_layers()]
            return {
                'attn': attn,
                'vars': self.var_names  # Add this
            }

    def get_attention_maps(self):
        """
        Returns stored attention maps (after a forward pass with `store_attn=True`).
        """
        if self.decomposition:
            trend_attn = self.model_trend.get_attention_maps()
            res_attn = self.model_res.get_attention_maps()
            return {
                'trend': trend_attn,
                'res': res_attn,
                'vars': self.var_names
            }
        else:
            attn = self.model.get_attention_maps()
            return {
                'attn': attn,
                'vars': self.var_names
            }
    
