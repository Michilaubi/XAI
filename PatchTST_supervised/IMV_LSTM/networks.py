import torch
from torch import nn
from torch import Tensor
from typing import List, Tuple, Optional


class IMVTensorLSTM(torch.jit.ScriptModule):
    
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units, device=x.device)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        
        return mean, alphas, betas

class IMVTensorMultiStepLSTM(torch.jit.ScriptModule):

    __constants__ = ["n_units", "input_dim", "forecast_steps"]

    def __init__(self, input_dim, output_dim, n_units, forecast_steps, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)

        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.input_projector = nn.Linear(output_dim, input_dim)  # For feeding predictions back in

        self.n_units = n_units
        self.input_dim = input_dim
        self.forecast_steps = forecast_steps

    @torch.jit.script_method
    def forward(self, x):
        batch_size = x.size(0)
        h_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        outputs = torch.jit.annotate(List[Tensor], [])

        # Encode input sequence
        for t in range(x.shape[1]):
            x_t = x[:, t, :].unsqueeze(1)
            j_tilda_t = torch.tanh(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                torch.einsum("bij,jik->bjk", x_t, self.U_j) + self.b_j
            )
            i_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                torch.einsum("bij,jik->bjk", x_t, self.U_i) + self.b_i
            )
            f_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                torch.einsum("bij,jik->bjk", x_t, self.U_f) + self.b_f
            )
            o_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                torch.einsum("bij,jik->bjk", x_t, self.U_o) + self.b_o
            )
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            outputs.append(h_tilda_t)

        outputs = torch.stack(outputs).permute(1, 0, 2, 3)
        alphas = torch.tanh(
            torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b
        )
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)

        predictions = torch.jit.annotate(List[Tensor], [])
        betas_list = torch.jit.annotate(List[Tensor], [])  # To store betas across steps

        prev_input = x[:, -1, :]

        # Autoregressive decoding
        for step in range(self.forecast_steps):
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.Phi(hg)
            betas = torch.tanh(self.F_beta(hg))
            betas = torch.exp(betas)
            betas = betas / torch.sum(betas, dim=1, keepdim=True)
            betas_list.append(betas)

            y_pred = torch.sum(betas * mu, dim=1)
            predictions.append(y_pred)

            prev_input = self.input_projector(y_pred).unsqueeze(1)

            # Decode step (same as above)
            j_tilda_t = torch.tanh(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                torch.einsum("bij,jik->bjk", prev_input, self.U_j) + self.b_j
            )
            i_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                torch.einsum("bij,jik->bjk", prev_input, self.U_i) + self.b_i
            )
            f_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                torch.einsum("bij,jik->bjk", prev_input, self.U_f) + self.b_f
            )
            o_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                torch.einsum("bij,jik->bjk", prev_input, self.U_o) + self.b_o
            )
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)

        forecasts = torch.stack(predictions, dim=1)
        betas_all = torch.stack(betas_list, dim=1)
        return forecasts, alphas, betas_all, outputs

    @torch.jit.script_method
    def forward_with_alphas_betas(
        self,
        x: Tensor,
        alphas_override: Optional[Tensor],
        betas_override: Optional[Tensor]
    ):
        """
        Same as forward(), but uses alphas_override and/or betas_override if provided.
        """
        batch_size = x.size(0)
        h_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        outputs = torch.jit.annotate(List[Tensor], [])

        # Encode input sequence (unchanged)
        for t in range(x.shape[1]):
            x_t = x[:, t, :].unsqueeze(1)
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + torch.einsum("bij,jik->bjk", x_t, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + torch.einsum("bij,jik->bjk", x_t, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + torch.einsum("bij,jik->bjk", x_t, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + torch.einsum("bij,jik->bjk", x_t, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            outputs.append(h_tilda_t)

        outputs = torch.stack(outputs).permute(1, 0, 2, 3)

        # Compute or override alphas
        if alphas_override is None:
            alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
            alphas = torch.exp(alphas)
            alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
            alphas = alphas.unsqueeze(-1)
        else:
            alphas = alphas_override

        g_n = torch.sum(alphas * outputs, dim=1)

        # Decoding loop with optional betas_override
        predictions = torch.jit.annotate(List[Tensor], [])
        betas_list  = torch.jit.annotate(List[Tensor], [])
        prev_input = x[:, -1, :]

        for step in range(self.forecast_steps):
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.Phi(hg)
            
            if betas_override is None:
                betas = torch.tanh(self.F_beta(hg))
                betas = torch.exp(betas)
                betas = betas / torch.sum(betas, dim=1, keepdim=True)
            else:
                betas = betas_override[:, step, :, :]
            betas_list.append(betas)

            y_pred = torch.sum(betas * mu, dim=1)
            predictions.append(y_pred)

            prev_input = self.input_projector(y_pred).unsqueeze(1)

            # Decode step LSTM cell (unchanged)
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + torch.einsum("bij,jik->bjk", prev_input, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + torch.einsum("bij,jik->bjk", prev_input, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + torch.einsum("bij,jik->bjk", prev_input, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + torch.einsum("bij,jik->bjk", prev_input, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)

        forecasts = torch.stack(predictions, dim=1)
        betas_all = torch.stack(betas_list, dim=1)
        return forecasts, alphas, betas_all, outputs



class DirectIMVTensorLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, forecast_horizon=1, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)

        # For beta attention and output
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)

        self.n_units = n_units
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon  # for labeling only

    @torch.jit.script_method
    def forward(self, x):
        batch_size = x.size(0)
        h_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        outputs = torch.jit.annotate(List[torch.Tensor], [])

        # Encoder
        for t in range(x.shape[1]):
            x_t = x[:, t, :].unsqueeze(1)  # [B, 1, D]
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", x_t, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            outputs.append(h_tilda_t)

        outputs = torch.stack(outputs).permute(1, 0, 2, 3)  # [B, T, D, H]

        # Attention: alpha
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)  # [B, T, D, 1]

        g_n = torch.sum(alphas * outputs, dim=1)  # [B, D, H]

        # Combine with final h_tilda_t for output
        hg = torch.cat([g_n, h_tilda_t], dim=2)  # [B, D, 2H]

        mu = self.Phi(hg)  # [B, D, output_dim]
        betas = torch.tanh(self.F_beta(hg))  # [B, D, 1]
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)  # [B, D, 1]

        y_pred = torch.sum(betas * mu, dim=1)  # [B, output_dim]

        return y_pred, alphas, betas, outputs

import torch
import torch.nn as nn
from typing import List

class IMVTensorMultiStepLSTM_Dropout(nn.Module):

    def __init__(self, input_dim, output_dim, n_units, forecast_steps, init_std=0.02, dropout_prob=0.2):
        super().__init__()

        # LSTM weight matrices
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)

        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)

        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)

        # Attention weights
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)

        # Output layers
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.input_projector = nn.Linear(output_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

        # Store constants
        self.n_units = n_units
        self.input_dim = input_dim
        self.forecast_steps = forecast_steps

    def forward(self, x):
        batch_size = x.size(0)
        h_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        outputs: List[torch.Tensor] = []

        # Encode input sequence
        for t in range(x.shape[1]):
            x_t = x[:, t, :].unsqueeze(1)

            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", x_t, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_o) + self.b_o)

            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            h_tilda_t = self.dropout(h_tilda_t)  # MC dropout

            outputs.append(h_tilda_t)

        outputs = torch.stack(outputs).permute(1, 0, 2, 3)  # [batch, time, input_dim, n_units]

        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)

        g_n = torch.sum(alphas * outputs, dim=1)

        predictions = []
        betas_list = []

        prev_input = x[:, -1, :]

        for step in range(self.forecast_steps):
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.dropout(self.Phi(hg))  # MC dropout here too

            betas = torch.tanh(self.F_beta(hg))
            betas = torch.exp(betas)
            betas = betas / torch.sum(betas, dim=1, keepdim=True)
            betas_list.append(betas)

            y_pred = torch.sum(betas * mu, dim=1)  # [batch, output_dim]
            predictions.append(y_pred)

            prev_input = self.input_projector(y_pred).unsqueeze(1)

            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", prev_input, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_o) + self.b_o)

            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            h_tilda_t = self.dropout(h_tilda_t)  # Apply dropout again during decoding

        forecasts = torch.stack(predictions, dim=1)     # [batch, forecast_steps, output_dim]
        betas_all = torch.stack(betas_list, dim=1)      # [batch, forecast_steps, ...]
        return forecasts, alphas, betas_all, outputs

class IMVTensorMultiStepLSTMConicity(torch.jit.ScriptModule):

    __constants__ = ["n_units", "input_dim", "forecast_steps"]

    def __init__(self, input_dim, output_dim, n_units, forecast_steps, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.input_projector = nn.Linear(output_dim, input_dim)

        self.n_units = n_units
        self.input_dim = input_dim
        self.forecast_steps = forecast_steps

    def compute_conicity(self, h_seq: torch.Tensor) -> torch.Tensor:
        h_bar = h_seq.mean(dim=0, keepdim=True)  # [1, n_units]
        h_bar_norm = h_bar / h_bar.norm(p=2, dim=1, keepdim=True)
        h_norm = h_seq / (h_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
        cosine_sims = torch.matmul(h_norm, h_bar_norm.t()).squeeze()
        return cosine_sims.mean()


    @torch.jit.script_method
    def forward(self, x):
        batch_size = x.size(0)
        h_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=x.device)
        outputs = torch.jit.annotate(List[torch.Tensor], [])

        for t in range(x.shape[1]):
            x_t = x[:, t, :].unsqueeze(1)
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", x_t, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            outputs.append(h_tilda_t)

        outputs_tensor = torch.stack(outputs).permute(1, 0, 2, 3)  # [batch, seq_len, input_dim, n_units]

        # Compute conicity per sample and input_dim
        conicity_scores = torch.zeros((batch_size, self.input_dim), device=x.device)
        for i in range(batch_size):
            for d in range(self.input_dim):
                h_seq = outputs_tensor[i, :, d, :]  # [seq_len, n_units]
                conicity_scores[i, d] = self.compute_conicity(h_seq)

        mean_conicity = conicity_scores.mean()

        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs_tensor, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs_tensor, dim=1)

        predictions = torch.jit.annotate(List[torch.Tensor], [])
        betas_list = torch.jit.annotate(List[torch.Tensor], [])

        prev_input = x[:, -1, :]

        for step in range(self.forecast_steps):
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.Phi(hg)
            betas = torch.tanh(self.F_beta(hg))
            betas = torch.exp(betas)
            betas = betas / torch.sum(betas, dim=1, keepdim=True)
            betas_list.append(betas)

            y_pred = torch.sum(betas * mu, dim=1)
            predictions.append(y_pred)

            prev_input = self.input_projector(y_pred).unsqueeze(1)

            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", prev_input, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)

        forecasts = torch.stack(predictions, dim=1)     # [batch, forecast_steps, output_dim]
        betas_all = torch.stack(betas_list, dim=1)      # [batch, forecast_steps, 1]

        return forecasts, alphas, betas_all, mean_conicity


class IMVEmbeddingLSTM(torch.jit.ScriptModule):

    __constants__ = ["n_units", "input_dim", "forecast_steps"]

    def __init__(self, input_dim, output_dim, n_units, forecast_steps,
                 cat1_num_classes=11, cat1_embed_dim=4,
                 cat2_num_classes=38, cat2_embed_dim=6,
                 init_std=0.02):
        super().__init__()

        self.cat1_embed = nn.Embedding(cat1_num_classes, cat1_embed_dim)
        self.cat2_embed = nn.Embedding(cat2_num_classes, cat2_embed_dim)

        # Total input dim increases due to embeddings
        self.total_input_dim = input_dim + cat1_embed_dim + cat2_embed_dim - 2

        self.U_j = nn.Parameter(torch.randn(self.total_input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(self.total_input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(self.total_input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(self.total_input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(self.total_input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(self.total_input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(self.total_input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(self.total_input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(self.total_input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(self.total_input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(self.total_input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(self.total_input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(self.total_input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(self.total_input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)

        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.input_projector = nn.Linear(output_dim, self.total_input_dim)

        self.n_units = n_units
        self.input_dim = input_dim
        self.forecast_steps = forecast_steps

    @torch.jit.script_method
    def forward(self, x, cat1_idx: Tensor, cat2_idx: Tensor):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Embed and repeat categorical features across sequence
        cat1_emb = self.cat1_embed(cat1_idx)  # Shape: [batch_size, seq_len, cat1_embed_dim]
        cat2_emb = self.cat2_embed(cat2_idx)  # Shape: [batch_size, seq_len, cat2_embed_dim]

        # Check dimensions again before repeating
        #print(f"Shape of cat1_emb after embedding: {cat1_emb.shape}")
        #print(f"Shape of cat2_emb after embedding: {cat2_emb.shape}")

        # Repeat embeddings across the sequence dimension
        cat1_emb = self.cat1_embed(cat1_idx)
        cat2_emb = self.cat2_embed(cat2_idx)

        #print(f"Shape of x: {x.shape}")
        #print(f"Shape of cat1_emb: {cat1_emb.shape}")
        #print(f"Shape of cat2_emb: {cat2_emb.shape}")

        # Concatenate embeddings to input
        x = torch.cat([x, cat1_emb, cat2_emb], dim=-1)
        
        h_tilda_t = torch.zeros(batch_size, self.total_input_dim, self.n_units, device=x.device)
        c_tilda_t = torch.zeros(batch_size, self.total_input_dim, self.n_units, device=x.device)
        outputs = torch.jit.annotate(List[Tensor], [])

        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(1)
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", x_t, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", x_t, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)
            outputs.append(h_tilda_t)

        outputs = torch.stack(outputs).permute(1, 0, 2, 3)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)

        predictions = torch.jit.annotate(List[Tensor], [])
        betas_list = torch.jit.annotate(List[Tensor], [])

        prev_input = x[:, -1, :].unsqueeze(1)

        for step in range(self.forecast_steps):
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.Phi(hg)
            betas = torch.tanh(self.F_beta(hg))
            betas = torch.exp(betas)
            betas = betas / torch.sum(betas, dim=1, keepdim=True)
            betas_list.append(betas)

            y_pred = torch.sum(betas * mu, dim=1)
            predictions.append(y_pred)

            prev_input = self.input_projector(y_pred).unsqueeze(1)

            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", prev_input, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", prev_input, self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)

        forecasts = torch.stack(predictions, dim=1)
        betas_all = torch.stack(betas_list, dim=1)
        return forecasts, alphas, betas_all

    
class IMVFullLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t*f_t + i_t*j_tilda_t.view(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t*torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        return mean, alphas, betas

def _estimate_alpha1(feature_reps, targets, device):

    X_T, X = feature_reps, feature_reps.permute(0,2,1)

    try:
        X_TX_inv = torch.linalg.inv(torch.bmm(X_T, X))
    except:
        X_TX_inv = torch.linalg.pinv(torch.bmm(X_T, X))



    X_Ty = torch.bmm(X_T, targets.unsqueeze(-1))

    # Compute likely scores
    alpha_hat = torch.bmm(X_TX_inv, X_Ty)
    return alpha_hat
    #return alpha_hat.squeeze(-1)  # shape (bs,  d) (NOT normalised)

class Delelstm(torch.jit.ScriptModule):
    def __init__(self, Delelstm_params,short):
        super(Delelstm, self).__init__()
        self.input_dim=Delelstm_params['input_dim']
        self.n_units=Delelstm_params['n_units']
        self.depth = Delelstm_params['time_depth']
        self.output_dim = Delelstm_params['output_dim']
        self.N_units = Delelstm_params['N_units']
        self.short=short
    
        #tensor LSTM parameter
        self.U_j = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_i = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_f = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_o = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.W_j = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_i = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_f = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_o = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.B_j = nn.Parameter(torch.randn(self.input_dim, self.n_units))
        self.B_i = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_f = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_o = nn.Parameter(torch.randn(self.input_dim, self.n_units))
  
        #vanilla LSTM parameter
        self.u_j = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.w_j = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_j = nn.Parameter(torch.zeros(self.N_units))
        self.w_i = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.u_i = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.b_i = nn.Parameter(torch.zeros(self.N_units))
        self.w_f = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.u_f = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.b_f = nn.Parameter(torch.zeros(self.N_units))
        self.w_o = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.u_o = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.b_o = nn.Parameter(torch.zeros(self.N_units))
        self.w_p = nn.Parameter(torch.randn(self.N_units, 1))
        self.b_p = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, x, device):

        H_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        C_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)


        h_tilda_t = torch.zeros(x.shape[0],  self.N_units).to(device)
        c_tilda_t = torch.zeros(x.shape[0],  self.N_units).to(device)
        unorm_list = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])


        for t in range(self.depth-1):
            # eq 1
            temp=H_tilda_t
            J_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.B_j)

            # eq 5
            I_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.B_i)
            F_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.B_f)
            O_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.B_o)
            # eq 6
            C_tilda_t = C_tilda_t * F_tilda_t + I_tilda_t * J_tilda_t
            # eq 7
            H_tilda_t = (O_tilda_t * torch.tanh(C_tilda_t)) #shape batch, feature, dim

            #normal lstm
            i_t = torch.sigmoid(((x[:, t, :] @ self.u_i) + (h_tilda_t @ self.w_i) + self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.u_f) + (h_tilda_t @ self.w_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.u_o) + (h_tilda_t @ self.w_o) + self.b_o))
            j_t = torch.tanh(((x[:, t, :] @ self.u_j) + (h_tilda_t @ self.w_j) + self.b_j))
            c_tilda_t = f_t * c_tilda_t + i_t * j_t
            h_tilda_t = o_t * torch.tanh(c_tilda_t)
            diff=H_tilda_t-temp
            if (t>self.short):
                newH_tilda_t=torch.concat([temp, diff],dim=1)
                unnorm_weight= _estimate_alpha1(newH_tilda_t, targets=h_tilda_t, device=device)
                h_tilda_t=torch.bmm(newH_tilda_t.permute(0,2,1),unnorm_weight).squeeze(-1)

            #predict
                pred_y=(h_tilda_t @ self.w_p) + self.b_p
                pred_list+=[pred_y]
                unorm_list += [unnorm_weight]
        pred = torch.stack(pred_list).permute(1,0,2)

        unorm = torch.stack(unorm_list) #shape time_depth, BATCH input_dim

        return pred, unorm,None
