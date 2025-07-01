#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# your imports
from IMV_LSTM.model_prep    import prepare_multistep_data
from IMV_LSTM.model_training import train_and_evaluate_IMV, evaluate_IMV_model
from IMV_LSTM.networks      import IMVTensorMultiStepLSTM

def parse_args():
    p = argparse.ArgumentParser(description="Train & eval your IMV‐LSTM multistep model")
    p.add_argument('--root_path',    type=str,   default='.',         help='root dir for data')
    p.add_argument('--data_path',    type=str,   default='data.csv', help='csv filename')
    p.add_argument('--input_window', type=int,   default=576,        help='history steps (10min each)')
    p.add_argument('--forecast_horizon', type=int, default=144,      help='pred steps (10min each)')
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--patience',     type=int,   default=25)
    p.add_argument('--save_dir',     type=str,   default='./imv_best', help='where to save model & attention')
    p.add_argument('--seed',         type=int,   default=2021)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load & prepare data
    df = pd.read_csv(os.path.join(args.root_path, args.data_path))
    cols_weather = ['p (mbar)', 'Tdew (degC)', 'sh (g/kg)', 
                    'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'T (degC)']
    target_col   = 'T (degC)'

    X_train, y_train, X_val, y_val, X_test, y_test, \
    in_scaler, out_scaler = prepare_multistep_data(
        df=df,
        input_columns=cols_weather,
        target_column=target_col,
        input_window=args.input_window,
        forecast_horizon=args.forecast_horizon,
        scale_data=True
    )

    # 2) Build DataLoaders
    def to_loader(X, y):
        tX = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(tX, ty)
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=(ds is not None))
    train_loader = to_loader(X_train, y_train)
    val_loader   = to_loader(X_val,   y_val)
    test_loader  = to_loader(X_test,  y_test)

    # 3) Instantiate model
    model = IMVTensorMultiStepLSTM(
        input_dim      = len(cols_weather),
        output_dim     = 1,
        n_units        = 140,
        forecast_steps = args.forecast_horizon
    ).to(device)

    # 4) Train + validate (saves best checkpoint)
    os.makedirs(args.save_dir, exist_ok=True)
    train_and_evaluate_IMV(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        target_scaler=out_scaler,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_attention_dir=os.path.join(args.save_dir, 'attention_val')
    )

    # 5) Final test‐set evaluation
    _, _, _, _, metrics = evaluate_IMV_model(
        model,
        test_loader,
        device,
        target_scaler=out_scaler,
        save_attention=True,
        save_dir=os.path.join(args.save_dir, 'attention_test'),
        print_metrics=True
    )
    print("Test metrics:", metrics)

if __name__ == '__main__':
    main()
