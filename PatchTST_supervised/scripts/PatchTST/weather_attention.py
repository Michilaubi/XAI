import os
import subprocess

# Make sure logs directories exist
os.makedirs("./logs/LongForecastingAttention", exist_ok=True)

seq_len = 336
model_name = "PatchTST"

root_path_name = "./dataset/"
data_path_name = "weather.csv"
model_id_name = "weather"
data_name = "custom"
random_seed = 2021

for pred_len in [96, 192, 336, 720]:
    cmd = [
        "python", "-u", "run_longExp_attention.py",
        "--random_seed", str(random_seed),
        "--is_training", "1",
        "--root_path", root_path_name,
        "--data_path", data_path_name,
        "--model_id", f"{model_id_name}_{seq_len}_{pred_len}",
        "--model", model_name,
        "--data", data_name,
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--enc_in", "21",
        "--e_layers", "3",
        "--n_heads", "16",
        "--d_model", "128",
        "--d_ff", "256",
        "--dropout", "0.2",
        "--fc_dropout", "0.2",
        "--head_dropout", "0",
        "--patch_len", "16",
        "--stride", "8",
        "--des", "Exp",
        "--train_epochs", "20",
        "--patience", "20",
        "--itr", "1",
        "--batch_size", "128",
        "--learning_rate", "0.0001"
    ]
    log_file = f"./logs/LongForecastingAttention/{model_name}_{model_id_name}_{seq_len}_{pred_len}.log"
    
    print(f"Running experiment with pred_len={pred_len}, logging to {log_file} ...")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()  # wait for the process to complete

    print(f"Experiment for pred_len={pred_len} finished.")

print("All experiments completed.")
