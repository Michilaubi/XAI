import os
import subprocess

# ensure log directory exists
os.makedirs("./logs/LongForecasting", exist_ok=True)

seq_len = 288
model_name = "PatchTST_Attention"
root_path_name = "C:/Users/miche/Documents/PatchTST/PatchTST_supervised/dataset/"
data_path_name = "weather_int.csv"
model_id_name = "weather_int"
data_name = "custom"
random_seed = 2021

# change horizon to 10 steps
for pred_len in [72]:
    cmd = [
        "python", "-u", "C:/Users/miche/Documents/PatchTST/PatchTST_supervised/run_longExp_attention.py",
        "--random_seed", str(random_seed),
        "--is_training", "1",
        "--do_predict",
        "--label_len", "72",                
        "--root_path", root_path_name,
        "--data_path", data_path_name,
        "--model_id", f"{model_id_name}_{seq_len}_{pred_len}",
        "--model", model_name,
        "--data", data_name,
        "--features", "MS",
        "--freq", "10T",
        "--target", "T (degC)",
        "--c_out", "1",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--individual", "1",
        "--enc_in", "7",
        "--e_layers", "3",
        "--n_heads", "7",
        "--d_model", "140",
        "--d_ff", "280",
        "--dropout", "0.2",
        "--fc_dropout", "0.2",
        "--head_dropout", "0",
        "--patch_len", "24",
        "--stride", "12",
        "--des", "Exp",
        "--use_amp",
        "--train_epochs", "100",
        "--patience", "14",
        "--itr", "1",
        "--batch_size", "128",
        "--learning_rate", "0.0003",
        "--num_workers", "4",
        "--output_attention",
    ]

    log_file = f"./logs/LongForecasting/{model_name}_{model_id_name}_{seq_len}_{pred_len}.log"
    print(f"Running experiment with pred_len={pred_len}, logging to {log_file} ...")

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')   # print to console
            f.write(line)         # write to log file

    print(f"Experiment for pred_len={pred_len} finished.\n")

print("All experiments completed.")
