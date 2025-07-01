import os
import subprocess

os.makedirs("./logs/LongForecasting", exist_ok=True)

seq_len = 336
model_name = "PatchTST_Attention"
root_path_name = "C:/Users/miche/Documents/PatchTST/PatchTST_supervised/dataset/"
data_path_name = "weather.csv"
model_id_name = "weather"
data_name = "custom"
random_seed = 2021
var_names_str = "p (mbar),T (degC),Tpot (K),Tdew (degC),rh (%),VPmax (mbar),VPact (mbar),VPdef (mbar),sh (g/kg)," \
                "H2OC (mmol/mol),rho (g/m**3),wv (m/s),max. wv (m/s),wd (deg),rain (mm),raining (s)," \
                "SWDR (W/m²),PAR (µmol/m²/s),max. PAR (µmol/m²/s),Tlog (degC),OT"
for pred_len in [192]: # in [96, 192, 336, 720]:
    cmd = [
        "python", "-u", "C:/Users/miche/Documents/PatchTST/PatchTST_supervised/run_longExp_attention.py",
        "--random_seed", str(random_seed),
        "--is_training", "1",
        "--root_path", root_path_name,
        "--data_path", data_path_name,
        "--model_id", f"{model_id_name}_{seq_len}_{pred_len}",
        "--model", model_name,
        "--data", data_name,
        "--features", "MS",
        "--target", "OT",
        "--c_out","1",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--individual", "1",
        "--enc_in", "21",
        "--dec_in","21" ,
        "--e_layers", "3",
        "--n_heads", "21",           # reduced heads
        "--d_model", "168",          # reduced model dimension
        "--d_ff", "672",            # reduced feedforward dim
        "--dropout", "0.2",
        "--fc_dropout", "0.2",
        "--head_dropout", "0",
        "--patch_len", "16",
        "--stride", "8",
        "--des", "Exp",
        "--train_epochs", "1",
        "--patience", "5",
        "--itr", "1",
        "--batch_size", "128",        # drastically smaller batch size
        "--learning_rate", "0.0001",
        "--num_workers", "8",
        "--output_attention",
    ]



    log_file = f"./logs/LongForecasting/{model_name}_{model_id_name}_{seq_len}_{pred_len}.log"
    print(f"Running experiment with pred_len={pred_len}, logging to {log_file} ...")

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')   # Print to console
            f.write(line)         # Write to log file

    print(f"Experiment for pred_len={pred_len} finished.\n")

print("All experiments completed.")
