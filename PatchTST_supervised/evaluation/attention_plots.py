import os
import numpy as np
import matplotlib.pyplot as plt

def plot_patchtst_temporal(
    attn_dir,
    num_layers,
    num_batches,
    batch_size,
    num_vars,
    num_heads,
    seq_len,              # <-- renamed from input_steps
    feature_names,
    model_name="PatchTST",
    run_name="default"
):
    """
    attn_dir: path to your .npy attention dumps
    seq_len: length of the input sequence used at training
    (other args as before)
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    temporal_acc = np.zeros((num_vars, seq_len), dtype=float)
    head_acc     = np.zeros((num_heads, seq_len), dtype=float)
    files = 0

    for b in range(num_batches):
        for layer in range(num_layers):
            fn = os.path.join(attn_dir, f"attn_batch_{b}_layer_{layer}.npy")
            if not os.path.isfile(fn): 
                continue
            A = np.load(fn)   # shape (B*V, H, Q, K)
            if A.size == 0: 
                continue
            try:
                A = A.reshape(batch_size, num_vars, num_heads, seq_len, seq_len)
            except ValueError:
                continue

            # avg over batch & queries → [V x K]
            temporal_acc += A.mean(axis=(0,2,3))
            # avg over batch, vars & queries → [H x K]
            head_acc     += A.mean(axis=(0,1,3))
            files += 1

    if files == 0:
        raise RuntimeError("No attention files found!")

    temporal_avg = temporal_acc / files
    head_avg     = head_acc     / files

    # --- Temporal per-feature heatmap ---
    plt.figure(figsize=(10, 5))
    im = plt.imshow(temporal_avg, aspect='auto', cmap='viridis')
    plt.colorbar(im, label="Avg attention weight")
    plt.yticks(np.arange(num_vars), feature_names, fontsize=10)
    plt.xticks(
        np.linspace(0, seq_len-1, 6),
        [f"-{i:d}" for i in np.linspace(seq_len, 0, 6, dtype=int)]
    )
    plt.xlabel("Input time step (earlier → later)")
    plt.title(f"{model_name} ({run_name}) ▶ Temporal attention per feature\n(seq_len={seq_len})")
    plt.tight_layout()
    savepath1 = os.path.join(out_dir, f"{model_name}_{run_name}_temporal_per_feature.png")
    plt.savefig(savepath1, dpi=300)
    plt.show()

    # --- Head-wise temporal heatmap ---
    plt.figure(figsize=(10, 4))
    im = plt.imshow(head_avg, aspect='auto', cmap='plasma')
    plt.colorbar(im, label="Avg attention weight")
    plt.yticks(
        np.arange(num_heads),
        [f"Head {h+1}" for h in range(num_heads)],
        fontsize=9
    )
    plt.xticks(
        np.linspace(0, seq_len-1, 6),
        [f"-{i:d}" for i in np.linspace(seq_len, 0, 6, dtype=int)]
    )
    plt.xlabel("Input time step")
    plt.title(f"{model_name} ({run_name}) ▶ Head-wise temporal attention\n(seq_len={seq_len})")
    plt.tight_layout()
    savepath2 = os.path.join(out_dir, f"{model_name}_{run_name}_headwise_temporal.png")
    plt.savefig(savepath2, dpi=300)
    plt.show()


def plot_imv_saved_attention(
    attn_dir,
    seq_len,
    pred_len,             # <-- new argument
    feature_names,
    model_name="IMV-LSTM",
    run_name="default"
):
    """
    attn_dir: path to your alpha/beta .npy files  
    seq_len: length of the input sequence  
    pred_len: forecasting horizon  
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # collect alpha files
    alpha_files = sorted(
        f for f in os.listdir(attn_dir)
        if f.lower().endswith(".npy") and "alpha" in f.lower()
    )
    # collect beta files
    beta_files = sorted(
        f for f in os.listdir(attn_dir)
        if f.lower().endswith(".npy") and "beta" in f.lower()
    )

    # accumulate alphas
    sum_alpha = None
    count_a   = 0
    for fn in alpha_files:
        arr = np.load(os.path.join(attn_dir, fn))
        # unify to [B, T, V]
        if arr.ndim == 4:
            arr = arr[..., 0]
        B, T, V = arr.shape
        sum_alpha = arr.sum(axis=0) if sum_alpha is None else sum_alpha + arr.sum(axis=0)
        count_a += B
    if count_a == 0:
        raise RuntimeError("No alpha files found!")
    avg_alpha = sum_alpha / count_a   # shape [T, V]

    # accumulate betas
    sum_beta = None
    count_b  = 0
    for fn in beta_files:
        arr = np.load(os.path.join(attn_dir, fn))
        if arr.ndim == 4:
            arr = arr[..., 0]
        B, H, V = arr.shape
        sum_beta = arr.sum(axis=0) if sum_beta is None else sum_beta + arr.sum(axis=0)
        count_b += B
    if count_b == 0:
        raise RuntimeError("No beta files found!")
    avg_beta = sum_beta / count_b      # shape [H, V]

    # --- α heatmap (temporal × feature) ---
    plt.figure(figsize=(10,4))
    im = plt.imshow(avg_alpha.T, aspect='auto', cmap='viridis')
    plt.colorbar(im, label="Avg α weight")
    plt.xticks(np.arange(seq_len), np.arange(1, seq_len+1))
    plt.yticks(np.arange(V), feature_names, fontsize=9)
    plt.xlabel("Input time step")
    plt.ylabel("Feature")
    plt.title(f"{model_name} ({run_name}) ▶ α temporal × feature\n(seq_len={seq_len})")
    plt.tight_layout()
    fn1 = os.path.join(out_dir, f"{model_name}_{run_name}_alpha_temporal_heatmap.png")
    plt.savefig(fn1, dpi=300)
    plt.show()

    # --- mean α over time (line) ---
    plt.figure(figsize=(8,3))
    plt.plot(np.arange(1, seq_len+1), avg_alpha.mean(axis=1), marker='o')
    plt.xlabel("Input time step")
    plt.ylabel("Mean α weight")
    plt.title(f"{model_name} ({run_name}) ▶ mean α over time\n(seq_len={seq_len})")
    plt.grid(True)
    plt.tight_layout()
    fn2 = os.path.join(out_dir, f"{model_name}_{run_name}_alpha_temporal_mean.png")
    plt.savefig(fn2, dpi=300)
    plt.show()

    # --- β heatmap (forecast-step × feature) ---
    H, V = avg_beta.shape
    plt.figure(figsize=(10,4))
    im = plt.imshow(avg_beta, aspect='auto', cmap='plasma')
    plt.colorbar(im, label="Avg β weight")
    plt.xticks(np.arange(V), feature_names, rotation=45, fontsize=9)
    plt.yticks(np.arange(H), [f"Step {i+1}" for i in range(H)], fontsize=9)
    plt.xlabel("Feature")
    plt.ylabel("Forecast step")
    plt.title(f"{model_name} ({run_name}) ▶ β forecast-step × feature\n(pred_len={pred_len})")
    plt.tight_layout()
    fn3 = os.path.join(out_dir, f"{model_name}_{run_name}_beta_forecast_heatmap.png")
    plt.savefig(fn3, dpi=300)
    plt.show()
