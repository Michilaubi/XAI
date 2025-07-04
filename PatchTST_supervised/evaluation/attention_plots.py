import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------
# PatchTST attention plotting
# ------------------

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_patchtst_temporal(
    attn_dir: str,
    num_layers: int,
    num_batches: int,
    num_vars: int,
    num_heads: int,
    feature_names: list[str],
    model_name: str = "PatchTST",
    run_name:   str = "default"
):
    """
    Plot patch‐level temporal & head‐wise attention heatmaps for PatchTST.
    attn_dir: directory containing .npy files named 'attn_batch_{b}_layer_{l}.npy'
    num_layers / num_batches: how many layers and batches you dumped
    num_vars, num_heads: your V and H
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # we’ll infer Q (#patches) from the first file we actually load
    temporal_acc = None   # will become V×K
    head_acc     = None   # will become H×K
    files = 0

    for b in range(num_batches):
        for l in range(num_layers):
            fn = os.path.join(attn_dir, f"attn_batch_{b}_layer_{l}.npy")
            if not os.path.isfile(fn):
                continue
            A = np.load(fn)    # shape (B*V, H, Q, K)
            if A.size == 0:
                continue

            Bv, H, Q, K = A.shape
            if H != num_heads:
                raise RuntimeError(f"Got H={H} heads, expected {num_heads}")
            B = Bv // num_vars
            if B * num_vars != Bv:
                raise RuntimeError(f"Batch‐var dim {Bv} not divisible by V={num_vars}")

            A = A.reshape(B, num_vars, num_heads, Q, K)

            # initialize accumulators once we know K
            if temporal_acc is None:
                temporal_acc = np.zeros((num_vars, K), dtype=float)
                head_acc     = np.zeros((num_heads, K), dtype=float)

            # — average over batch & queries (axis=0,3) → [V × K]
            temporal_acc += A.mean(axis=(0,2,3))
            # — average over batch, vars & queries (axis=0,1,3) → [H × K]
            head_acc     += A.mean(axis=(0,1,3))

            files += 1

    if files == 0:
        raise RuntimeError(f"No PatchTST attention files found in {attn_dir}!")

    temporal_avg = temporal_acc / files   # V×K
    head_avg     = head_acc     / files   # H×K
    K = temporal_avg.shape[1]

    # --- Plot temporal per-feature (V×K) ---
    plt.figure(figsize=(10,5))
    im = plt.imshow(temporal_avg, aspect='auto', cmap='viridis')
    plt.colorbar(im, label="Avg attention weight")
    plt.yticks(np.arange(num_vars), feature_names, fontsize=10)
    # only label 6 ticks along K
    xt = np.linspace(0, K-1, 6, dtype=int)
    plt.xticks(xt, [f"-{i}" for i in np.linspace(Q, 0, 6, dtype=int)])
    plt.xlabel("Patch index (earlier → later)")
    plt.title(f"{model_name} ({run_name}) ▶ Temporal attention per feature\n(patches={K})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_temporal_per_feature.png"), dpi=300)
    plt.show()

    # --- Plot head‐wise (H×K) ---
    plt.figure(figsize=(10,4))
    im = plt.imshow(head_avg, aspect='auto', cmap='plasma')
    plt.colorbar(im, label="Avg attention weight")
    plt.yticks(np.arange(num_heads), [f"Head {h+1}" for h in range(num_heads)], fontsize=9)
    plt.xticks(xt, [f"-{i}" for i in np.linspace(Q, 0, 6, dtype=int)])
    plt.xlabel("Patch index")
    plt.title(f"{model_name} ({run_name}) ▶ Head‐wise attention\n(patches={K})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_headwise_temporal.png"), dpi=300)
    plt.show()


# ------------------
# IMV-LSTM attention plotting
# ------------------

def plot_imv_saved_attention(
    attn_dir: str,
    seq_len: int,
    pred_len: int,
    feature_names: list[str],
    model_name: str = "IMV-LSTM",
    run_name: str = "default"
):
    """
    Plot α (temporal×feature), mean α over time, and β (forecast×feature) heatmaps,
    but only show 6 nicely spaced ticks on each axis.
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # — load & average —
    alpha_files = [f for f in os.listdir(attn_dir) if f.startswith("alphas") and f.endswith(".npy")]
    beta_files  = [f for f in os.listdir(attn_dir) if f.startswith("betas")  and f.endswith(".npy")]

    sum_alpha = None; count_a = 0
    for fn in alpha_files:
        arr = np.load(os.path.join(attn_dir, fn))
        if arr.ndim == 4: arr = arr[...,0]   # [B, T, V]
        B, T, V = arr.shape
        sum_alpha = arr.sum(axis=0) if sum_alpha is None else sum_alpha + arr.sum(axis=0)
        count_a += B
    avg_alpha = sum_alpha/count_a  # [T, V]

    sum_beta = None; count_b = 0
    for fn in beta_files:
        arr = np.load(os.path.join(attn_dir, fn))
        if arr.ndim == 4: arr = arr[...,0]   # [B, H, V]
        B, H, V = arr.shape
        sum_beta = arr.sum(axis=0) if sum_beta is None else sum_beta + arr.sum(axis=0)
        count_b += B
    avg_beta = sum_beta/count_b   # [H, V]

    # prepare tick positions
    def spaced_ticks(length, n_ticks=6):
        """Return n_ticks evenly spaced integer positions from 0→length-1."""
        return np.linspace(0, length-1, n_ticks, dtype=int)

    # — α heatmap —  
    plt.figure(figsize=(10,4))
    im = plt.imshow(avg_alpha.T, aspect='auto', cmap='viridis')
    plt.colorbar(im, label="Avg α weight")

    xt = spaced_ticks(seq_len, 6)
    plt.xticks(xt, (xt+1).tolist(), rotation=0)        # label 1, …, seq_len
    plt.yticks(np.arange(len(feature_names)), feature_names, fontsize=9)

    plt.xlabel("Encoding time step")
    plt.ylabel("Feature")
    plt.title(f"{model_name} ({run_name}) ▶ α temporal×feature")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_alpha_temporal.png"), dpi=300)
    plt.show()

    # — mean α over time —  
    plt.figure(figsize=(8,3))
    mt = spaced_ticks(seq_len, 6)
    plt.plot(np.arange(1, seq_len+1), avg_alpha.mean(axis=1), marker='o')
    plt.xticks(mt, (mt+1).tolist())
    plt.xlabel("Encoding time step")
    plt.ylabel("Mean α weight")
    plt.title(f"{model_name} ({run_name}) ▶ mean α over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_alpha_mean.png"), dpi=300)
    plt.show()

    # — β heatmap —  
    plt.figure(figsize=(10,4))
    im = plt.imshow(avg_beta, aspect='auto', cmap='plasma')
    plt.colorbar(im, label="Avg β weight")

    yt = spaced_ticks(avg_beta.shape[0], 6)
    plt.yticks(yt, [f"Step {i+1}" for i in yt], fontsize=9)
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, fontsize=9)

    plt.xlabel("Feature")
    plt.ylabel("Forecast step")
    plt.title(f"{model_name} ({run_name}) ▶ β forecast×feature")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_beta_forecast.png"), dpi=300)
    plt.show()