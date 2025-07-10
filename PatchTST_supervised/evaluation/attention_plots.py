import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from cycler import cycler
import seaborn as sns



def plot_patchtst_temporal(
    attn_dir: str,
    num_layers: int,
    num_batches: int,     # only used if no valid avg files
    num_vars: int,
    num_heads: int,
    feature_names: list[str],
    model_name: str = "PatchTST",
    run_name:   str = "default"
):
    """
    Plot patch‐level temporal & head‐wise attention heatmaps for PatchTST.
    If you have files named avg_attn_layer0.npy … avg_attn_layer{num_layers-1}.npy
    *and* their first dim is divisible by num_vars, they will be used.
    Otherwise we fall back to per‐batch files attn_batch_{b}_layer_{l}.npy.
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Try aggregated‐files first
    avg_files = sorted(glob.glob(os.path.join(attn_dir, "avg_attn_layer*.npy")))
    use_agg = False
    if len(avg_files) == num_layers:
        Bv0, H0, Q, K = np.load(avg_files[0]).shape
        # only if Bv0 divisible by V do we treat as CI aggregated
        if Bv0 % num_vars == 0 and H0 == num_heads:
            use_agg = True

    if use_agg:
        V = num_vars
        H = num_heads
        B = Bv0 // V

        temporal_acc = np.zeros((V, K), dtype=float)
        head_acc     = np.zeros((H, K), dtype=float)

        for fn in avg_files:
            A = np.load(fn)            # shape (Bv, H, Q, K)
            Bv, h, q, k = A.shape
            # reshape -> (B, V, H, Q, K)
            A = A.reshape(B, V, H, q, k)

            temporal_acc += A.mean(axis=(0,2,3))
            head_acc     += A.mean(axis=(0,1,3))

        files = len(avg_files)
        temporal_avg = temporal_acc / files
        head_avg     = head_acc     / files

    else:
        # fallback to per‐batch files
        temporal_acc = None
        head_acc     = None
        files = 0

        for b in range(num_batches):
            for l in range(num_layers):
                fn = os.path.join(attn_dir, f"attn_batch_{b}_layer_{l}.npy")
                if not os.path.isfile(fn): continue
                A = np.load(fn)  # shape (B*V, H, Q, K)
                if A.size == 0:  continue

                Bv, H, Q, K = A.shape
                V = num_vars
                B = Bv // V
                if B * V != Bv:
                    raise RuntimeError(f"Bv={Bv} not divisible by V={V}")

                A = A.reshape(B, V, H, Q, K)
                if temporal_acc is None:
                    temporal_acc = np.zeros((V, K), dtype=float)
                    head_acc     = np.zeros((H, K), dtype=float)

                temporal_acc += A.mean(axis=(0,2,3))
                head_acc     += A.mean(axis=(0,1,3))
                files += 1

        if files == 0:
            raise RuntimeError(f"No attention files found in {attn_dir}!")
        temporal_avg = temporal_acc / files
        head_avg     = head_acc     / files

    # now plot (identical styling for both branches)
    xt = np.linspace(0, K-1, 6, dtype=int)

    sns.set_theme(style="white", palette="deep", font="serif", font_scale=1.1)
    plt.rcParams['axes.prop_cycle'] = cycler('color',
        ['#004488','#DDAA33','#BB5566','#66BBEE']
    )

    # --- Temporal per‐feature ---
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(temporal_avg, aspect='auto',
                   cmap='viridis', interpolation='nearest')
    ax.grid(False)
    ax.set_yticks(np.arange(num_vars))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"-{i}" for i in np.linspace(Q, 0, len(xt), dtype=int)])
    ax.set_xlabel("Patch index (earlier → later)")
    ax.set_title(f"{model_name} ({run_name}) ▶ Temporal attention per feature\n(patches={K})")
    fig.colorbar(im, ax=ax, label="Avg attention weight")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(
        out_dir, f"{model_name}_{run_name}_temporal_per_feature.png"
    ), dpi=300)
    plt.show
    #plt.close(fig)

    # --- Head‐wise ---
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(head_avg, aspect='auto',
                   cmap='plasma', interpolation='nearest')
    ax.grid(False)
    ax.set_yticks(np.arange(num_heads))
    ax.set_yticklabels([f"Head {h+1}" for h in range(num_heads)], fontsize=9)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"-{i}" for i in np.linspace(Q, 0, len(xt), dtype=int)])
    ax.set_xlabel("Patch index")
    ax.set_title(f"{model_name} ({run_name}) ▶ Head-wise attention\n(patches={K})")
    fig.colorbar(im, ax=ax, label="Avg attention weight")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(
        out_dir, f"{model_name}_{run_name}_headwise_temporal.png"
    ), dpi=300)
    plt.show
    #plt.close(fig)

    print(f"✅ Saved plots to {out_dir}")
    # --- Mean attention over patches (across variables) ---
    # temporal_avg: V×K  → mean over V → length‐K
    mean_patch = temporal_avg.mean(axis=0)
    sns.set_theme(style="white", palette="deep", font="serif", font_scale=1.1)
    plt.rcParams['axes.prop_cycle'] = cycler('color',
        ['#004488','#DDAA33','#BB5566','#66BBEE'])
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(np.arange(1, K+1), mean_patch, marker='o', linewidth=1.5)
    # reuse the same xticks as before
    xt = np.linspace(0, K-1, 6, dtype=int)
    ax.set_xticks(xt + 1)   # 1-based
    ax.set_xticklabels([f"-{i}" for i in np.linspace(Q, 0, len(xt), dtype=int)])
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(f"{model_name} ({run_name}) ▶ mean attention per patch")
    ax.grid(False)
    plt.tight_layout()

    save_path3 = os.path.join(
        out_dir, f"{model_name}_{run_name}_mean_patch_attention.png"
    )
    fig.savefig(save_path3, dpi=300)
    print(f"Saved: {save_path3}")
    plt.show()
    #plt.close(fig)


def plot_patchtst_cross_series(
    attn_dir: str,
    num_layers: int,
    num_batches: int,
    num_vars: int,
    num_heads: int,
    patch_num: int,
    feature_names: list[str],
    model_name: str = "PatchTST",
    run_name:   str = "default",
):
    """
    Compute and plot the V×V cross-series attention matrix.
    If you have avg_attn_layer{ℓ}.npy, those will be used.
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    avg_files = sorted(glob.glob(os.path.join(attn_dir, "avg_attn_layer*.npy")))
    cross_acc = np.zeros((num_vars, num_vars), dtype=float)
    files = 0

    if len(avg_files)==num_layers:
        # use the aggregated files
        for fn in avg_files:
            A = np.load(fn)  # shape (Bv, H, Q, K) or for CM (B, H, Q, Q)
            Bv, H, Q, K = A.shape
            # CI case?
            if Bv % num_vars == 0 and Q==K:
                B = Bv//num_vars
                A = A.reshape(B, num_vars, H, Q, Q)  # (B, V, H, P, P)
                # only diagonal blocks survive, average over batch, head, patch→(V,)
                M = A.mean(axis=(0,2,3,4))
                cross_acc += np.diag(M)
            else:
                # CM: Q == num_vars*patch_num
                if Q != num_vars*patch_num or Q!=K:
                    raise ValueError("Unexpected dims")
                B = Bv
                A = A.reshape(B, H, num_vars, patch_num, num_vars, patch_num)
                M = A.mean(axis=(0,1,3,5))  # (V, V)
                cross_acc += M
            files += 1

    else:
        # fallback to your old per-batch loop
        for b in range(num_batches):
            for l in range(num_layers):
                fn = os.path.join(attn_dir, f"attn_batch_{b}_layer_{l}.npy")
                if not os.path.isfile(fn): continue
                A = np.load(fn)
                if A.ndim==4 and A.shape[0] % num_vars==0 and A.shape[2]==A.shape[3]:
                    B = A.shape[0]//num_vars
                    _,H,P,_ = A.shape
                    A = A.reshape(B, num_vars, H, P, P)
                    M = A.mean(axis=(0,2,3,4))
                    cross_acc += np.diag(M)
                    files += 1
                else:
                    B,H,Q,_ = A.shape
                    if Q!=num_vars*patch_num:
                        raise ValueError("bad dims")
                    A = A.reshape(B, H, num_vars, patch_num, num_vars, patch_num)
                    M = A.mean(axis=(0,1,3,5))
                    cross_acc += M
                    files += 1

    if files==0:
        raise RuntimeError("No attention files found!")

    cross_avg = cross_acc / files
    
    # Common styling
    plt.rcParams['axes.prop_cycle'] = cycler('color',
        ['#004488','#DDAA33','#BB5566','#66BBEE'])
    sns.set_theme(
        style="white",      # no grid
        palette="deep",
        font="serif",
        font_scale=1.1
    )
    
    # Cross-series attention heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(
        cross_avg,
        aspect='equal',
        cmap='coolwarm',
        interpolation='nearest'
    )
    ax.grid(False)
    
    ax.set_xticks(np.arange(num_vars))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(num_vars))
    ax.set_yticklabels(feature_names, fontsize=9)
    
    ax.set_xlabel("Key series")
    ax.set_ylabel("Query series")
    ax.set_title(f"PatchTST ▶ Cross-series attention")
    
    cbar = fig.colorbar(im, ax=ax, label="Avg attn weight")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    fig.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_cross_series_attn.png"), dpi=300)
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
    mean_alpha = avg_alpha.mean(axis=1)       # shape (T,)
    np.save(os.path.join(out_dir, f"{model_name}_{run_name}_mean_alpha.npy"),
        mean_alpha.astype(np.float32))
    # prepare tick positions
    def spaced_ticks(length, n_ticks=6):
        """Return n_ticks evenly spaced integer positions from 0→length-1."""
        return np.linspace(0, length-1, n_ticks, dtype=int)

    # 1) Common styling
    plt.rcParams['axes.prop_cycle'] = cycler('color',
        ['#004488','#DDAA33','#BB5566','#66BBEE'])
    sns.set_theme(
        style="white",      # no grid
        palette="deep",
        font="serif",
        font_scale=1.1
    )
    
    # — α temporal×feature heatmap —
    fig, ax = plt.subplots(figsize=(10, 5))
    
    im = ax.imshow(
        avg_alpha.T,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    ax.grid(False)
    
    # ticks
    xt = spaced_ticks(seq_len, 6)
    ax.set_xticks(xt)
    ax.set_xticklabels((xt+1).tolist(), rotation=0)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)
    
    ax.set_xlabel("Encoding time step")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name} ({run_name}) ▶ α temporal×feature")
    
    cbar = fig.colorbar(im, ax=ax, label="Avg α weight")
    fig.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_alpha_temporal.png"), dpi=300)
    plt.show()
    
    
    # — mean α over time line plot —
    fig, ax = plt.subplots(figsize=(10,5))
    
    mt = spaced_ticks(seq_len, 6)
    ax.plot(np.arange(1, seq_len+1), avg_alpha.mean(axis=1), marker='o')
    ax.set_xticks(mt)
    ax.set_xticklabels((mt+1).tolist())
    ax.set_xlabel("Encoding time step")
    ax.set_ylabel("Mean α weight")
    ax.set_title(f"{model_name} ({run_name}) ▶ mean α over time")
    
    ax.grid(False)
    fig.subplots_adjust(bottom=0.20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_alpha_mean.png"), dpi=300)
    plt.show()

    # — β heatmap — 

        # your color cycle
    plt.rcParams['axes.prop_cycle'] = cycler(
        'color', ['#004488','#DDAA33','#BB5566','#66BBEE']
    )
    
    # turn off the grid style
    sns.set_theme(
        style="white",      # <-- no grid
        palette="deep",
        font="serif",
        font_scale=1.1
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # show heatmap without interpolation artifacts
    im = ax.imshow(
        avg_beta,
        aspect='auto',
        cmap='plasma',
        interpolation='nearest'   # no lines
    )
    
    # turn off any remaining grid
    ax.grid(False)
    
    # ticks
    yt = spaced_ticks(avg_beta.shape[0], 6)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"Step {i+1}" for i in yt], fontsize=9)
    
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel("Feature")
    ax.set_ylabel("Forecast step")
    ax.set_title(f"{model_name} ({run_name}) ▶ β forecast×feature")
    
    # colorbar
    cbar = fig.colorbar(im, ax=ax, label="Avg β weight")
    
    # push up the plot a bit so labels don't get clipped
    fig.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_{run_name}_beta_forecast.png"), dpi=300)
    plt.show()

def plot_imv_cross_series(
    attn_dir: str,
    feature_names: list[str],
    model_name: str = "IMV-LSTM",
    run_name:   str = "default"
):
    """
    Compute & plot the V×V 'cross-series' attention matrix for IMV-LSTM.
    It simply averages all stored β arrays over batch & forecast steps,
    then places that V-vector on the diagonal of a V×V matrix.
    """
    out_dir = os.path.join(attn_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # 1) find all of your saved betas files
    beta_files = sorted(glob.glob(os.path.join(attn_dir, "betas*.npy")))
    if not beta_files:
        raise RuntimeError(f"No β files found in {attn_dir}")

    # 2) accumulate them
    sum_beta = None
    total_steps = 0
    for fn in beta_files:
        arr = np.load(fn)  
        # arr might be shape [B, H, V, 1] or [B, H, V]
        if arr.ndim == 4:
            arr = arr[...,0]
        B, H, V = arr.shape
        # average over forecast steps H first
        mean_over_steps = arr.mean(axis=1)  # → shape [B, V]
        # sum across batches
        sum_beta = mean_over_steps.sum(axis=0) if sum_beta is None else sum_beta + mean_over_steps.sum(axis=0)
        total_steps += B

    # 3) get final per‐variable attention vector
    avg_beta = sum_beta / total_steps       # → shape (V,)

    # 4) build diagonal cross‐series matrix
    cross = np.diag(avg_beta)               # shape (V, V)

    # 5) plot!
    plt.rcParams['axes.prop_cycle'] = cycler('color',
        ['#004488','#DDAA33','#BB5566','#66BBEE'])
    sns.set_theme(style="white", palette="deep", font="serif", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cross, cmap="coolwarm", aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(V))
    ax.set_yticks(np.arange(V))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Key series")
    ax.set_ylabel("Query series")
    ax.set_title(f"{model_name} ({run_name}) ▶ Cross-series attention")
    cbar = fig.colorbar(im, ax=ax, label="Avg β weight")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_{run_name}_cross_series_attn.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✓ saved cross-series attention plot → {out_path}")
