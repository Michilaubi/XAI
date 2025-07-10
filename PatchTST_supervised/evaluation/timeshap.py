import pandas as pd
import os
import re
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timeshap.wrappers import TorchModelWrapper
from timeshap.explainer.pruning import local_pruning
from timeshap.plot.pruning import plot_temp_coalition_pruning
from timeshap.explainer.event_level import local_event
from timeshap.plot.event_level import plot_event_heatmap
from timeshap.explainer.feature_level import local_feat
from timeshap.plot.feature_level import plot_feat_barplot

class WindowedDataset(Dataset):
    """
    Wraps a dataset of (X, y, ...) pairs and returns only a temporal window [start:end]
    of the X array, leaving other outputs unchanged.
    """
    def __init__(self, dataset, start, end):
        self.dataset = dataset
        self.start = start
        self.end = end

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = item[0]
        x_sub = x[..., self.start:self.end, :]
        return (x_sub, *item[1:])


def slice_loader(loader, start, end):
    """
    Returns a new DataLoader over a WindowedDataset slicing each X in [start:end].
    Uses num_workers=0 to avoid pickling issues.
    """
    ds = loader.dataset
    subds = WindowedDataset(ds, start, end)
    return DataLoader(
        subds,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
    )



def explain_model_window(
    model,
    train_loader,
    test_dataset,
    seq_len,
    start,
    end,
    feature_names: list[str],
    target_channel: int = -1,
    background_size: int = 10,
    pruning_kwargs: dict | None = None,
    event_kwargs:   dict | None = None,
    feature_kwargs: dict | None = None,
    device=None,
    out_prefix="run",
):
    os.makedirs(out_prefix, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.eval().to(device)

    # 1) build full‐sequence background & baseline
    bg_list, cnt = [], 0
    for batch in train_loader:
        xb = batch[0] if isinstance(batch, (tuple,list)) else batch
        arr = xb.cpu().numpy() if torch.is_tensor(xb) else xb
        bg_list.append(arr); cnt += arr.shape[0]
        if cnt >= background_size: break

    background_full = np.vstack(bg_list)[:background_size]  # (B, seq_len, n_feat)

    # instead of picking the first one, use the mean (or median) across B
    baseline_full = background_full.mean(axis=0, keepdims=True)  # (1, seq_len, n_feat)
    # -- or:
    # baseline_full = np.median(background_full, axis=0, keepdims=True)
                 # (1, seq_len, n_feat)

    # 2) load & window the test example
    x0       = test_dataset[0][0]
    x_full   = (x0.cpu().numpy() if torch.is_tensor(x0) else x0)[None,...]
    x_window = x_full[:, start:end, :]                      # (1, win_len, n_feat)

    # 3) model‐reconstruction helper
    def f_last(X_sub: np.ndarray) -> np.ndarray:
        N, wlen, nfeat = X_sub.shape
        X_recon = np.repeat(baseline_full, N, axis=0)
        X_recon[:, start:end, :] = X_sub
        t     = torch.from_numpy(X_recon).float().to(device)
        out   = model(t)
        y_hat = out[0] if isinstance(out, tuple) else out
        last  = y_hat[:, -1, target_channel]
        return last.detach().cpu().numpy().reshape(-1,1)

    # 4) prepare windowed baseline for SHAP calls
    background_sub = background_full[:, start:end, :]  # (B, win_len, n_feat)

    # 5) Temporal–coalition pruning
    pruning_dict    = pruning_kwargs or {"tol":0.1, "nsamples":200}
    coal_plot, coal_idx = local_pruning(
        f_last,
        x_window,
        pruning_dict,
        background_sub,
        verbose=False
    )
    chart1 = plot_temp_coalition_pruning(coal_plot, coal_idx)
    chart1.save(os.path.join(out_prefix, "pruning.html"))

    # 6) Event‐level
    ev_dict = event_kwargs or {"nsamples":40, "noise_variance":1.0}
    ev_attr = local_event(
        f_last,
        x_window,
        ev_dict,
        None, None,
        background_sub,
        coal_idx
    )
    chart2 = plot_event_heatmap(ev_attr)
    chart2.save(os.path.join(out_prefix, "event.html"))

    # 7) Feature‐level
    feat_dict = feature_kwargs or {"nsamples":30}
    feat_attr = local_feat(
        f_last,
        x_window,
        feat_dict,
        None, None,
        background_sub,
        coal_idx
    )
    # map integer indices → human names
    def map_feat_label(val):
        s = str(val)
        m = re.search(r"(\d+)", s)
        return feature_names[int(m.group(1))] if m else s
    feat_attr["Feature"] = feat_attr["Feature"].apply(map_feat_label)

    chart3 = plot_feat_barplot(feat_attr)
    chart3.save(os.path.join(out_prefix, "feature.html"))

    return {
        "pruning": (coal_plot, coal_idx),
        "event":   ev_attr,
        "feature": feat_attr,
        "charts":  (chart1, chart2, chart3),
    }


def explain_with_chunks(
    model,
    train_loader,
    test_dataset,
    seq_len,
    *,
    feature_names: list[str],
    chunk_thresh: int = 100,
    **explain_kwargs
):
    """
    Splits any seq_len > chunk_thresh into non-overlapping windows of size chunk_thresh.
    Passes `feature_names` & all other kwargs down to explain_model_window.
    """
    base_dir  = explain_kwargs.get("out_prefix", "run")
    n_chunks  = math.ceil(seq_len / chunk_thresh)
    results   = {}

    def call_window(s,e,out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return explain_model_window(
            model,
            train_loader,
            test_dataset,
            seq_len,
            start          = s,
            end            = e,
            feature_names  = feature_names,
            **{k:v for k,v in explain_kwargs.items() if k!="out_prefix"}
        )

    if n_chunks == 1:
        results["full"] = call_window(0, seq_len, base_dir)
    else:
        for c in range(n_chunks):
            s    = c*chunk_thresh
            e    = min((c+1)*chunk_thresh, seq_len)
            cdir = os.path.join(base_dir, f"chunk{c}")
            print(f">>> Explaining chunk {c+1}/{n_chunks}: [{s}:{e}] …")
            results[c] = call_window(s, e, cdir)

    return results




def plot_chunk_importance(results, top_key):
    """
    Summarize and plot Σ|Shapley| per time-chunk.
    
    Args:
      results: dict returned by explain_with_chunks
      top_key: the top-level key under which your chunks live, e.g. "IMV_short"
    """
    inner = results[top_key]
    chunk_ids, chunk_scores = [], []
    
    for cid, info in inner.items():
        if "pruning" not in info:
            continue
        coal_plot, _ = info["pruning"]
        # extract numeric values
        if isinstance(coal_plot, pd.DataFrame):
            num = coal_plot.select_dtypes(include=[np.number]).iloc[:,0].values
        else:
            num = np.asarray(coal_plot, dtype=float)
        
        chunk_ids.append(cid)
        chunk_scores.append(np.sum(np.abs(num)))
    
    df = pd.DataFrame({
        "chunk": chunk_ids,
        "abs_shap_sum": chunk_scores
    }).sort_values("chunk")
    
    fig, ax = plt.subplots()
    ax.bar(df["chunk"].astype(str), df["abs_shap_sum"])
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Σ |Shapley value|")
    ax.set_title("Relative importance by time-chunk")
    plt.tight_layout()
    plt.show()
    
    
def plot_feature_importance(results, top_key, chunk_id):
    """
    Summarize and plot Σ|Shapley| per feature within a single chunk.
    
    Args:
      results: dict returned by explain_with_chunks
      top_key: the top-level key, e.g. "IMV_short"
      chunk_id: which chunk to drill into (e.g. 0,1,2,3)
    """
    info = results[top_key][chunk_id]
    feat_df = info["feature"]  # pandas DataFrame with columns ["Feature","Shapley Value",...]
    
    # assume column named "Shapley Value"
    shap_col = "Shapley Value"
    if shap_col not in feat_df.columns:
        raise KeyError(f"Expected column '{shap_col}' in feature DataFrame")
    
    df = feat_df.assign(abs_shap=feat_df[shap_col].abs())
    df = df.groupby("Feature")["abs_shap"].sum().reset_index()
    df = df.sort_values("abs_shap", ascending=False)
    
    fig, ax = plt.subplots()
    ax.barh(df["Feature"], df["abs_shap"])
    ax.invert_yaxis()
    ax.set_xlabel("Σ |Shapley value|")
    ax.set_title(f"Feature importance (chunk {chunk_id})")
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import warnings

def plot_aggregated_feature_importance(attn_dir,
    results: dict,
    top_key: str,
    feature_names: list[str] | None = None
):
    out_dir = os.path.join(attn_dir, "plots")
    """
    Aggregate and plot Σ|Shapley| per feature across *all* chunks.

    Args:
      results:      dict returned by explain_with_chunks
      top_key:      the top-level key under which your chunks live, e.g. "PatchTST_mid"
      feature_names:
        If provided, should be the full list of your dataset’s features in
        their *original* order.  We'll try to map whatever SHAP put in the
        "Feature" column back to these names by index.
    """
    inner = results[top_key]

    # 1) collect per‐chunk DataFrames
    dfs = []
    for cid, info in inner.items():
        feat_df = info.get("feature")
        if feat_df is None:
            continue
        if not {"Feature", "Shapley Value"}.issubset(feat_df.columns):
            raise KeyError(f"Chunk {cid} missing expected columns")
        df = feat_df[["Feature", "Shapley Value"]].copy()
        df["abs_shap"] = df["Shapley Value"].abs()
        dfs.append(df[["Feature", "abs_shap"]])

    # 2) concatenate and sum by feature
    all_feats = pd.concat(dfs, ignore_index=True)
    agg = (
        all_feats
        .groupby("Feature")["abs_shap"]
        .sum()
        .reset_index()
        .sort_values("abs_shap", ascending=False)
    )

    # 3) if user supplied a feature_names list, build an index→name map
    if feature_names is not None:
        # build mapping: both int and str keys to your names
        mapping = {}
        for idx, name in enumerate(feature_names):
            mapping[idx] = name
            mapping[str(idx)] = name

        # try to map
        agg["Mapped"] = agg["Feature"].map(mapping)
        missing = agg.loc[agg["Mapped"].isna(), "Feature"].unique()
        if len(missing) > 0:
            warnings.warn(
                f"Could not map feature keys {list(missing)} → "
                f"check your feature_names list or SHAP output."
            )
            # fallback to original
            agg["Mapped"] = agg["Mapped"].fillna(agg["Feature"].astype(str))

        agg["Feature"] = agg["Mapped"]
        agg = agg.drop(columns=["Mapped"])
   

    # 4) plot 
    sns.set_theme(
    style="white",      # no grid
    palette="deep",
    font="serif",
    font_scale=1.1)
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#004488','#DDAA33','#BB5566','#66BBEE'])
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3*len(agg))))
    ax.barh(agg["Feature"], agg["abs_shap"])
    ax.invert_yaxis()
    ax.set_xlabel("Σ |Shapley value| across all chunks")
    ax.set_title(f"Overall feature importance ({top_key})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{top_key}_feature_importance.png"), dpi=300)
    plt.show()


def plot_chunk_importance_by_events(results, top_key):
    """
    Summarize and plot Σ |Shapley| per time-chunk, using event-level attributions.
    
    Args:
      results: dict returned by explain_with_chunks
      top_key: the top-level key under which your chunks live, e.g. "IMV_long"
    """
    inner = results[top_key]
    chunk_ids, chunk_scores = [], []

    for cid, info in sorted(inner.items()):
        ev_df = info.get("event")
        if ev_df is None or ev_df.empty:
            continue

        # just sum the absolute per‐event SHAP values
        shap_abs = ev_df["Shapley Value"].abs().fillna(0)
        chunk_ids.append(cid)
        chunk_scores.append(shap_abs.sum())

    df = pd.DataFrame({
        "chunk":        chunk_ids,
        "abs_shap_sum": chunk_scores
    }).sort_values("chunk")
    
    sns.set_theme(
        style="white",      # no grid
        palette="deep",
        font="serif",
        font_scale=1.1
    )
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(df["chunk"].astype(str), df["abs_shap_sum"])
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Σ |Shapley value|")
    ax.set_title("Relative importance by time-chunk (event-level SHAP)")
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

def plot_chunk_importance_by_events_line(attn_dir: str,results, top_key):
    """
    Summarize and plot Σ |Shapley| per time-chunk as a line chart, using event-level attributions.
    
    Args:
      results: dict returned by explain_with_chunks
      top_key: the top-level key under which your chunks live, e.g. "IMV_long"
    """
    out_dir = os.path.join(attn_dir, "plots")
    # --- 1) Common styling ---


    inner = results[top_key]
    chunk_ids, chunk_scores = [], []

    for cid, info in sorted(inner.items()):
        ev_df = info.get("event")
        if ev_df is None or ev_df.empty:
            continue

        # sum absolute per‐event SHAP values
        shap_abs = ev_df["Shapley Value"].abs().fillna(0)
        chunk_ids.append(cid)
        chunk_scores.append(shap_abs.sum())

    df = pd.DataFrame({
        "chunk":        chunk_ids,
        "abs_shap_sum": chunk_scores
    }).sort_values("chunk")

    # --- Line plot ---
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["chunk"], df["abs_shap_sum"], marker='o', linewidth=1.5)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Σ |Shapley value|")
    ax.set_title("Relative importance by time-chunk (event-level SHAP)")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{top_key}_chunk_event_importance.png"), dpi=300)
    plt.show()

def export_full_pruning_shap(results_dict, seq_len, chunk_thresh, out_path):
    """
    Stitch per‐chunk TimeSHAP pruning outputs into one full-length array
    and save as .npy.
    
    Args:
      results_dict: dict from explain_with_chunks(...)[out_prefix]
      seq_len:      int, total input length (e.g. 144)
      chunk_thresh: int, size of each chunk (e.g. 20)
      out_path:     str, where to write the final .npy
    """
    stitched = []
    for chunk_id, info in sorted(results_dict.items()):
        coal_plot, coal_idx = info["pruning"]   # coal_plot is a pandas.DataFrame
        df = coal_plot.copy()
        
        # --- identify which column is the local time index (integer dtype)
        #     and which is the Shapley value (float dtype)
        is_int   = df.dtypes.apply(lambda dt: np.issubdtype(dt, np.integer))
        is_float = df.dtypes.apply(lambda dt: np.issubdtype(dt, np.floating))
        tcol     = df.columns[is_int][0]
        valcol   = df.columns[is_float][0]
        
        # cast to the right types (just in case)
        df[tcol]   = df[tcol].astype(int)
        df[valcol] = df[valcol].astype(float)
        
        # compute global time
        df["t_global"] = df[tcol] + chunk_id * chunk_thresh
        
        # keep only (t_global, shap) 
        stitched.append(df[["t_global", valcol]].rename(columns={valcol: "shap"}))
    
    # concatenate all chunks
    all_ = pd.concat(stitched, ignore_index=True)
    
    # build a dict of t_global→shap (last one wins, but there should be no overlap)
    mapping = dict(zip(all_["t_global"], all_["shap"]))
    
    # now pull out exactly seq_len values in order
    full = np.array([mapping.get(t, 0.0) for t in range(seq_len)], dtype=np.float32)
    
    # save
    np.save(out_path, full)
    print(f"Wrote {len(full)} values → {out_path}")