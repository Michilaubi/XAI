import numpy as np
import torch
from timeshap.explainer.pruning import local_pruning
from timeshap.plot.pruning import plot_temp_coalition_pruning
from timeshap.explainer.event_level import local_event
from timeshap.plot.event_level import plot_event_heatmap
from timeshap.explainer.feature_level import local_feat
from timeshap.plot.feature_level import plot_feat_barplot

def explain_model(
    model,
    train_loader,
    test_dataset,
    seq_len,
    target_channel=-1,
    background_size=20,
    pruning_kwargs=None,
    device=None,
    out_prefix="run",
):
    """
    Runs TimeSHAP (pruning, event level, feature level) on every 4th time-step.
    Saves plots to files named with out_prefix.
    """
    # ─── Device & model setup ───────────────────────────────────────────
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # ─── 1) Build & subsample background ───────────────────────────────
    bg_list, count = [], 0
    for batch in train_loader:
        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
        arr = xb.cpu().numpy() if torch.is_tensor(xb) else xb
        bg_list.append(arr)
        count += arr.shape[0]
        if count >= background_size:
            break
    background = np.vstack(bg_list)[:background_size]     # (B, seq_len, n_feat)
    background = background[:, ::4, :]                     # subsample every 4th step → (B, seq_len/4, n_feat)

    # ─── 2) Pick & subsample one test example ─────────────────────────
    batch0 = test_dataset[0]
    x0     = batch0[0] if isinstance(batch0, (tuple, list)) else batch0
    x_test = x0.cpu().numpy() if torch.is_tensor(x0) else x0
    x_test = x_test[None, ...]                             # (1, seq_len, n_feat)
    x_test = x_test[:, ::4, :]                             # subsample → (1, seq_len/4, n_feat)

    # ─── 3) Helper: last-step forecast ────────────────────────────────
    def f_last(X: np.ndarray) -> np.ndarray:
        t     = torch.from_numpy(X).float().to(device)
        out   = model(t)
        y_hat = out[0] if isinstance(out, tuple) else out
        return y_hat[:, -1, target_channel].detach().cpu().numpy().reshape(-1, 1)

    # ─── 4) Temporal-coalition pruning ─────────────────────────────────
    pruning_dict = pruning_kwargs or {"tol": 0.1}
    coal_plot, coal_idx = local_pruning(f_last, x_test, pruning_dict, background)
    chart1 = plot_temp_coalition_pruning(coal_plot, coal_idx)
    chart1.save(f"{out_prefix}_pruning.html")

    # ─── 5) Event-level attribution ───────────────────────────────────
    event_dict = {"nsamples": 50, "noise_variance": 1.0}
    ev_attr = local_event(
        f_last,
        x_test,
        event_dict,
        None,        # entity_uuid
        None,        # entity_col
        background,
        coal_idx,
    )
    chart2 = plot_event_heatmap(ev_attr)
    chart2.save(f"{out_prefix}_event.html")

    # ─── 6) Feature-level attribution ─────────────────────────────────
    feat_dict = {"nsamples": 50}
    feat_attr = local_feat(
        f_last,
        x_test,
        feat_dict,
        None,
        None,
        background,
        coal_idx,
    )
    # Map raw feature indices to names
    raw_feats = feat_attr["Feature"].tolist()
    names     = [f"f{i}" for i in range(len(raw_feats))]
    feat_map  = {orig: new for orig, new in zip(raw_feats, names)}

    chart3 = plot_feat_barplot(feat_attr, None, feat_map)
    chart3.save(f"{out_prefix}_feature.html")

    # ─── 7) Return attributions & charts ───────────────────────────────
    return {
        "pruning": (coal_plot, coal_idx),
        "event":   ev_attr,
        "feature": feat_attr,
        "charts":  (chart1, chart2, chart3),
    }
