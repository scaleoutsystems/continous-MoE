def plot_results(log_dir):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import torch
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    log_dir = Path(log_dir)

    data = None
    if log_dir.is_file() and log_dir.suffix == ".pt":
        # direct file path passed
        data = torch.load(log_dir, weights_only=False)
    else:
        raise FileNotFoundError(
            f"Please pass the path to a .pt file, not {log_dir}"
        )
    # support both old-format (list) and new format (dict with metadata)
    hist = data.get("history", data)

    epochs = range(len(hist))

    # Domain boundaries
    boundaries = []
    for i, h in enumerate(hist):
        if i > 0 and h["domain"] != hist[i - 1]["domain"]:
            boundaries.append(i)

    # -----------------------
    # Accuracy plot
    # -----------------------
    plt.figure(figsize=(10, 6))

    plt.plot(epochs,
             [h["overall_acc"] for h in hist],
             label="Overall Accuracy")

    plt.plot(epochs,
             [h["avg_inc_acc"] for h in hist],
             label="Avg Incremental Acc")

    for d in range(max(h["domain"] for h in hist) + 1):
        domain_curve = []
        for h in hist:
            if d < len(h["domain_acc_vector"]):
                domain_curve.append(h["domain_acc_vector"][d])
            else:
                domain_curve.append(float('nan'))
        plt.plot(epochs, domain_curve,
                 linestyle="--", alpha=0.5,
                 label=f"Domain {d}")

    for b in boundaries:
        plt.axvline(b, color="black", linestyle=":")

    plt.legend()
    plt.title("Accuracy Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # -----------------------
    # Continual metrics plot
    # -----------------------
    plt.figure(figsize=(10, 6))

    for key in [
        "plasticity",
        "forgetting",
        "bwt",
        "fwt",
        "intransience",
    ]:
        # preserve epoch alignment by creating an array with NaNs for missing values
        values = [h.get(key, None) for h in hist]
        values = [np.nan if v is None else v for v in values]
        if not all(np.isnan(values)):
            plt.plot(list(epochs), values, label=key)

    for b in boundaries:
        plt.axvline(b, color="black", linestyle=":")

    plt.legend()
    plt.title("Continual Learning Metrics")
    plt.xlabel("Epoch")
    plt.show()

    # -----------------------
    # Confusion matrices
    # -----------------------
    confusions = data.get("confusion_matrices", None)
    baseline_confusions = data.get("baseline_confusion_matrices", None)
    if confusions is None:
        print("No confusion matrices found in log data.")
        return

    n_domains = max(len(confusions), len(baseline_confusions) if baseline_confusions is not None else 0)

    for d in range(n_domains):
        model_conf = confusions[d] if d < len(confusions) else None
        base_conf = (baseline_confusions[d] if baseline_confusions is not None and d < len(baseline_confusions) else None)

        def _prepare_conf(c):
            if c is None:
                return None, None
            if torch.is_tensor(c):
                c = c.cpu().numpy()
            c = c.astype(float)
            row_sums = c.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            c_norm = c / row_sums
            # labels (optionally could come from metadata)
            labels = np.arange(c.shape[0])
            return c_norm, labels

        model_norm, labels = _prepare_conf(model_conf)
        base_norm, _ = _prepare_conf(base_conf)

        # choose layout depending on available matrices
        if model_norm is not None and base_norm is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax_model, ax_base = axes
            left_axes = (ax_model, ax_base)
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            left_axes = (ax,)

        if model_norm is not None:
            disp = ConfusionMatrixDisplay(confusion_matrix=model_norm, display_labels=labels)
            disp.plot(ax=left_axes[0], cmap="Blues", values_format=".2f", colorbar=True, im_kw={"vmin": 0.0, "vmax": 1.0})
            left_axes[0].set_title(f"Model Domain {d} (Row-normalized)")

        if base_norm is not None:
            ax_idx = 1 if len(left_axes) > 1 else 0
            disp_b = ConfusionMatrixDisplay(confusion_matrix=base_norm, display_labels=labels)
            disp_b.plot(ax=left_axes[ax_idx], cmap="Oranges", values_format=".2f", colorbar=True, im_kw={"vmin": 0.0, "vmax": 1.0})
            left_axes[ax_idx].set_title(f"Baseline Domain {d} (Row-normalized)")

        plt.tight_layout()
        plt.show()