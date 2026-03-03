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
        "intransigence",
    ]:
        values = [
            h[key] for h in hist
            if h[key] is not None
        ]
        if len(values) > 0:
            plt.plot(epochs[:len(values)], values,
                     label=key)

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
    if confusions is None:
        print("No confusion matrices found in log data.")
        return

    for d, conf in enumerate(confusions):

        if torch.is_tensor(conf):
            conf = conf.cpu().numpy()

        conf = conf.astype(float)

        # Row-normalize (P(pred=j | true=i))
        row_sums = conf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        conf_norm = conf / row_sums

        remove_empty = False # set to True to exclude classes with no samples in this domain
        class_names = None # optionally provide class names as a list, not currently supported by DIL_Logger but could be added as metadata in the future

        # Optionally remove empty classes
        if remove_empty:
            active = (conf.sum(axis=1) > 0) | (conf.sum(axis=0) > 0)
            conf_norm = conf_norm[active][:, active]
            labels = (
                np.array(class_names)[active]
                if class_names is not None
                else np.arange(conf.shape[0])[active]
            )
        else:
            labels = (
                class_names
                if class_names is not None
                else np.arange(conf.shape[0])
            )

        fig, ax = plt.subplots(figsize=(6, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_norm,
            display_labels=labels,
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            colorbar=True,
            im_kw={"vmin": 0.0, "vmax": 1.0}
        )

        ax.set_title(f"Domain {d} (Row-normalized)")
        plt.tight_layout()
        plt.show()