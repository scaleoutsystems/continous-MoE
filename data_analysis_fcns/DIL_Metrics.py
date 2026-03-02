def plot_results(log_dir="results"):
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    log_dir = Path(log_dir)

    with open(log_dir / "metrics.json") as f:
        data = json.load(f)
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