def plot_results(log_dir):
    """
    Plot general results from a single saved results file. This includes the
    confusion matrices, epoch-by-epoch accuracy, and epoch-by-epoch continual
    learning metrics.

    Parameters:
      - log_dir: path to .pt result file saved by DIL_Logger.save()
    """
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


def _load_log_data(path):
    """Load a .pt result file and return the data dict and history list."""
    import torch
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file {p} does not exist")
    data = torch.load(p, weights_only=False)
    hist = data.get("history", data)
    return data, hist


def plot_confusion_matrices(result_path, per_domain=None, compare_baseline=None, per_epoch=None, epoch_indices=None, normalize='row'):
    """
    Plot confusion matrices from a saved results file.

    Parameters:
      - result_path: path to .pt result file saved by DIL_Logger.save()
      - per_domain: if True, plot per-domain model confusion matrices; if False, skip.
      - compare_baseline: if True and baseline matrices exist, plot baseline alongside model per domain.
      - per_epoch: if True, plot epoch-by-epoch confusion matrices saved under 'epoch_confusion_matrices'.
      - epoch_indices: optional list of epoch indices to plot when per_epoch is True. If None, plot all epochs.
      - normalize: 'row' to row-normalize, 'none' to plot raw counts.

    If none of the options are explicitly True/False, all available confusion plots are shown.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    data, hist = _load_log_data(result_path)

    confusions = data.get("confusion_matrices", None)
    baseline_confusions = data.get("baseline_confusion_matrices", None)
    epoch_confusions = data.get("epoch_confusion_matrices", None)

    # default behavior: if no flags set, plot everything available
    if per_domain is None and compare_baseline is None and per_epoch is None:
        per_domain = True
        compare_baseline = True
        per_epoch = True
    if per_domain is None:
        per_domain = True
    if compare_baseline is None:
        compare_baseline = True
    if per_epoch is None:
        per_epoch = True

    # helper to normalise matrix
    def _norm_matrix(m):
        if m is None:
            return None
        m = np.array(m, dtype=float)
        if normalize == 'row':
            rs = m.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            return m / rs
        return m

    # per-domain comparison
    if per_domain and confusions is not None:
        n_domains = len(confusions)
        for d in range(n_domains):
            model_conf = confusions[d] if d < len(confusions) else None
            base_conf = (baseline_confusions[d] if baseline_confusions is not None and d < len(baseline_confusions) else None)

            m_norm = _norm_matrix(model_conf)
            b_norm = _norm_matrix(base_conf)

            if m_norm is None and b_norm is None:
                continue

            if m_norm is not None and b_norm is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                ax_model, ax_base = axes
                disp = ConfusionMatrixDisplay(confusion_matrix=m_norm, display_labels=np.arange(m_norm.shape[0]))
                disp.plot(ax=ax_model, cmap='Blues', values_format='.2f', colorbar=True, im_kw={"vmin": 0.0, "vmax": 1.0})
                ax_model.set_title(f"Model Domain {d} ({normalize}-norm)")

                disp_b = ConfusionMatrixDisplay(confusion_matrix=b_norm, display_labels=np.arange(b_norm.shape[0]))
                disp_b.plot(ax=ax_base, cmap='Oranges', values_format='.2f', colorbar=True, im_kw={"vmin": 0.0, "vmax": 1.0})
                ax_base.set_title(f"Baseline Domain {d} ({normalize}-norm)")

            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                mat = m_norm if m_norm is not None else b_norm
                disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=np.arange(mat.shape[0]))
                disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True, im_kw={"vmin": 0.0, "vmax": 1.0})
                ax.set_title(f"Domain {d} ({normalize}-norm)")

            plt.tight_layout()
            plt.show()

    # per-epoch list
    if per_epoch and epoch_confusions is not None:
        epochs = list(range(len(epoch_confusions)))
        if epoch_indices is None:
            selected = epochs
        else:
            selected = [e for e in epoch_indices if 0 <= e < len(epoch_confusions)]

        for e in selected:
            conf = epoch_confusions[e]
            if conf is None:
                continue
            mat = _norm_matrix(conf)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=np.arange(mat.shape[0]))
            disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True, im_kw={"vmin": 0.0, "vmax": 1.0})
            ax.set_title(f"Epoch {e} Full-test Confusion ({normalize}-norm)")
            plt.tight_layout()
            plt.show()


def _compute_imbalanced_metrics_from_conf(conf):
    """
    Given an m x m confusion matrix (rows=true, cols=pred), compute
    per-class and aggregated imbalanced learning metrics.

    Metrics taken from Chen et al. (2024):
    https://doi.org/10.1007/s10462-024-10759-6

    Returns a dict with per-class arrays and macro-aggregates:
      - per_class: dict of recall, specificity, sensitivity, precision, f1
      - macro_{recall,specificity,sensitivity,precision,f1}
      - g_mean
      - mauc
    """
    import numpy as np

    M = np.array(conf, dtype=float)
    m = M.shape[0]
    totals = M.sum()
    per = {
        'recall': np.full(m, np.nan),
        'specificity': np.full(m, np.nan),
        'sensitivity': np.full(m, np.nan),
        'precision': np.full(m, np.nan),
        'f1': np.full(m, np.nan),
    }

    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)

    for k in range(m):
        TP = M[k, k]
        FN = row_sums[k] - TP
        FP = col_sums[k] - TP
        TN = totals - TP - FN - FP

        # recall = TP/(TP+FN)
        denom = TP + FN
        per['recall'][k] = TP / denom if denom != 0 else np.nan

        # specificity = TN/(TN+FP)
        denom = TN + FP
        per['specificity'][k] = TN / denom if denom != 0 else np.nan

        # sensitivity = TP/(TN+FP)
        denom = TN + FP
        per['sensitivity'][k] = TP / denom if denom != 0 else np.nan

        # precision = TP/(TP+FP)
        denom = TP + FP
        per['precision'][k] = TP / denom if denom != 0 else np.nan

        # f1 score = 2*recall*precision/(recall+precision)
        r = per['recall'][k]
        p = per['precision'][k]
        per['f1'][k] = 2 * r * p / (r + p) if (r is not None and p is not None and (r + p) != 0) else np.nan

    # macro aggregates (simple average ignoring NaNs)
    macro = {}
    for k in ['recall', 'specificity', 'sensitivity', 'precision', 'f1']:
        macro[f'macro_{k}'] = float(np.nanmean(per[k])) if np.any(~np.isnan(per[k])) else float('nan')

    # G-mean for multiclass: (product(recall_k))^(1/m)
    recalls = per['recall']
    if np.any(np.isnan(recalls)):
        g_mean = float('nan')
    else:
        # if any recall is zero, g_mean becomes zero
        g_mean = float(np.prod(recalls) ** (1.0 / m))

    # MAUC computation: pairwise binary AUCs derived from the 2x2 submatrices
    # MAUC = (2 / (m*(m-1))) * sum( (AUC(i,j) + AUC(j,i)) /2) where i<j
    auc_sum = 0.0
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            # submatrix for classes i and j
            TPi = M[i, i]
            FNi = M[i, j]
            FPi = M[j, i]
            TNi = M[j, j]

            denom_i = TPi + FNi
            denom_i2 = FPi + TNi
            TPR_i = TPi / denom_i if denom_i != 0 else np.nan
            FPR_i = FPi / denom_i2 if denom_i2 != 0 else np.nan
            AUC_i_j = (1.0 + (TPR_i if not np.isnan(TPR_i) else 0.0) - (FPR_i if not np.isnan(FPR_i) else 0.0)) / 2.0

            # reverse
            TPj = M[j, j]
            FNj = M[j, i]
            FPj = M[i, j]
            TNj = M[i, i]
            denom_j = TPj + FNj
            denom_j2 = FPj + TNj
            TPR_j = TPj / denom_j if denom_j != 0 else np.nan
            FPR_j = FPj / denom_j2 if denom_j2 != 0 else np.nan
            AUC_j_i = (1.0 + (TPR_j if not np.isnan(TPR_j) else 0.0) - (FPR_j if not np.isnan(FPR_j) else 0.0)) / 2.0

            pair_mean = np.nanmean([AUC_i_j, AUC_j_i])
            if not np.isnan(pair_mean):
                auc_sum += pair_mean
                count += 1

    if count > 0:
        mauc = float((2.0 / (m * (m - 1))) * auc_sum)
    else:
        mauc = float('nan')

    out = {
        'per_class': per,
        **macro,
        'g_mean': float(g_mean) if g_mean is not None else float('nan'),
        'mauc': float(mauc),
    }
    return out


def plot_imbalanced_metrics(result_path, epoch_indices=None):
    """
    Compute and plot imbalanced-learning metrics per epoch from a results file.

    Produces macro-averaged recall, precision, f1 and the MAUC and G-mean per epoch.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data, hist = _load_log_data(result_path)
    epoch_confusions = data.get('epoch_confusion_matrices', None)
    if epoch_confusions is None:
        print("No epoch-level confusion matrices found in results.")
        return

    n_epochs = len(epoch_confusions)
    indices = list(range(n_epochs)) if epoch_indices is None else [i for i in epoch_indices if 0 <= i < n_epochs]

    macro_recalls = []
    macro_precisions = []
    macro_f1s = []
    macro_specificities = []
    macro_sensitivities = []
    g_means = []
    maucs = []

    for e in indices:
        conf = epoch_confusions[e]
        if conf is None:
            macro_recalls.append(np.nan)
            macro_precisions.append(np.nan)
            macro_f1s.append(np.nan)
            macro_specificities.append(np.nan)
            macro_sensitivities.append(np.nan)
            g_means.append(np.nan)
            maucs.append(np.nan)
            continue
        metrics = _compute_imbalanced_metrics_from_conf(conf)
        macro_recalls.append(metrics.get('macro_recall', np.nan))
        macro_precisions.append(metrics.get('macro_precision', np.nan))
        macro_f1s.append(metrics.get('macro_f1', np.nan))
        macro_specificities.append(metrics.get('macro_specificity', np.nan))
        macro_sensitivities.append(metrics.get('macro_sensitivity', np.nan))
        g_means.append(metrics.get('g_mean', np.nan))
        maucs.append(metrics.get('mauc', np.nan))

    epochs = indices
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, macro_recalls, label='Macro Recall')
    plt.plot(epochs, macro_precisions, label='Macro Precision')
    plt.plot(epochs, macro_f1s, label='Macro F1')
    plt.plot(epochs, macro_specificities, label='Macro Specificity')
    plt.plot(epochs, macro_sensitivities, label='Macro Sensitivity')
    plt.legend()
    plt.title('Imbalanced Learning Metrics (macro averages)')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, maucs, label='MAUC')
    plt.plot(epochs, g_means, label='G-Mean')
    plt.legend()
    plt.title('MAUC and G-Mean per epoch')
    plt.xlabel('Epoch')
    plt.show()


def plot_population_statistics(config_names, results_root='results', labels=None, metrics=None):
    """
    For multiple configs (each corresponding to a subfolder or filename token under `results_root`),
    aggregate runs (multiple .pt files) per config and plot mean +/- std shaded regions for selected metrics.

    Parameters:
      - config_names: list of config name substrings to search for under `results_root`.
      - results_root: folder where result subfolders/files are stored.
      - labels: optional list of labels for plotting (same length as config_names).
      - metrics: list of metric keys from history to aggregate; defaults to: 
        ['overall_acc', 'avg_inc_acc', 'plasticity', 'forgetting', 'bwt', 'fwt', 'intransience']
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    if labels is None:
        labels = config_names
    if metrics is None:
        metrics = ['overall_acc', 'avg_inc_acc', 'plasticity', 'forgetting', 'bwt', 'fwt', 'intransience']

    # find result files for each config name
    config_runs = {cn: [] for cn in config_names}
    for root, _, files in os.walk(results_root):
        for f in files:
            if not f.endswith('.pt'):
                continue
            path = os.path.join(root, f)
            for cn in config_names:
                if cn in root or cn in f:
                    config_runs[cn].append(path)

    # load histories and optionally epoch_confusions per run
    aggregated = {}
    for cn in config_names:
        runs = config_runs.get(cn, [])
        hist_arrays = {m: [] for m in metrics}
        imbalanced_arrays = {'mauc': [], 'g_mean': [], 'macro_recall': [], 'macro_f1': []}

        for run_path in runs:
            try:
                data = torch.load(run_path, weights_only=False)
            except Exception:
                continue
            hist = data.get('history', data)
            L = len(hist)
            for m in metrics:
                arr = np.full(L, np.nan)
                for i, h in enumerate(hist):
                    arr[i] = h.get(m, np.nan)
                hist_arrays[m].append(arr)

            # imbalanced metrics: compute per-epoch arrays (MAUC and g_mean)
            epoch_confusions = data.get('epoch_confusion_matrices', None)
            if epoch_confusions is not None:
                mauc_arr = np.full(len(epoch_confusions), np.nan)
                g_arr = np.full(len(epoch_confusions), np.nan)
                for i, c in enumerate(epoch_confusions):
                    if c is None:
                        continue
                    mets = _compute_imbalanced_metrics_from_conf(c)
                    mauc_arr[i] = mets.get('mauc', np.nan)
                    g_arr[i] = mets.get('g_mean', np.nan)
                imbalanced_arrays['mauc'].append(mauc_arr)
                imbalanced_arrays['g_mean'].append(g_arr)

        aggregated[cn] = {
            'hist_arrays': hist_arrays,
            'imbalanced': imbalanced_arrays,
            'n_runs': len(runs),
        }

    colors = plt.cm.tab10.colors

    # Plot each requested metric across configs
    for m in metrics:
        plt.figure(figsize=(10, 6))
        for idx, cn in enumerate(config_names):
            arrs = aggregated[cn]['hist_arrays'].get(m, [])
            if len(arrs) == 0:
                continue
            maxL = max(a.shape[0] for a in arrs)
            mat = np.full((len(arrs), maxL), np.nan)
            for i, a in enumerate(arrs):
                mat[i, :a.shape[0]] = a
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            x = np.arange(mean.shape[0])
            c = colors[idx % len(colors)]
            plt.plot(x, mean, label=labels[idx], color=c)
            plt.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
        plt.title(f"Population: {m}")
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    # Plot imbalanced MAUC and G-mean across configs
    for im in ['mauc', 'g_mean']:
        plt.figure(figsize=(10, 6))
        for idx, cn in enumerate(config_names):
            arrs = aggregated[cn]['imbalanced'].get(im, [])
            if len(arrs) == 0:
                continue
            maxL = max(a.shape[0] for a in arrs)
            mat = np.full((len(arrs), maxL), np.nan)
            for i, a in enumerate(arrs):
                mat[i, :a.shape[0]] = a
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            x = np.arange(mean.shape[0])
            c = colors[idx % len(colors)]
            plt.plot(x, mean, label=labels[idx], color=c)
            plt.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
        plt.title(f"Population: {im}")
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
