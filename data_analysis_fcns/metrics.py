import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def compute_continual_metrics(test_history: list, reference: dict = None) -> Dict[str, Any]:
    """Compute continual-learning summary metrics from `test_history`.

    Parameters
    - test_history: list of test-result dicts produced by the project's `test` helpers.
    - reference: optional. If provided, should be either a `test_history`-style list
      or a dict containing an `R` entry (accuracy matrix). When `reference` is
      supplied the function will also compute Intransience (IM) and Forward Transfer (FWT).

    Notes / definitions used here (explicit so behaviour is deterministic):
    - R (accuracy matrix): R[i, j] = accuracy on task j after learning step i.
    - AIA: average incremental accuracy computed from the R matrix (always returned
      when `test_history` is available).
    - BWT / FM: backward transfer and forgetting computed from R (always returned
      when possible).
    - IM (Intransience): computed only when a `reference` is provided. IM =
      mean_j (ref_final_acc[j] - R[j, j]) where `ref_final_acc` is taken from the
      reference final evaluation (reference R last row).
    - FWT (Forward Transfer): computed only when a `reference` is provided. Here
      FWT = mean_{j>=1} (R[j-1, j] - ref_final_acc[j]) (see docstring); this is a
      reference-based forward-transfer measure and will be None when no reference
      is supplied.

    Returns a dict containing the R matrix, AIA, FM, BWT and — only when a
    reference is provided — IM and FWT. If a metric cannot be computed it will
    be returned as None.
    """
    all_classes = sorted({
        cls for t in test_history for cls in t.get('overall', {}).get('per_class_accuracies', {}).keys()
    }) if test_history else []

    num_steps = len(test_history)
    num_tasks = len(all_classes)

    def build_R(from_history):
        if not from_history:
            return np.empty((0, 0))
        cols = sorted({
            cls for t in from_history for cls in t.get('overall', {}).get('per_class_accuracies', {}).keys()
        })
        if not cols:
            return np.empty((0, 0))
        col_map = {c: i for i, c in enumerate(cols)}
        Rm = np.full((len(from_history), len(cols)), np.nan)
        for step_idx, t in enumerate(from_history):
            for cls, acc in t.get('overall', {}).get('per_class_accuracies', {}).items():
                Rm[step_idx, col_map[cls]] = acc
        return Rm

    if num_steps > 0 and num_tasks > 0:
        class_to_col = {c: i for i, c in enumerate(all_classes)}
        R = np.full((num_steps, num_tasks), np.nan)
        for step_idx, t in enumerate(test_history):
            for cls, acc in t.get('overall', {}).get('per_class_accuracies', {}).items():
                R[step_idx, class_to_col[cls]] = acc
    else:
        R = np.empty((0, 0))

    # Average Incremental Accuracy (AIA)
    aia_per_step = []
    if R.size:
        for k in range(R.shape[0]):
            row = R[k, :k+1]  # j <= k
            aia_per_step.append(np.nanmean(row))
        aia_final = aia_per_step[-1]
    else:
        aia_final = np.nan

    # Forgetting / Backward Transfer (FM / BWT) - unchanged behaviour
    fm = None
    bwt = None
    if R.size and R.shape[0] > 1:
        final_row = R[-1, :]
        diag = np.array([R[i, i] for i in range(min(R.shape[0], R.shape[1]))])
        max_task = min(R.shape[0], R.shape[1]) - 1
        if max_task > 0:
            valid_idx = [i for i in range(max_task) if (not np.isnan(diag[i]) and not np.isnan(final_row[i]))]
            if valid_idx:
                forgetting_vals = [diag[i] - final_row[i] for i in valid_idx]
                fm = np.mean(forgetting_vals)
                bwt_vals = [final_row[i] - diag[i] for i in valid_idx]
                bwt = np.mean(bwt_vals)

    # IM / FWT: only computed when a reference is provided (per user's request)
    im = None
    fwt = None
    if reference is not None:
        # accept either a reference test_history (list) or dict with 'R'
        if isinstance(reference, list):
            R_ref = build_R(reference)
        elif isinstance(reference, dict) and 'R' in reference and isinstance(reference['R'], np.ndarray):
            R_ref = reference['R']
        elif isinstance(reference, dict) and 'test_history' in reference:
            R_ref = build_R(reference['test_history'])
        else:
            # best-effort: try to build from a dict of per-step 'overall' entries
            try:
                R_ref = build_R(reference.get('test_history', None) if isinstance(reference, dict) else None)
            except Exception:
                R_ref = np.empty((0, 0))

        # ref_final_acc: final evaluation row from reference (if available)
        if R_ref.size:
            ref_final = R_ref[-1, :]
        else:
            ref_final = np.array([np.nan] * num_tasks)

        # Intransience (IM): mean_j (ref_final_acc[j] - R[j,j]) for valid entries
        diag_idxs = list(range(min(R.shape[0], R.shape[1]))) if R.size else []
        im_vals = []
        for j in diag_idxs:
            if j < ref_final.shape[0] and not np.isnan(ref_final[j]) and not np.isnan(R[j, j]):
                im_vals.append(ref_final[j] - R[j, j])
        im = float(np.mean(im_vals)) if im_vals else None

        # Forward Transfer (FWT): mean_{j>=1} (R[j-1, j] - ref_final_acc[j]) where defined
        fwt_vals = []
        if R.size:
            for j in range(1, min(R.shape[1], R.shape[0])):
                pre_acc = R[j-1, j]
                ref_acc = ref_final[j] if j < ref_final.shape[0] else np.nan
                if (not np.isnan(pre_acc)) and (not np.isnan(ref_acc)):
                    fwt_vals.append(pre_acc - ref_acc)
        fwt = float(np.mean(fwt_vals)) if fwt_vals else None

    return {
        'R': R,
        'all_classes': all_classes,
        'aia_per_step': aia_per_step,
        'aia_final': aia_final,
        'fm': fm,
        'bwt': bwt,
        'im': im,
        'fwt': fwt,
        'test_history': test_history
    }


def print_continual_summary(test_history: list):
    if not test_history:
        print('No test history available')
        return
    final = test_history[-1]
    overall = final.get('overall', {})
    print('=' * 80)
    print('Final test summary')
    print('=' * 80)
    print(f"Overall Accuracy: {overall.get('accuracy', 0):>0.2f}%")
    print(f"Average Task Accuracy: {overall.get('average_task_accuracy', 0):>0.2f}%")
    print('\nPer-Class Accuracy:')
    for cls, acc in sorted(overall.get('per_class_accuracies', {}).items()):
        print(f"  Class {cls}: {acc:>0.2f}%")


def plot_confusion_matrix(confusion_mat: dict):
    if not confusion_mat:
        return
    all_classes_conf = sorted(set(
        list(confusion_mat.keys()) +
        [item for subdict in confusion_mat.values() for item in subdict.keys()]
    ))
    cm_array = np.zeros((len(all_classes_conf), len(all_classes_conf)))
    for true_cls in confusion_mat:
        for pred_cls in confusion_mat[true_cls]:
            true_idx = all_classes_conf.index(true_cls)
            pred_idx = all_classes_conf.index(pred_cls)
            cm_array[true_idx, pred_idx] = confusion_mat[true_cls][pred_cls]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=all_classes_conf, yticklabels=all_classes_conf)
    plt.title('Confusion Matrix - Final Evaluation')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.show()


def plot_metrics_over_time(test_history: list):
    if len(test_history) <= 1:
        return
    overall_accs = [t.get('overall', {}).get('accuracy', 0) for t in test_history]
    avg_task_accs = [t.get('overall', {}).get('average_task_accuracy', 0) for t in test_history]
    avg_losses = [t.get('overall', {}).get('avg_loss', 0) for t in test_history]
    steps = [t.get('step', i) for i, t in enumerate(test_history)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, overall_accs, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Overall Accuracy (%)')
    axes[0].set_title('Overall Accuracy Over Time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, avg_task_accs, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Average Task Accuracy (%)')
    axes[1].set_title('Average Task Accuracy Over Time')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, avg_losses, marker='^', linewidth=2, markersize=8, color='red')
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Average Loss')
    axes[2].set_title('Average Loss Over Time')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_moe_utilization(model) -> list:
    """Return per-MoE-layer cumulative utilization (fractions) if available."""
    layers = []
    for idx, m in enumerate(model.modules()):
        if hasattr(m, 'get_cumulative_stats'):
            s = m.get_cumulative_stats()
            layers.append({'layer_index': idx, 'fraction': s.get('fraction', []), 'samples': s.get('samples', 0)})
    return layers
