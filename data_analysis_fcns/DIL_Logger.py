import os
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import re

class DIL_Logger:
    def __init__(self, N, C, baseline=None, save_dir="results", config_file=None):
        self.N = N  # number of domains
        self.C = C  # number of classes
        self.baseline = baseline  # baseline accuracy

        self.R_final = []
        self.best_past = np.zeros(N)

        # history of per-epoch metrics
        self.history = []
        # record when domains end (index in history)
        self.domain_boundaries = []

        # hold confusion matrices for every finalized domain
        self.confusions = []
        # confusion matrix after pretraining (single matrix) and for baseline models
        self.pretrain_confusion = []
        self.baseline_confusions = []
        # epoch-by-epoch confusion matrices (full test set evaluated each epoch)
        self.epoch_confusion_matrices = []
        # per-epoch expert usage history (list of lists)
        self.expert_usage_history = []
        # keep all intermediate R matrices (after each compute_metrics call)
        self.R_history = []

        # saving directory is fixed to results by default; callers may override
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.config_file = config_file

    @torch.no_grad()
    def evaluate(self, model, loaders, device):
        model.eval()

        domain_acc = []
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []
        expert_usage = None

        for loader in loaders:
            correct, total = 0, 0

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

            domain_acc.append(correct / total)

            # collect per-domain expert usage immediately after evaluating this loader
            if hasattr(model, "get_and_reset_usage_counts"):
                try:
                    if expert_usage is None:
                        expert_usage = []
                    # get usage counts for this domain (resets epoch counters in model)
                    domain_usage = model.get_and_reset_usage_counts()
                    expert_usage.append(domain_usage)
                except Exception:
                    pass

        overall_acc = total_correct / total_samples

        # (expert_usage already collected per-domain during loader loop)

        # collect cumulative usage if model exposes it
        try:
            if hasattr(model, "get_cumulative_usage"):
                self.expert_cumulative_usage = model.get_cumulative_usage()
        except Exception:
            pass

        return (
            np.array(domain_acc),
            overall_acc,
            torch.cat(all_preds),
            torch.cat(all_targets),
            expert_usage,
        )

    def compute_metrics(self, domain_id, epoch, domain_acc, preds=None, targets=None, expert_usage=None):
        i = domain_id

        # Average Accuracy (AA): mean of current row
        avg_acc = domain_acc.mean()

        # Construct temporary R matrix
        R_temp = (
            np.vstack(self.R_final + [domain_acc])
            if self.R_final
            else np.array([domain_acc])
        )
        # keep a copy of the current R matrix for later inspection/plotting
        self.R_history.append(R_temp.copy())

        # Average Incremental Accuracy (AIA): mean of current row over seen domains
        avg_inc_acc = np.nanmean(R_temp[i, :i+1])

        # Plasticity (PL): mean of diagonal
        diag = np.diag(R_temp)
        plasticity = diag.mean()

        # Forgetting & Backward transfer (FM & BWT)
        if i > 0:
            forgetting = np.mean(
                self.best_past[:i] - domain_acc[:i]
            )

            bwt = np.mean(
                domain_acc[:i] -
                np.diag(np.array(self.R_final))[:i]
            )
        else:
            forgetting = 0.0
            bwt = 0.0

        # Forward Transfer (FWT): compute per-epoch using R_temp as stand-in for final R
        fwt = None
        if self.baseline is not None and i > 0:
            vals = []
            for j in range(i):
                # R_temp may contain NaNs for unseen entries; guard access
                try:
                    r_val = R_temp[j][j + 1]
                except Exception:
                    r_val = np.nan
                # baseline should be a list/array with per-domain accuracies
                try:
                    baseline_val = self.baseline[j + 1]
                except Exception:
                    baseline_val = None
                if baseline_val is None:
                    continue
                vals.append(r_val - baseline_val)
            fwt = float(np.nanmean(vals)) if len(vals) > 0 else None

        # Intransience
        intrans = None
        if self.baseline is not None:
            intrans = self.baseline[i] - domain_acc[i]
        # if predictions/targets provided, compute and store epoch confusion
        if preds is not None and targets is not None:
            try:
                conf = torch.zeros(self.C, self.C)
                for t, p in zip(targets, preds):
                    conf[int(t), int(p)] += 1
                self.epoch_confusion_matrices.append(conf)
            except Exception:
                # if something goes wrong, append None to preserve epoch alignment
                self.epoch_confusion_matrices.append(None)

        # record expert usage if provided
        if expert_usage is not None:
            try:
                self.expert_usage_history.append(list(expert_usage))
            except Exception:
                self.expert_usage_history.append(None)

        return {
            "epoch": epoch,
            "domain": i,
            "overall_acc": float(avg_acc),
            "avg_acc": float(avg_acc),
            "avg_inc_acc": float(avg_inc_acc),
            "plasticity": float(plasticity),
            "forgetting": float(forgetting),
            "bwt": float(bwt),
            "fwt": None if fwt is None else float(fwt),
            "intransience": None if intrans is None else float(intrans),
            "domain_acc_vector": domain_acc.tolist(),
            "expert_usage": None if expert_usage is None else expert_usage,
        }

    def finalize_domain(self, domain_acc, preds, targets):
        """Call at the end of each domain.

        The old implementation persisted a per-domain confusion matrix to
        disk; we now accumulate everything in memory and let ``save`` dump a
        single consolidated file.
        """
        self.R_final.append(domain_acc.copy())
        self.best_past = np.maximum(self.best_past, domain_acc)

        # Confusion matrix for this domain
        conf = torch.zeros(self.C, self.C)
        for t, p in zip(targets, preds):
            conf[t.long(), p.long()] += 1
        self.confusions.append(conf)

        self.domain_boundaries.append(len(self.history))

    def log(self, metrics):
        self.history.append(metrics)

    def save(self):
        """Write all logged information to a single file.

        The filename encodes the model & dataset (if available) plus a
        timestamp in MMddhhmm format.  The resulting file is a pickled
        dictionary that contains the history, confusion matrices, R matrices
        and any metadata; loading it back provides everything needed to
        reproduce the plots.
        """
        # create common metadata
        meta = {"save_time": np.datetime64("now").astype(str)}
        model_name = "model"
        dataset_name = "dataset"
        if hasattr(self, "config_file") and self.config_file is not None:
            meta["config_file"] = self.config_file
            try:
                # Load but remove comment lines (JSON + comments)
                with open(self.config_file) as f:
                    raw = f.read()
                # remove line comments
                raw = re.sub(r"//.*", "", raw)
                # remove block comments
                raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
                cfg = json.loads(raw)
                
                model_name = cfg.get("model", {}).get("name", model_name)
                dataset_name = cfg.get("dataset", dataset_name)
            except Exception:
                pass

        now = datetime.now()
        stamp = now.strftime("%m%d%H%M")
        fname = f"{model_name}_{dataset_name}_{stamp}.pt"

        out_dir = self.save_dir
        if hasattr(self, 'config_file') and self.config_file:
            cfg_name = os.path.splitext(os.path.basename(self.config_file))[0]
            out_dir = out_dir / cfg_name
        out_dir.mkdir(parents=True, exist_ok=True)

        outpath = out_dir / fname

        save_dict = {
            "metadata": meta,
            "history": self.history,
            "R_final": np.array(self.R_final),
            "R_history": [r.tolist() for r in self.R_history],
            # convert confusions to cpu tensors for portability
            "confusion_matrices": [c.cpu() if torch.is_tensor(c) else c
                                   for c in self.confusions],
            # epoch-by-epoch full-test confusion matrices
            "epoch_confusion_matrices": [c.cpu() if torch.is_tensor(c) else c
                                          for c in self.epoch_confusion_matrices],
            # pretrain confusion (single matrix) and baseline (per-domain) confusions
            "pretrain_confusion": (self.pretrain_confusion.cpu() if torch.is_tensor(self.pretrain_confusion) else self.pretrain_confusion),
            "baseline_confusion_matrices": [c.cpu() if torch.is_tensor(c) else c
                                           for c in self.baseline_confusions],
            "expert_usage_history": self.expert_usage_history,
            # cumulative per-experiment usage if model provided it via logger hooks
            "expert_cumulative_usage": getattr(self, "expert_cumulative_usage", None),
            # seed information if available (populated by the experiment driver)
            "seeds": getattr(self, "seeds", None),
        }
        torch.save(save_dict, outpath)

        return outpath
