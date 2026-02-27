import torch
import numpy as np
from pathlib import Path
import json


class DIL_Logger:
    def __init__(self, N, C, baseline=None, save_dir="logs", config_file=None):
        self.N = N
        self.C = C
        self.baseline = baseline

        self.R_final = []
        self.best_past = np.zeros(N)

        self.history = []
        self.domain_boundaries = []

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

            domain_acc.append(100 * correct / total)

        overall_acc = 100 * total_correct / total_samples

        return (
            np.array(domain_acc),
            overall_acc,
            torch.cat(all_preds),
            torch.cat(all_targets),
        )

    def compute_metrics(self, domain_id, epoch, domain_acc):
        i = domain_id

        # Average Accuracy (AA): mean of current row
        avg_acc = domain_acc.mean()

        # Construct temporary R matrix
        R_temp = (
            np.vstack(self.R_final + [domain_acc])
            if self.R_final
            else np.array([domain_acc])
        )

        # Average Incremental Accuracy (AIA): mean of current row over seen domains
        avg_inc_acc = np.nanmean(R_temp[i, :i+1])

        # Plasticity (PL): mean of diagonal
        diag = np.diag(R_temp)
        plasticity = diag.mean()

        # Forgetting & BWT (FM & BWT)
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

        # FWT
        fwt = None
        if self.baseline is not None and i > 0:
            fwt = np.mean([
                self.R_final[j][j + 1] - self.baseline[j + 1]
                for j in range(i)
            ])

        # Intransigence
        intrans = None
        if self.baseline is not None:
            intrans = self.baseline[i] - domain_acc[i]

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
            "intransigence": None if intrans is None else float(intrans),
            "domain_acc_vector": domain_acc.tolist(),
        }

    def finalize_domain(self, domain_acc, preds, targets):
        self.R_final.append(domain_acc.copy())
        self.best_past = np.maximum(self.best_past, domain_acc)

        # Confusion matrix
        conf = torch.zeros(self.C, self.C)
        for t, p in zip(targets, preds):
            conf[t.long(), p.long()] += 1

        torch.save(conf, self.save_dir / f"conf_domain_{len(self.R_final)-1}.pt")

        self.domain_boundaries.append(len(self.history))

    def log(self, metrics):
        self.history.append(metrics)

    def save(self):
        # include metadata such as configuration path and timestamp if available
        meta = {"save_time": np.datetime64("now").astype(str)}
        if hasattr(self, "config_file") and self.config_file is not None:
            meta["config_file"] = self.config_file
        out = {
            "metadata": meta,
            "history": self.history
        }
        # if history already had config info it can override; caller may add it before save
        with open(self.save_dir / "metrics.json", "w") as f:
            json.dump(out, f, indent=2)

        np.save(self.save_dir / "R_final.npy",
                np.array(self.R_final))
