import torch
import numpy as np
from data_analysis_fcns.DIL_Logger import DIL_Logger

if __name__ == '__main__':
    logger = DIL_Logger(N=3, C=10, baseline=None, config_file=None, save_dir='results')
    # synthetic domain_acc
    domain_acc = np.array([0.5, 0.6, 0.7])
    # synthetic preds/targets
    preds = torch.randint(0, 10, (200,))
    targets = torch.randint(0, 10, (200,))

    metrics = logger.compute_metrics(0, 0, domain_acc, preds=preds, targets=targets)
    metrics['overall_acc'] = float((preds == targets).float().mean().item())
    logger.log(metrics)

    # attach seed info
    logger.seeds = {'global': None, 'pretrain': None, 'training': None, 'baseline': None}

    out = logger.save()
    print('Saved logger to:', out)
