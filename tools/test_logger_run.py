from experiment_fcns.setup import load_config
from data_analysis_fcns.DIL_Logger import DIL_Logger

if __name__ == '__main__':
    cfg_file = 'configs/example_config.jsonc'
    print('Loading config...')
    setup = load_config(cfg_file)
    cfg = setup['cfg']
    device = setup['device']
    model = setup['model'].to(device)
    test_loaders = setup['test_loaders']
    train_loaders = setup['train_loaders']

    logger = DIL_Logger(N=len(train_loaders), C=cfg.get('model', {}).get('num_classes', 10), baseline=None, config_file=cfg_file)
    logger.seeds = setup.get('resolved_seeds', None)

    print('Running one evaluation...')
    domain_acc, overall_acc, preds, targets, expert_usage = logger.evaluate(model, test_loaders, device)
    print('Eval done. Overall acc:', overall_acc)

    metrics = logger.compute_metrics(0, 0, domain_acc, preds=preds, targets=targets, expert_usage=expert_usage)
    metrics['overall_acc'] = float(overall_acc)
    logger.log(metrics)

    out = logger.save()
    print('Saved logger to:', out)
