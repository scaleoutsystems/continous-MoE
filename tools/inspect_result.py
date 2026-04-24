import torch
import pprint
p = 'results/mini_debug_vit_imagelevel/vit_moe_imagelevel_cifar10_04241501.pt'
try:
    d = torch.load(p)
    keys = sorted(list(d.keys()))
    print('keys:', keys)
    print('expert_usage_history' in d)
    print('expert_cumulative_usage' in d)
    eu = d.get('expert_usage_history', None)
    ec = d.get('expert_cumulative_usage', None)
    print('expert_usage_history length:', None if eu is None else len(eu))
    print('expert_cumulative_usage:', ec)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('LOAD_FAIL')
