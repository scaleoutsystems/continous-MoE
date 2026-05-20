import sys
import pprint

try:
    import torch
except Exception as e:
    print('Could not import torch:', e)
    sys.exit(1)

p = sys.argv[1] if len(sys.argv) > 1 else r'c:\Users\naido\Documents\ChalmersCourses\Thesis\continous-MoE\results\officehome-vit-modeltransfer\vit_moe_officehome_05162159.pt'
print('Inspecting:', p)
try:
    data = torch.load(p, weights_only=False)
except Exception as e:
    print('Error loading file:', e)
    sys.exit(1)

print('\nType of loaded object:', type(data))
if isinstance(data, dict):
    print('Top-level keys:', list(data.keys()))
else:
    print('Top-level repr (truncated):', repr(data)[:500])

hist = data.get('history', data)
print('\nHistory type:', type(hist))
if isinstance(hist, list):
    print('History length:', len(hist))
    print('\nSample epochs (first 5):')
    for i, h in enumerate(hist[:5]):
        print('\n--- epoch', i, '---')
        if isinstance(h, dict):
            keys = list(h.keys())
            print('keys:', keys)
            for k in ['domain', 'domain_acc_vector', 'overall_acc', 'avg_inc_acc', 'epoch']:
                if k in h:
                    print(f"  {k}: {h[k]}")
        else:
            print('epoch not a dict; repr:', repr(h)[:500])
else:
    print('history is not a list; repr (truncated):', repr(hist)[:1000])

# Try to extract domain boundaries if possible
if isinstance(hist, list) and len(hist) > 1 and isinstance(hist[0], dict) and 'domain' in hist[0]:
    boundaries = [i for i in range(1, len(hist)) if hist[i].get('domain') != hist[i-1].get('domain')]
    print('\nExtracted domain boundaries (epoch indices):', boundaries)
else:
    print('\nNo domain field found in history entries; cannot extract boundaries automatically.')
