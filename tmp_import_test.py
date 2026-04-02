import importlib
m = importlib.import_module('models_classes.moe_vit')
print('import_ok')
print('moe_version:', getattr(m, '__name__', 'unknown'))
