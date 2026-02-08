import yaml
from pathlib import Path

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        return {
            'training': {'batch_size': 32, 'epochs': 10, 'lr': 1e-3, 'seed': 42},
            'loss_weights': {'rec': 0.2, 'contrast': 0.3}
        }
    with open(p, 'r') as f:
        return yaml.safe_load(f)
