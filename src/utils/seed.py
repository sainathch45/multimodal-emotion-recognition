"""
Reproducibility Utilities
Ensures deterministic training and evaluation
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Use deterministic algorithms (slower but reproducible)
        benchmark: Use cudnn benchmark (faster but non-deterministic)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
    
    print(f"Random seed set to: {seed}")
    print(f"Deterministic mode: {deterministic}")
    print(f"CuDNN benchmark: {benchmark}")


def get_random_state():
    """Get current random state for all libraries"""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_random_state(state):
    """Restore random state"""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


if __name__ == '__main__':
    # Test seed setting
    set_seed(42, deterministic=True)
    
    print("\nTesting reproducibility:")
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    # Reset and test again
    set_seed(42, deterministic=True)
    print("\nAfter reset:")
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
