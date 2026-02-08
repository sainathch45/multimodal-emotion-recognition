"""
Evaluation Metrics
Computes performance and efficiency metrics for multimodal emotion recognition
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import time


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        predictions: [N] predicted class indices
        targets: [N] true class indices
        num_classes: Number of classes
        class_names: Optional list of class names
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, predictions)
    
    # Macro F1 (unweighted average across classes)
    metrics['macro_f1'] = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # Weighted F1 (weighted by support)
    metrics['weighted_f1'] = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Micro F1 (global)
    metrics['micro_f1'] = f1_score(targets, predictions, average='micro', zero_division=0)
    
    # UAR (Unweighted Average Recall) - common in emotion recognition
    recalls = recall_score(targets, predictions, average=None, zero_division=0)
    metrics['uar'] = np.mean(recalls)
    
    # Per-class metrics
    per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
    per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
    
    if class_names is not None:
        for idx, name in enumerate(class_names):
            metrics[f'f1_{name}'] = per_class_f1[idx]
            metrics[f'precision_{name}'] = per_class_precision[idx]
            metrics[f'recall_{name}'] = per_class_recall[idx]
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
    
    return metrics


def compute_efficiency_metrics(
    model: torch.nn.Module,
    input_sample: Dict[str, torch.Tensor],
    device: str = 'cpu',
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Compute model efficiency metrics
    
    Args:
        model: PyTorch model
        input_sample: Sample input dictionary
        device: Device to run on
        n_iterations: Number of iterations for latency measurement
    
    Returns:
        Dictionary of efficiency metrics
    """
    model = model.to(device)
    model.eval()
    
    metrics = {}
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    metrics['params_mb'] = (total_params * 4) / (1024 ** 2)  # Assuming float32
    
    # FLOPs calculation (requires thop)
    try:
        from thop import profile, clever_format
        
        # Prepare input for thop
        sample_inputs = tuple(input_sample.values())
        
        flops, _ = profile(model, inputs=sample_inputs, verbose=False)
        metrics['flops'] = flops
        metrics['gflops'] = flops / 1e9
    except ImportError:
        print("Warning: thop not available for FLOPs calculation")
        metrics['flops'] = None
    
    # Latency measurement
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(**input_sample)
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(n_iterations):
            _ = model(**input_sample)
            if device == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / n_iterations * 1000  # ms
        metrics[f'latency_{device}_ms'] = avg_latency
        metrics[f'throughput_{device}_fps'] = 1000 / avg_latency
    
    # Memory usage (approximate)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        _ = model(**input_sample)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        metrics['peak_memory_mb'] = peak_memory
    
    return metrics


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics in table format"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"{key:30s}: {value:>10.4f}")
            else:
                print(f"{key:30s}: {value:>10,}")
        elif isinstance(value, np.ndarray):
            print(f"{key:30s}:")
            print(value)
    
    print(f"{'='*60}\n")


# Placeholder for remaining implementation
# TODO: Add more specific evaluation functions
