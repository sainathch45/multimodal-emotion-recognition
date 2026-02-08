import torch

def add_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def zero_out(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)
