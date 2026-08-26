"""
Device selection for PyTorch training. Auto-detects Apple Silicon MPS,
CUDA, or falls back to CPU — so the same code runs on your Apple Silicon
and any future CUDA machine without edits.
"""

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Quick sanity check: run a small matmul on the selected device
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"Test matmul on {device} successful. Result shape: {z.shape}")