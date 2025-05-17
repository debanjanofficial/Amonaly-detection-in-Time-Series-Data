import torch
import sys

def get_device():
    """Checks for MPS availability and returns the appropriate PyTorch device."""
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found. Using MPS.")
        device = torch.device("cpu")
        print("MPS (or CUDA) device not found. Using CPU.")
    return device
