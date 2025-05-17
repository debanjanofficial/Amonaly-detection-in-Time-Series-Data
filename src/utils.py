import torch
import sys 

def get_device():
    """Checks for MPS or CUDA availability and returns the appropriate PyTorch device."""
    device_str = ""
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        if torch.backends.mps.is_built(): # Check if MPS is actually built and usable
            device = torch.device("mps")
            device_str = "MPS device found and built. Using MPS."
        else:
            device = torch.device("cpu")
            device_str = "MPS available but not built. Using CPU."
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "CUDA device found. Using CUDA."
    else:
        device = torch.device("cpu")
        device_str = "Neither MPS nor CUDA available. Using CPU."
    
    print(device_str) # Print the final decision once
    return device