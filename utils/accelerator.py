import random

import numpy as np
import torch


def _load_torch_npu():
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return True


def get_device():
    _load_torch_npu()
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def maybe_data_parallel(model, device):
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model
