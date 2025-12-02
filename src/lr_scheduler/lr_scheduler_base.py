import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, _enable_get_lr_call
import warnings

class LRSchedulerBase(_LRScheduler):
    pass