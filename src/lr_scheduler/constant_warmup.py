import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

from lr_scheduler.lr_scheduler_base import LRSchedulerBase

class ConstantWarmup(LRSchedulerBase):
    """
        optimizer (Optimizer): Wrapped optimizer.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_epoch : float = 0,
                 decay_at_epoch: float = -1,
                 total_epoch: float = -1,
                 last_epoch: int = -1
        ):
        assert isinstance(max_lr, float) or isinstance(max_lr, int)
        assert isinstance(min_lr, float) or isinstance(min_lr, int)
        assert isinstance(warmup_epoch, float) or isinstance(warmup_epoch, int)
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_epoch = warmup_epoch # warmup step size
        self.decay_at_epoch = decay_at_epoch
        self.total_epoch = total_epoch
        if self.decay_at_epoch > 0:
            assert self.decay_at_epoch > self.warmup_epoch
            assert self.total_epoch > self.decay_at_epoch
        self.cur_epoch = -1
        super().__init__(optimizer, last_epoch=last_epoch)
        # set learning rate min_lr
        
    def get_lr(self):
        assert self.last_epoch >= 0
        self.cur_epoch = self.last_epoch
        if self.cur_epoch < self.warmup_epoch:
            ratio = self.cur_epoch / self.warmup_epoch
        elif self.decay_at_epoch >= 0 and self.cur_epoch >= self.decay_at_epoch:
            cur_epoch = min(self.total_epoch, self.cur_epoch)
            ratio = 0.5 * (1.0 + math.cos((cur_epoch - self.decay_at_epoch)/(self.total_epoch - self.decay_at_epoch) * math.pi))
        else:
            ratio = 1.0
        return [(self.max_lr - self.min_lr) * ratio + self.min_lr for _ in self.base_lrs]