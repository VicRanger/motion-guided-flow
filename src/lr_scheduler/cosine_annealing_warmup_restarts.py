import math
import torch

from utils.str_utils import dict_to_string
from utils.log import log
from lr_scheduler.lr_scheduler_base import LRSchedulerBase

class CosineAnnealingWarmupRestarts(LRSchedulerBase):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_epoch : float,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_epoch : float = 0,
                 gamma : float = 1.,
                 last_epoch: int = -1
        ):
        assert warmup_epoch < first_cycle_epoch
        assert isinstance(max_lr, float) or isinstance(max_lr, int)
        assert isinstance(min_lr, float) or isinstance(min_lr, int)
        assert isinstance(warmup_epoch, float) or isinstance(warmup_epoch, int)
        assert isinstance(first_cycle_epoch, float) or isinstance(first_cycle_epoch, int)
        assert isinstance(cycle_mult, float) or isinstance(cycle_mult, int)
        assert isinstance(gamma, float) or isinstance(gamma, int)
        
        self.first_cycle_epoch = first_cycle_epoch # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_epoch = warmup_epoch # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_epoch = first_cycle_epoch # first cycle step size
        self.cycle = 0 # cycle count
        self.epoch_in_cycle = 0 # step size of the current cycle
        
        super().__init__(optimizer, last_epoch=last_epoch)
        
        # set learning rate min_lr
    
    def get_lr(self):
        assert self.last_epoch >= 0
        epoch = self.last_epoch
        if epoch >= self.first_cycle_epoch:
            # log.debug(dict_to_string([epoch, self.cycle_mult, self.first_cycle_epoch]))
            if self.cycle_mult == 1.:
                self.cycle = int(epoch / self.first_cycle_epoch)
                self.epoch_in_cycle = epoch - self.cycle * self.first_cycle_epoch
            else:
                n = int(math.log((epoch / self.first_cycle_epoch * (self.cycle_mult - 1) + 1), self.cycle_mult))
                self.cycle = n
                self.epoch_in_cycle = epoch - int(self.first_cycle_epoch * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                self.cur_cycle_epoch = self.first_cycle_epoch * self.cycle_mult ** (n)
        else:
            self.cur_cycle_epoch = self.first_cycle_epoch
            self.epoch_in_cycle = epoch
                
        if self.gamma != 1.0:
            # log.debug(dict_to_string([self.base_max_lr, self.gamma, self.gamma ** self.cycle, self.cycle]))
            self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
            self.max_lr = max(self.max_lr, self.min_lr)
            
        if self.epoch_in_cycle < self.warmup_epoch:
            return [(self.max_lr - self.min_lr) * self.epoch_in_cycle / self.warmup_epoch + self.min_lr for _ in self.base_lrs]
        else:
            return [self.min_lr + (self.max_lr - self.min_lr) \
                    * (1 + math.cos(math.pi * (self.epoch_in_cycle - self.warmup_epoch) \
                                    / (self.cur_cycle_epoch - self.warmup_epoch))) / 2
                    for _ in self.base_lrs]

            