from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            if not self.finished:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.finished = True
            return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.finished and epoch is None:
            self.after_scheduler.step(None)
        else:
            super(WarmupScheduler, self).step(epoch)
            if self.finished:
                self.after_scheduler.step(epoch - self.warmup_steps if epoch is not None else None)
