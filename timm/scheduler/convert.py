from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler

def convert_scheduler_from_epoch_to_iter(lr_scheduler, num_iter_per_epoch):
    if hasattr(lr_scheduler, 't_in_epochs'):
        lr_scheduler.t_in_epochs = False

    lr_scheduler.warmup_t = num_iter_per_epoch * lr_scheduler.warmup_t
    lr_scheduler.warmup_steps = [(v - lr_scheduler.warmup_lr_init) / lr_scheduler.warmup_t for v in lr_scheduler.base_values]


    if isinstance(lr_scheduler, CosineLRScheduler):
        lr_scheduler.t_initial = num_iter_per_epoch * lr_scheduler.t_initial
    elif isinstance(lr_scheduler, TanhLRScheduler):
        lr_scheduler.t_initial = num_iter_per_epoch * lr_scheduler.t_initial
    elif isinstance(lr_scheduler, StepLRScheduler):
        lr_scheduler.decay_t = num_iter_per_epoch * lr_scheduler.decay_t
    elif isinstance(lr_scheduler, MultiStepLRScheduler):
        lr_scheduler.decay_t = [num_iter_per_epoch * t for t in lr_scheduler.decay_t]
    elif isinstance(lr_scheduler, PlateauLRScheduler):
        lr_scheduler.patience_t = num_iter_per_epoch * lr_scheduler.patience_t
    elif isinstance(lr_scheduler, PolyLRScheduler):
        lr_scheduler.t_initial = num_iter_per_epoch * lr_scheduler.t_initial
    elif lr_scheduler is not None:
        raise NotImplementedError

    return lr_scheduler



