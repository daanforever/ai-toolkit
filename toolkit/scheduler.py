import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


class SequentialLRWrapper(torch.optim.lr_scheduler.SequentialLR):
    """
    Wrapper for SequentialLR that ignores extra arguments to step().

    Cosine+warmup resume uses `_seek_sequential_warmup_cosine` at create time
    (via `resume_step`), not absolute `step(step_num)`. This wrapper remains so
    callers that still pass args (legacy pre-loop / train loop) do not crash —
    SequentialLR.step() accepts no arguments.
    """
    def step(self, *args, **kwargs):
        # Ignore all arguments and call parent step() without them
        super().step()


def _seek_sequential_warmup_cosine(scheduler, target_last_epoch: int) -> None:
    """
    Set SequentialLR(warmup, cosine) to the state after `target_last_epoch`
    advances from init (matches uninterrupted create + N× step()).
    """
    warmup_end = scheduler._milestones[0]
    warm, main = scheduler._schedulers
    scheduler.last_epoch = target_last_epoch

    if target_last_epoch < warmup_end:
        warm._update_lr(target_last_epoch)
    else:
        warm.last_epoch = warmup_end - 1
        main._update_lr(target_last_epoch - warmup_end)

    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]


def _create_scheduler_with_warmup(
        scheduler_type: str,
        optimizer: torch.optim.Optimizer,
        **scheduler_kwargs
):
    """
    Creates a scheduler with optional warmup period using SequentialLR.
    
    Args:
        scheduler_type: 'cosine' or 'cosine_with_restarts'
        optimizer: The optimizer to schedule
        scheduler_kwargs: Parameters for the scheduler. Can include:
            - warmup_steps: Number of warmup steps (default: 0, no warmup)
            - total_iters: TOTAL number of iterations INCLUDING warmup (default: 1000)
            - resume_step: Training step to seek to after create (default: 0)
            - T_0/T_max: Iterations for MAIN scheduler, overrides calculation from total_iters
            - Other scheduler parameters (T_mult, eta_min, etc.)
    
    Semantics:
        - total_iters: Total training iterations (warmup + main scheduler)
        - T_0/T_max: Main scheduler iterations (if specified, total_iters is ignored)
        - If only total_iters specified: main_iters = total_iters - warmup_steps
        - If T_0/T_max specified: main_iters = T_0/T_max (total_iters ignored)
        - resume_step: after create, seek so SequentialLR.last_epoch == resume_step + 1
          (same as legacy pre-loop step() once when resume_step == 0)
    
    Returns:
        A scheduler (SequentialLR if warmup_steps > 0, otherwise base scheduler)
    """
    # Must pop before CosineAnnealingLR(**kwargs)
    resume_step = int(scheduler_kwargs.pop('resume_step', 0))

    # Extract warmup_steps (default: 0, no warmup)
    warmup_steps = scheduler_kwargs.pop('warmup_steps', 0)
    
    # Extract total_iters (GENERAL total, including warmup)
    total_iters = scheduler_kwargs.pop('total_iters', 1000)
    
    # Calculate main scheduler iterations
    # T_0/T_max have priority and specify main scheduler iterations directly
    if scheduler_type == "cosine":
        if 'T_max' in scheduler_kwargs:
            # T_max specifies main scheduler iterations (ignores total_iters)
            main_total_iters = scheduler_kwargs.pop('T_max')
        else:
            # Calculate from total_iters
            main_total_iters = total_iters - warmup_steps
    elif scheduler_type == "cosine_with_restarts":
        if 'T_0' in scheduler_kwargs:
            # T_0 specifies main scheduler iterations (ignores total_iters)
            main_total_iters = scheduler_kwargs.pop('T_0')
        else:
            # Calculate from total_iters
            main_total_iters = total_iters - warmup_steps
    
    # Validation: warn if configuration seems incorrect
    if main_total_iters <= 0:
        raise ValueError(
            f"Main scheduler iterations must be positive, got {main_total_iters}. "
            f"Check your total_iters ({total_iters}) and warmup_steps ({warmup_steps})."
        )
    if warmup_steps > 0 and warmup_steps >= total_iters:
        print(f"WARNING: warmup_steps ({warmup_steps}) >= total_iters ({total_iters}). "
              f"The main scheduler will have very few or no iterations.")
    
    if warmup_steps <= 0:
        # No warmup: resume via BaseSDTrainProcess pre-loop step(step_num)
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_total_iters, **scheduler_kwargs
            )
        elif scheduler_type == "cosine_with_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=main_total_iters, **scheduler_kwargs
            )
  
    # Warmup: linear from eta_min to base_lr (explicit, not tied to optimizer initial_lr)
    base_lr = optimizer.param_groups[0]['lr']
    eta_min = scheduler_kwargs.get('eta_min', base_lr / 10.0)
    print(f"Using LR {base_lr} for warmup (eta_min={eta_min})")
    print(f"Param groups LR: {[g['lr'] for g in optimizer.param_groups]}")

    def _warmup_lambda(step: int) -> float:
        step = max(0, step)  # guard for last_epoch=-1 before first step()
        if warmup_steps <= 1:
            return 1.0
        t = min(step, warmup_steps - 1) / (warmup_steps - 1)
        return eta_min / base_lr + (1.0 - eta_min / base_lr) * t

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_warmup_lambda,
        last_epoch=-1
    )
    
    # Create main scheduler
    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_total_iters, **scheduler_kwargs
        )
    elif scheduler_type == "cosine_with_restarts":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=main_total_iters, **scheduler_kwargs
        )
    
    # Combine schedulers using SequentialLRWrapper
    combined_scheduler = SequentialLRWrapper(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )

    # Seek to match start of training step `resume_step`
    # (uninterrupted: create → step() → … → last_epoch == resume_step + 1)
    _seek_sequential_warmup_cosine(combined_scheduler, resume_step + 1)
    
    return combined_scheduler


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        # All parameters passed via kwargs, handled in _create_scheduler_with_warmup
        return _create_scheduler_with_warmup(
            "cosine", optimizer, **kwargs
        )
    elif name == "cosine_with_restarts":
        # All parameters passed via kwargs, handled in _create_scheduler_with_warmup
        return _create_scheduler_with_warmup(
            "cosine_with_restarts", optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == "cyclic":
        # CyclicLR scheduler: cycles between base_lr and max_lr
        default_step_size_up = kwargs.get('step_size_up', 2000)
        default_step_size_down = kwargs.get('step_size_down', default_step_size_up)

        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-7),
            max_lr=kwargs.get('max_lr', 1e-4),
            step_size_up=kwargs.get('step_size_up', 2000),
            step_size_down=kwargs.get('step_size_down', default_step_size_down),
            mode=kwargs.get('mode', 'triangular'),
            gamma=kwargs.get('gamma', 1.0),
            scale_fn=kwargs.get('scale_fn', None),
            scale_mode=kwargs.get('scale_mode', 'cycle'),
            cycle_momentum=kwargs.get('cycle_momentum', False),
            base_momentum=kwargs.get('base_momentum', 0.85),
            max_momentum=kwargs.get('max_momentum', 0.95),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    elif name == 'constant_with_warmup':
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear, constant or cyclic"
        )
