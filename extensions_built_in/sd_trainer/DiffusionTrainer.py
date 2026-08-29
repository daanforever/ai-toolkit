from collections import OrderedDict
import json
import math
import os
import sqlite3
import asyncio
import concurrent.futures
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.accelerator import unwrap_model
from toolkit.data_loader import get_dataloader_datasets
from toolkit.print import print_acc
from toolkit.util.debug import is_debug_enabled
from typing import Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar
import threading
import time
import signal

AITK_Status = Literal["running", "stopped", "error", "completed"]
T = TypeVar("T")
RuntimeScaleLrMaskStatus = Literal["absent", "invalid", "ok"]


class RuntimeScaleLrMaskRead(NamedTuple):
    """Typed outcome of reading runtime_scale_lr_mask from SQLite.

    - status "ok": value is a list of strings (possibly empty [])
    - status "absent": column missing, NULL, or blank; value is None
    - status "invalid": malformed or partially invalid JSON; value is None
    """

    status: RuntimeScaleLrMaskStatus
    value: Optional[List[str]] = None


class DiffusionTrainer(SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(DiffusionTrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        self.is_ui_trainer = True
        if not os.path.exists(self.sqlite_db_path):
            self.is_ui_trainer = False
        else:
            print(f"Using SQLite database at {self.sqlite_db_path}")
        if self.job_id is None:
            self.is_ui_trainer = False
        else:
            print(f"Job ID: \"{self.job_id}\"")
        
        if self.is_ui_trainer:
            self.is_stopping = False
            # Create a thread pool for database operations
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # Track all async tasks
            self._async_tasks = []
            self._last_applied_runtime_lr = None
            self._last_applied_runtime_gaussian_mean = None
            self._last_applied_runtime_gaussian_std = None
            self._last_applied_runtime_gaussian_mean_2 = None
            self._last_applied_runtime_gaussian_std_2 = None
            self._last_applied_runtime_weight_decay = None
            self._last_applied_runtime_weight_decay_increment = None
            self._last_applied_runtime_weight_decay_mode = None
            self._last_applied_runtime_beta1 = None
            self._last_applied_runtime_beta2 = None
            self._last_applied_runtime_content_or_style = None
            self._last_applied_runtime_timestep_type = None
            self._last_applied_runtime_timestep_weighting = None
            self._last_applied_runtime_network_weights: Optional[tuple] = None
            self._last_applied_runtime_prompts: Optional[tuple] = None
            self._last_applied_runtime_batch_size = None
            self._last_applied_runtime_gradient_accumulation = None
            self._last_applied_runtime_save_every = None
            self._last_applied_runtime_sample_every = None
            self._last_applied_runtime_warmup_steps = None
            self._last_applied_runtime_warmup_boost = None
            self._last_applied_runtime_scale_lr_by_index = None
            self._last_applied_runtime_scale_lr_config: Optional[
                Tuple[float, float, Tuple[str, ...]]
            ] = None
            self._last_applied_runtime_min_snr_gamma = None
            self._last_applied_runtime_debug = None
            self._last_applied_runtime_fc_key: Optional[
                Tuple[
                    Optional[Tuple[float, ...]],
                    Optional[int],
                    Optional[Tuple[float, ...]],
                    Optional[float],
                ]
            ] = None
            self._last_applied_runtime_turbo_prior_steps = None
            self._last_applied_runtime_turbo_t_jitter = None
            self._last_applied_runtime_turbo_teacher_weight = None
            # Initialize the status
            self._run_async_operation(self._update_status("running", "Starting"))
            self._stop_watcher_started = False
            # self.start_stop_watcher(interval_sec=2.0)
    
    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
        if not self.is_ui_trainer:
            return
        if getattr(self, "_stop_watcher_started", False):
            return
        self._stop_watcher_started = True
        t = threading.Thread(
            target=self._stop_watcher_thread, args=(interval_sec,), daemon=True
        )
        t.start()

    def _stop_watcher_thread(self, interval_sec: float):
        while True:
            try:
                if self.should_stop():
                    # Mark and update status (non-blocking; uses existing infra)
                    self.is_stopping = True
                    self._run_async_operation(
                        self._update_status("stopped", "Job stopped (remote)")
                    )
                    # Best-effort flush pending async ops
                    try:
                        asyncio.run(self.wait_for_all_async())
                    except RuntimeError:
                        pass
                    # Try to stop DB thread pool quickly
                    try:
                        self.thread_pool.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        self.thread_pool.shutdown(wait=False)
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    os.kill(os.getpid(), signal.SIGINT)
                time.sleep(interval_sec)
            except Exception:
                time.sleep(interval_sec)

    def _run_async_operation(self, coro):
        """Helper method to run an async coroutine and track the task."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create a task and track it
        if loop.is_running():
            task = asyncio.run_coroutine_threadsafe(coro, loop)
            self._async_tasks.append(asyncio.wrap_future(task))
        else:
            task = loop.create_task(coro)
            self._async_tasks.append(task)
            loop.run_until_complete(task)

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, operation_func)

    def _db_connect(self):
        """Create a new connection for each operation to avoid locking."""
        conn = sqlite3.connect(self.sqlite_db_path, timeout=10.0)
        conn.isolation_level = None  # Enable autocommit mode
        return conn

    def _get_runtime_scalar(self, column_name: str, caster: Callable[[object], T]) -> Optional[T]:
        """Read one runtime scalar column from RuntimeParams for current job."""
        if not self.is_ui_trainer:
            return None
        with self._db_connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {column_name} FROM RuntimeParams WHERE jobId = ?",
                (self.job_id,),
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                return None
            return caster(row[0])

    def should_stop(self):
        if not self.is_ui_trainer:
            return False
        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        return _check_stop()

    def should_return_to_queue(self):
        if not self.is_ui_trainer:
            return False
        def _check_return_to_queue():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT return_to_queue FROM Job WHERE id = ?", (self.job_id,))
                return_to_queue = cursor.fetchone()
                return False if return_to_queue is None else return_to_queue[0] == 1

        return _check_return_to_queue()

    def maybe_stop(self):
        if not self.is_ui_trainer:
            return
        if self.should_stop():
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise Exception("Job stopped")
        if self.should_return_to_queue():
            self._run_async_operation(
                self._update_status("queued", "Job queued"))
            self.is_stopping = True
            raise Exception("Job returning to queue")

    def get_runtime_lr(self):
        """Read runtime_lr from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_lr", float)

    def apply_runtime_lr(self):
        """If runtime_lr is set in DB, apply it to the optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_lr()
        if value is None:
            return
        if value == self._last_applied_runtime_lr:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_lr"):
            if getattr(optimizer, "_lr", None) == value:
                self._last_applied_runtime_lr = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_lr from UI/DB: {value}")
            optimizer.set_lr(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_lr from DB not applied: optimizer has no set_lr (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_lr = value

    def get_runtime_gaussian_params(self):
        """Read runtime_gaussian_mean, runtime_gaussian_std from DB (only when is_ui_trainer). Returns (mean, std) or (None, None)."""
        if not self.is_ui_trainer:
            return (None, None)

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_gaussian_mean, runtime_gaussian_std FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return (None, None)
                mean = float(row[0]) if row[0] is not None else None
                std = float(row[1]) if row[1] is not None else None
                return (mean, std)

        return _read()

    def apply_runtime_gaussian_params(self):
        """If runtime_gaussian_mean/std are set in DB, apply them to train_config."""
        if not self.is_ui_trainer:
            return
        mean, std = self.get_runtime_gaussian_params()
        if mean is None and std is None:
            return
        if mean == self._last_applied_runtime_gaussian_mean and std == self._last_applied_runtime_gaussian_std:
            return
        mean_unchanged = mean is None or mean == self.train_config.gaussian_mean
        std_unchanged = std is None or std == self.train_config.gaussian_std
        if mean_unchanged and std_unchanged:
            self._last_applied_runtime_gaussian_mean = mean
            self._last_applied_runtime_gaussian_std = std
            return
        if mean is not None:
            self.train_config.gaussian_mean = mean
        if std is not None:
            self.train_config.gaussian_std = std
        self._last_applied_runtime_gaussian_mean = mean
        self._last_applied_runtime_gaussian_std = std
        if is_debug_enabled():
            print_acc(
                f"\nruntime gaussian_mean/std from UI/DB: {self.train_config.gaussian_mean}, {self.train_config.gaussian_std}"
            )

    def get_runtime_gaussian_peak2_params(self):
        """Read runtime_gaussian_mean_2, runtime_gaussian_std_2 from DB. Returns (mean2, std2) or (None, None)."""
        if not self.is_ui_trainer:
            return (None, None)

        def _read():
            try:
                with self._db_connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT runtime_gaussian_mean_2, runtime_gaussian_std_2 FROM RuntimeParams WHERE jobId = ?",
                        (self.job_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return (None, None)
                    mean2 = float(row[0]) if row[0] is not None else None
                    std2 = float(row[1]) if row[1] is not None else None
                    return (mean2, std2)
            except sqlite3.OperationalError:
                # DB not migrated yet (columns missing)
                return (None, None)

        return _read()

    def apply_runtime_gaussian_peak2_params(self):
        """If runtime_gaussian_mean_2/std_2 are set in DB, apply to train_config (partial updates supported)."""
        if not self.is_ui_trainer:
            return
        mean2, std2 = self.get_runtime_gaussian_peak2_params()
        if mean2 is None and std2 is None:
            return
        if (
            mean2 == self._last_applied_runtime_gaussian_mean_2
            and std2 == self._last_applied_runtime_gaussian_std_2
        ):
            return
        mean2_unchanged = mean2 is None or mean2 == self.train_config.gaussian_mean_2
        std2_unchanged = std2 is None or std2 == self.train_config.gaussian_std_2
        if mean2_unchanged and std2_unchanged:
            self._last_applied_runtime_gaussian_mean_2 = mean2
            self._last_applied_runtime_gaussian_std_2 = std2
            return
        if mean2 is not None:
            self.train_config.gaussian_mean_2 = mean2
        if std2 is not None:
            self.train_config.gaussian_std_2 = std2
        self._last_applied_runtime_gaussian_mean_2 = mean2
        self._last_applied_runtime_gaussian_std_2 = std2
        if is_debug_enabled():
            print_acc(
                f"\nruntime gaussian_mean_2/std_2 from UI/DB: {self.train_config.gaussian_mean_2}, {self.train_config.gaussian_std_2}"
            )

    def get_runtime_batch_size(self):
        """Read runtime_batch_size from DB (only when is_ui_trainer). Returns int or None."""
        return self._get_runtime_scalar("runtime_batch_size", int)

    def get_runtime_weight_decay(self):
        """Read runtime_weight_decay from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_weight_decay", float)

    def get_runtime_weight_decay_increment(self):
        """Read runtime_weight_decay_increment from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_weight_decay_increment", float)

    def get_runtime_weight_decay_mode(self):
        """Read runtime_weight_decay_mode from DB (only when is_ui_trainer). Returns str or None."""
        return self._get_runtime_scalar("runtime_weight_decay_mode", str)

    def get_runtime_beta1(self):
        """Read runtime_beta1 from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_beta1", float)

    def get_runtime_beta2(self):
        """Read runtime_beta2 from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_beta2", float)

    def apply_runtime_batch_size(self):
        """If runtime_batch_size is set in DB, apply it to train_config and resize data loaders."""
        if not self.is_ui_trainer:
            return
        batch_size = self.get_runtime_batch_size()
        if batch_size is None:
            return
        if batch_size == self._last_applied_runtime_batch_size:
            return
        if batch_size == self.train_config.batch_size:
            self._last_applied_runtime_batch_size = batch_size
            return

        old_batch_size = self.train_config.batch_size
        self.train_config.batch_size = batch_size
        self._last_applied_runtime_batch_size = batch_size

        from toolkit.data_loader import resize_dataloader_batch_size

        if self.data_loader is not None:
            self.data_loader = resize_dataloader_batch_size(
                self.data_loader,
                self.train_config.batch_size,
                epoch_num=self.epoch_num,
            )

        if self.data_loader_reg is not None:
            self.data_loader_reg = resize_dataloader_batch_size(
                self.data_loader_reg,
                self.train_config.batch_size,
                epoch_num=self.epoch_num,
            )

        if is_debug_enabled():
            print_acc(
                f"\nruntime batch_size from UI/DB: {old_batch_size} -> {batch_size}, data loaders resized"
            )

    def get_runtime_gradient_accumulation(self):
        """Read runtime_gradient_accumulation from DB (only when is_ui_trainer). Returns int or None."""
        return self._get_runtime_scalar("runtime_gradient_accumulation", int)

    def apply_runtime_gradient_accumulation(self):
        """If runtime_gradient_accumulation is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_gradient_accumulation()
        if value is None:
            return
        if value == self._last_applied_runtime_gradient_accumulation:
            return
        if value == self.train_config.gradient_accumulation:
            self._last_applied_runtime_gradient_accumulation = value
            return
        self.train_config.gradient_accumulation = value
        self._last_applied_runtime_gradient_accumulation = value
        if is_debug_enabled():
            print_acc(
                f"\nruntime gradient_accumulation from UI/DB: {value}"
            )

    def get_runtime_save_every(self):
        """Read runtime_save_every from DB (only when is_ui_trainer). Returns int or None."""
        return self._get_runtime_scalar("runtime_save_every", int)

    def get_runtime_sample_every(self):
        """Read runtime_sample_every from DB (only when is_ui_trainer). Returns int or None."""
        return self._get_runtime_scalar("runtime_sample_every", int)

    def get_runtime_warmup_steps(self):
        """Read runtime_warmup_steps from DB (only when is_ui_trainer). Returns int or None."""
        return self._get_runtime_scalar("runtime_warmup_steps", int)

    def get_runtime_warmup_boost(self):
        """Read runtime_warmup_boost from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_warmup_boost", float)

    def get_runtime_scale_lr_by_index(self):
        """Read runtime_scale_lr_by_index from DB. Returns bool or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            try:
                return self._get_runtime_scalar("runtime_scale_lr_by_index", bool)
            except sqlite3.OperationalError:
                return None

        return _read()

    def get_runtime_scale_lr_mean(self) -> Optional[float]:
        """Read runtime_scale_lr_mean from DB. Returns finite float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            try:
                raw = self._get_runtime_scalar("runtime_scale_lr_mean", float)
            except sqlite3.OperationalError:
                return None
            except (ValueError, TypeError):
                return None
            if raw is None:
                return None
            try:
                mean = float(raw)
            except (ValueError, TypeError):
                return None
            if not math.isfinite(mean):
                return None
            return mean

        return _read()

    def get_runtime_scale_lr_std(self) -> Optional[float]:
        """Read runtime_scale_lr_std from DB. Returns positive finite float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            try:
                raw = self._get_runtime_scalar("runtime_scale_lr_std", float)
            except sqlite3.OperationalError:
                return None
            except (ValueError, TypeError):
                return None
            if raw is None:
                return None
            try:
                std = float(raw)
            except (ValueError, TypeError):
                return None
            if not math.isfinite(std) or std <= 0.0:
                return None
            return std

        return _read()

    def _read_runtime_scale_lr_mask(self) -> RuntimeScaleLrMaskRead:
        """Read runtime_scale_lr_mask JSON from DB as a typed status/value outcome."""
        if not self.is_ui_trainer:
            return RuntimeScaleLrMaskRead("absent")

        def _read() -> RuntimeScaleLrMaskRead:
            try:
                with self._db_connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT runtime_scale_lr_mask FROM RuntimeParams WHERE jobId = ?",
                        (self.job_id,),
                    )
                    row = cursor.fetchone()
                    if row is None or row[0] is None:
                        return RuntimeScaleLrMaskRead("absent")
                    if isinstance(row[0], str) and row[0].strip() == "":
                        return RuntimeScaleLrMaskRead("absent")
                    try:
                        raw = row[0] if isinstance(row[0], str) else str(row[0])
                        parsed = json.loads(raw)
                    except (ValueError, TypeError, json.JSONDecodeError):
                        return RuntimeScaleLrMaskRead("invalid")
                    if not isinstance(parsed, list):
                        return RuntimeScaleLrMaskRead("invalid")
                    out: List[str] = []
                    for item in parsed:
                        if not isinstance(item, str) or item == "":
                            return RuntimeScaleLrMaskRead("invalid")
                        out.append(item)
                    return RuntimeScaleLrMaskRead("ok", out)
            except sqlite3.OperationalError:
                return RuntimeScaleLrMaskRead("absent")

        return _read()

    def get_runtime_scale_lr_mask(self) -> Optional[List[str]]:
        """Read runtime_scale_lr_mask from DB. Returns list[str] or None.

        Returns the list when status is ok (including []). Returns None for absent
        and invalid; apply_runtime_scale_lr uses _read_runtime_scale_lr_mask to
        distinguish invalid (abort) from absent (default mask []).
        """
        result = self._read_runtime_scale_lr_mask()
        if result.status != "ok":
            return None
        values = result.value
        if values is None:
            return []
        return values

    def apply_runtime_scale_lr(self):
        """Atomically apply runtime mean/std/mask then scale_lr_by_index to optimizer."""
        if not self.is_ui_trainer:
            return

        mean = self.get_runtime_scale_lr_mean()
        std = self.get_runtime_scale_lr_std()
        mask_result = self._read_runtime_scale_lr_mask()
        by_index = self.get_runtime_scale_lr_by_index()

        # Malformed / partially invalid mask: do not change optimizer at all.
        if mask_result.status == "invalid":
            return

        if mask_result.status == "ok":
            mask_list = mask_result.value if mask_result.value is not None else []
        else:
            # absent / NULL / missing column → treat as explicit empty clear for apply
            mask_list = []

        has_mean_std = mean is not None and std is not None

        # Incomplete mean/std (only one present / non-finite / non-numeric): no apply.
        if (mean is None) ^ (std is None):
            return

        # Leftover by_index=true without valid mean/std: do not enable, do not change.
        if by_index is True and not has_mean_std:
            return

        if not has_mean_std and by_index is None:
            return

        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer

        if mean is not None and std is not None:
            # Cache key is exactly (mean, std, tuple(mask)); [] is a valid explicit clear.
            config_key = (mean, std, tuple(mask_list))
            if config_key != self._last_applied_runtime_scale_lr_config:
                if hasattr(optimizer, "set_scale_lr_config"):
                    if is_debug_enabled():
                        print_acc(
                            f"\nruntime_scale_lr_config from UI/DB: "
                            f"mean={mean} std={std} mask={mask_list}"
                        )
                    try:
                        optimizer.set_scale_lr_config(mean, std, mask_list)
                    except ValueError as e:
                        print_acc(
                            f"\nruntime_scale_lr_config from DB not applied: {e}"
                        )
                        return
                else:
                    if is_debug_enabled():
                        print_acc(
                            "\nruntime_scale_lr_config from DB not applied: optimizer "
                            f"has no set_scale_lr_config (type: {type(optimizer).__name__})"
                        )
                self._last_applied_runtime_scale_lr_config = config_key

        if by_index is None:
            return
        if by_index == self._last_applied_runtime_scale_lr_by_index:
            return
        if hasattr(optimizer, "set_scale_lr_by_index"):
            if getattr(optimizer, "scale_lr_by_index", None) == by_index:
                self._last_applied_runtime_scale_lr_by_index = by_index
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_scale_lr_by_index from UI/DB: {by_index}")
            try:
                optimizer.set_scale_lr_by_index(by_index)
            except ValueError as e:
                print_acc(f"\nruntime_scale_lr_by_index from DB not applied: {e}")
                return
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_scale_lr_by_index from DB not applied: optimizer has no "
                    f"set_scale_lr_by_index (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_scale_lr_by_index = by_index

    def get_runtime_min_snr_gamma(self):
        """Read runtime_min_snr_gamma from DB (only when is_ui_trainer). Returns float or None."""
        return self._get_runtime_scalar("runtime_min_snr_gamma", float)

    def get_runtime_debug(self):
        """Read runtime_debug from DB (only when is_ui_trainer). Returns bool or None."""
        return self._get_runtime_scalar("runtime_debug", bool)

    def apply_runtime_debug(self):
        """If runtime_debug is set in DB, apply it to logging_config.debug."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_debug()
        if value is None or value == self._last_applied_runtime_debug:
            return
        if value == self.logging_config.debug:
            self._last_applied_runtime_debug = value
            return
        self.logging_config.debug = value
        self._last_applied_runtime_debug = value
        if is_debug_enabled():
            print_acc(f"\nruntime debug from UI/DB: {value}")

    def apply_runtime_save_every(self):
        """If runtime_save_every is set in DB, apply it to save_config."""
        if not self.is_ui_trainer:
            return
        save_every = self.get_runtime_save_every()
        if save_every is None:
            return
        if save_every == self._last_applied_runtime_save_every:
            return
        if save_every == self.save_config.save_every:
            self._last_applied_runtime_save_every = save_every
            return

        old_save_every = self.save_config.save_every
        self.save_config.save_every = save_every
        self._last_applied_runtime_save_every = save_every

        if is_debug_enabled():
            print_acc(
                f"\nruntime save_every from UI/DB: {old_save_every} -> {save_every}"
            )

    def apply_runtime_sample_every(self):
        """If runtime_sample_every is set in DB, apply it to sample_config."""
        if not self.is_ui_trainer:
            return
        sample_every = self.get_runtime_sample_every()
        if sample_every is None:
            return
        if sample_every == self._last_applied_runtime_sample_every:
            return
        if sample_every == self.sample_config.sample_every:
            self._last_applied_runtime_sample_every = sample_every
            return
        
        old_sample_every = self.sample_config.sample_every
        self.sample_config.sample_every = sample_every
        self._last_applied_runtime_sample_every = sample_every
        
        if is_debug_enabled():
            print_acc(
                f"\nruntime sample_every from UI/DB: {old_sample_every} -> {sample_every}"
            )

    def apply_runtime_warmup_steps(self):
        """If runtime_warmup_steps is set in DB, apply it to optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_warmup_steps()
        if value is None:
            return
        if value == self._last_applied_runtime_warmup_steps:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_warmup_steps"):
            pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
            if pg0.get("warmup_steps", None) == value:
                self._last_applied_runtime_warmup_steps = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_warmup_steps from UI/DB: {value}")
            optimizer.set_warmup_steps(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_warmup_steps from DB not applied: optimizer has no set_warmup_steps (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_warmup_steps = value

    def apply_runtime_warmup_boost(self):
        """If runtime_warmup_boost is set in DB, apply it to optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_warmup_boost()
        if value is None:
            return
        if value == self._last_applied_runtime_warmup_boost:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_warmup_boost"):
            pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
            if pg0.get("warmup_boost", None) == value:
                self._last_applied_runtime_warmup_boost = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_warmup_boost from UI/DB: {value}")
            optimizer.set_warmup_boost(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_warmup_boost from DB not applied: optimizer has no set_warmup_boost (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_warmup_boost = value

    def apply_runtime_min_snr_gamma(self):
        """If runtime_min_snr_gamma is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_min_snr_gamma()
        if value is None:
            return
        if value == self._last_applied_runtime_min_snr_gamma:
            return
        if value == self.train_config.min_snr_gamma:
            self._last_applied_runtime_min_snr_gamma = value
            return
        
        old_min_snr_gamma = self.train_config.min_snr_gamma
        self.train_config.min_snr_gamma = value
        self._last_applied_runtime_min_snr_gamma = value
        
        if is_debug_enabled():
            print_acc(
                f"\nruntime min_snr_gamma from UI/DB: {old_min_snr_gamma} -> {value}"
            )

    def apply_runtime_weight_decay(self):
        """If runtime_weight_decay is set in DB, apply it to the optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_weight_decay()
        if value is None:
            return
        if value == self._last_applied_runtime_weight_decay:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_weight_decay"):
            if getattr(optimizer, "_weight_decay", None) == value:
                self._last_applied_runtime_weight_decay = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_weight_decay from UI/DB: {value}")
            optimizer.set_weight_decay(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_weight_decay from DB not applied: optimizer has no set_weight_decay (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_weight_decay = value

    def apply_runtime_weight_decay_increment(self):
        """If runtime_weight_decay_increment is set in DB, apply it to the optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_weight_decay_increment()
        if value is None:
            return
        if value == self._last_applied_runtime_weight_decay_increment:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_weight_decay_increment"):
            pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
            if pg0.get("weight_decay_increment", 0.0) == value:
                self._last_applied_runtime_weight_decay_increment = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_weight_decay_increment from UI/DB: {value}")
            optimizer.set_weight_decay_increment(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_weight_decay_increment from DB not applied: optimizer has no set_weight_decay_increment (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_weight_decay_increment = value

    def apply_runtime_weight_decay_mode(self):
        """If runtime_weight_decay_mode is set in DB, apply it to the optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_weight_decay_mode()
        if value is None:
            return
        if value == self._last_applied_runtime_weight_decay_mode:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_weight_decay_mode"):
            if getattr(optimizer, "_weight_decay_mode", None) == value:
                self._last_applied_runtime_weight_decay_mode = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_weight_decay_mode from UI/DB: {value}")
            optimizer.set_weight_decay_mode(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_weight_decay_mode from DB not applied: optimizer has no set_weight_decay_mode (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_weight_decay_mode = value

    def apply_runtime_beta1(self):
        """If runtime_beta1 is set in DB, apply it to the optimizer (e.g. Adafactor). None disables momentum."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_beta1()
        if value == self._last_applied_runtime_beta1:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_beta1"):
            if getattr(optimizer, "_beta1", None) == value:
                self._last_applied_runtime_beta1 = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_beta1 from UI/DB: {value}")
            optimizer.set_beta1(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_beta1 from DB not applied: optimizer has no set_beta1 (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_beta1 = value

    def apply_runtime_beta2(self):
        """If runtime_beta2 is set in DB, apply it to the optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_beta2()
        if value is None:
            return
        if value == self._last_applied_runtime_beta2:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_beta2"):
            if getattr(optimizer, "_beta2", None) == value:
                self._last_applied_runtime_beta2 = value
                return
            if is_debug_enabled():
                print_acc(f"\nruntime_beta2 from UI/DB: {value}")
            optimizer.set_beta2(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_beta2 from DB not applied: optimizer has no set_beta2 (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_beta2 = value

    def get_runtime_content_or_style(self):
        """Read runtime_content_or_style from DB (only when is_ui_trainer). Returns str or None."""
        return self._get_runtime_scalar("runtime_content_or_style", str)

    def apply_runtime_content_or_style(self):
        """If runtime_content_or_style is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_content_or_style()
        if value is None:
            return
        if value == self._last_applied_runtime_content_or_style:
            return
        if (value == self.train_config.content_or_style):
            self._last_applied_runtime_content_or_style = value
            return
        self.train_config.content_or_style = value
        self._last_applied_runtime_content_or_style = value
        if is_debug_enabled():
            print_acc(f"\nruntime content_or_style from UI/DB: {value}")

    def get_runtime_timestep_type(self):
        """Read runtime_timestep_type from DB (only when is_ui_trainer). Returns str or None."""
        return self._get_runtime_scalar("runtime_timestep_type", str)

    def apply_runtime_timestep_type(self):
        """If runtime_timestep_type is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_timestep_type()
        if value is None:
            return
        if value == self._last_applied_runtime_timestep_type:
            return
        if value == self.train_config.timestep_type:
            self._last_applied_runtime_timestep_type = value
            return
        self.train_config.timestep_type = value
        self._last_applied_runtime_timestep_type = value
        if is_debug_enabled():
            print_acc(f"\nruntime timestep_type from UI/DB: {value}")

    def get_runtime_turbo_prior_params(self):
        """Read runtime_turbo_prior_steps, runtime_turbo_t_jitter from DB. Returns (steps, jitter) or (None, None)."""
        if not self.is_ui_trainer:
            return (None, None)

        def _read():
            try:
                with self._db_connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT runtime_turbo_prior_steps, runtime_turbo_t_jitter "
                        "FROM RuntimeParams WHERE jobId = ?",
                        (self.job_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return (None, None)
                    steps = int(row[0]) if row[0] is not None else None
                    jitter = float(row[1]) if row[1] is not None else None
                    return (steps, jitter)
            except sqlite3.OperationalError:
                # DB not migrated yet (columns missing)
                return (None, None)

        return _read()

    def apply_runtime_turbo_prior_params(self):
        """If runtime turbo_prior steps/jitter are set in DB, apply them to train_config."""
        if not self.is_ui_trainer:
            return
        steps, jitter = self.get_runtime_turbo_prior_params()
        if steps is None and jitter is None:
            return
        if (
            steps == self._last_applied_runtime_turbo_prior_steps
            and jitter == self._last_applied_runtime_turbo_t_jitter
        ):
            return
        steps_unchanged = steps is None or steps == self.train_config.turbo_prior_steps
        jitter_unchanged = jitter is None or jitter == self.train_config.turbo_t_jitter
        if steps_unchanged and jitter_unchanged:
            self._last_applied_runtime_turbo_prior_steps = steps
            self._last_applied_runtime_turbo_t_jitter = jitter
            return
        if steps is not None:
            self.train_config.turbo_prior_steps = steps
        if jitter is not None:
            self.train_config.turbo_t_jitter = jitter
        self._last_applied_runtime_turbo_prior_steps = steps
        self._last_applied_runtime_turbo_t_jitter = jitter
        if is_debug_enabled():
            print_acc(
                f"\nruntime turbo_prior from UI/DB: "
                f"steps={self.train_config.turbo_prior_steps}, "
                f"jitter={self.train_config.turbo_t_jitter}"
            )

    def get_runtime_turbo_teacher_weight(self):
        """Read runtime_turbo_teacher_weight from DB. Returns bool or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            try:
                return self._get_runtime_scalar("runtime_turbo_teacher_weight", bool)
            except sqlite3.OperationalError:
                # DB not migrated yet (column missing)
                return None

        return _read()

    def apply_runtime_turbo_teacher_weight(self):
        """If runtime_turbo_teacher_weight is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_turbo_teacher_weight()
        if value is None:
            return
        if value == self._last_applied_runtime_turbo_teacher_weight:
            return
        current = bool(getattr(self.train_config, "turbo_teacher_weight", False))
        if value == current:
            self._last_applied_runtime_turbo_teacher_weight = value
            return
        self.train_config.turbo_teacher_weight = bool(value)
        if hasattr(self, "apply_runtime_turbo_teacher_mode"):
            self.apply_runtime_turbo_teacher_mode(bool(value))
        self._last_applied_runtime_turbo_teacher_weight = value
        if is_debug_enabled():
            print_acc(f"\nruntime turbo_teacher_weight from UI/DB: {value}")

    def get_runtime_timestep_weighting(self):
        """Read runtime_timestep_weighting from DB (only when is_ui_trainer). Returns str or None."""
        return self._get_runtime_scalar("runtime_timestep_weighting", str)

    def apply_runtime_timestep_weighting(self):
        """If runtime_timestep_weighting is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_timestep_weighting()
        if value is None:
            return
        if value == self._last_applied_runtime_timestep_weighting:
            return
        if value == self.train_config.timestep_weighting:
            self._last_applied_runtime_timestep_weighting = value
            return
        self.train_config.timestep_weighting = value
        self._last_applied_runtime_timestep_weighting = value
        if is_debug_enabled():
            print_acc(f"\nruntime timestep_weighting from UI/DB: {value}")

    def get_runtime_network_weights(self):
        """Read runtime_network_weights from DB (only when is_ui_trainer). Returns list of float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_network_weights FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None or (isinstance(row[0], str) and row[0].strip() == ""):
                    return None
                try:
                    raw = row[0] if isinstance(row[0], str) else str(row[0])
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        return None
                    return [
                        float(x) for x in parsed
                        if isinstance(x, (int, float))
                        and (v := float(x)) == v
                        and abs(v) != float("inf")
                        and v >= 0
                    ]
                except (ValueError, TypeError):
                    return None

        return _read()

    def apply_runtime_network_weights(self):
        """If runtime_network_weights are set in DB, apply them to dataloader dataset_configs."""
        if not self.is_ui_trainer:
            return
        weights = self.get_runtime_network_weights()
        if weights is None:
            return
        weights_tuple = tuple(weights)
        if weights_tuple == self._last_applied_runtime_network_weights:
            return
        if self.data_loader is None:
            return

        from toolkit.data_loader import rebuild_dataloader_network_weights

        for loader in (self.data_loader, self.data_loader_reg):
            if loader is None:
                continue
            datasets = get_dataloader_datasets(loader)
            for i, ds in enumerate(datasets):
                if i < len(weights):
                    w = weights[i]
                    ds.dataset_config.network_weight = w
                    for file_item in ds.file_list:
                        file_item.network_weight = w

        try:
            self.data_loader = rebuild_dataloader_network_weights(
                self.data_loader, epoch_num=self.epoch_num
            )
        except ValueError:
            pass
        if self.data_loader_reg is not None:
            try:
                self.data_loader_reg = rebuild_dataloader_network_weights(
                    self.data_loader_reg, epoch_num=self.epoch_num
                )
            except ValueError:
                pass

        self._last_applied_runtime_network_weights = weights_tuple
        if is_debug_enabled():
            print_acc(f"\nruntime network_weights from UI/DB applied: {list(weights_tuple)}")

    def get_runtime_prompts(self):
        """Read runtime_prompts from DB (only when is_ui_trainer). Returns list of str or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_prompts FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None or (isinstance(row[0], str) and row[0].strip() == ""):
                    return None
                try:
                    raw = row[0] if isinstance(row[0], str) else str(row[0])
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        return None
                    return [x if isinstance(x, str) else None for x in parsed]
                except (ValueError, TypeError, json.JSONDecodeError):
                    return None

        return _read()

    def _recache_sample_prompts_runtime(self) -> None:
        """Recache sample prompt embeds via common text-cache residency lifecycle.

        Caching path: reload stashed TE (CPU) → enter → cache → unload/Fake → exit.
        Unload-only path: enter → cache → real TE→CPU (no Fake/stash) → exit.
        On any failure, abort offloads live+stashed TE and non-TE owners so TE and
        backbone never co-reside on CUDA; original error propagates (cleanup chained).
        """
        from toolkit.basic import flush
        from toolkit.unloader import (
            abort_text_cache_residency,
            enter_text_cache_residency,
            exit_text_cache_residency,
            reload_text_encoder,
            unload_text_encoder,
        )

        caching = bool(getattr(self, "is_caching_text_embeddings", False))
        unload_only = bool(getattr(self.train_config, "unload_text_encoder", False)) and not caching
        if not caching and not unload_only:
            return

        try:
            if caching:
                reload_text_encoder(self.sd)
            enter_text_cache_residency(self.sd, self.device_torch)
            self.cache_sample_prompts()
            if caching:
                unload_text_encoder(self.sd)
            else:
                self.sd.text_encoder_to("cpu")
                flush()
            # Exit only after real TE is CPU/Fake (unload or TE→CPU above).
            exit_text_cache_residency(self.sd, self.device_torch)
        except Exception as err:
            try:
                abort_text_cache_residency(self.sd)
            except Exception as cleanup_err:
                raise err from cleanup_err
            raise

    def apply_runtime_prompts(self):
        """If runtime_prompts are set in DB, update sample_config and recache embeds if needed."""
        if not self.is_ui_trainer:
            return
        prompts = self.get_runtime_prompts()
        if prompts is None:
            return
        if any(p is None for p in prompts):
            return
        prompts_tuple = tuple(prompts)
        if prompts_tuple == getattr(self, "_last_applied_runtime_prompts", None):
            return

        samples = getattr(self.sample_config, "samples", None) or []
        current = tuple((s.prompt if s.prompt is not None else "") for s in samples)
        if prompts_tuple == current:
            self._last_applied_runtime_prompts = prompts_tuple
            return

        changed = False
        for i, prompt in enumerate(prompts):
            if i >= len(samples):
                break
            if samples[i].prompt != prompt:
                samples[i].prompt = prompt
                changed = True

        self._last_applied_runtime_prompts = prompts_tuple
        if not changed:
            return

        need_recache = bool(getattr(self, "is_caching_text_embeddings", False)) or bool(
            getattr(self.train_config, "unload_text_encoder", False)
        )
        if need_recache:
            self._recache_sample_prompts_runtime()

        if is_debug_enabled():
            print_acc(
                f"\nruntime prompts from UI/DB applied ({len(prompts_tuple)} prompts)"
            )

    def _reset_fixed_cycle_sampling_cache(self) -> None:
        """Invalidate TimestepSampler fixed_cycle resolution after timesteps/seed change."""
        bp = getattr(self, "_batch_processor", None)
        if bp is None:
            return
        sampler = getattr(bp, "_timestep_sampler", None)
        if sampler is not None:
            sampler.reset_cache()

    def get_runtime_fixed_cycle_params(self):
        """
        Read fixed-cycle runtime columns from DB.
        Returns (timesteps, seed, weight_peak_timesteps, weight_sigma); each None if missing/invalid.
        weight_peak_timesteps may be [] when DB stores empty JSON array.
        """
        if not self.is_ui_trainer:
            return (None, None, None, None)

        def _parse_json_float_list(raw, min_length: int) -> Optional[List[float]]:
            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                return None
            try:
                s = raw if isinstance(raw, str) else str(raw)
                parsed = json.loads(s)
                if not isinstance(parsed, list):
                    return None
                out: List[float] = []
                for x in parsed:
                    if not isinstance(x, (int, float)):
                        return None
                    v = float(x)
                    if v != v or abs(v) == float("inf"):
                        return None
                    out.append(v)
                if len(out) < min_length:
                    return None
                return out
            except (TypeError, ValueError, json.JSONDecodeError):
                return None

        def _read():
            try:
                with self._db_connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT runtime_fixed_cycle_timesteps, runtime_fixed_cycle_seed, "
                        "runtime_fixed_cycle_weight_peak_timesteps, runtime_fixed_cycle_weight_sigma "
                        "FROM RuntimeParams WHERE jobId = ?",
                        (self.job_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return (None, None, None, None)
                    ts_raw, seed_raw, peaks_raw, sigma_raw = (
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                    )
                    ts_p = _parse_json_float_list(ts_raw, min_length=1)
                    peaks_p = _parse_json_float_list(peaks_raw, min_length=0)
                    seed_p = int(seed_raw) if seed_raw is not None else None
                    sigma_p = (
                        float(sigma_raw) if sigma_raw is not None else None
                    )
                    return (ts_p, seed_p, peaks_p, sigma_p)
            except sqlite3.OperationalError:
                return (None, None, None, None)

        return _read()

    def apply_runtime_fixed_cycle_params(self):
        """Apply runtime fixed-cycle fields from DB when Timestep Bias uses fixed_cycle (main or reg)."""
        if not self.is_ui_trainer:
            return
        cos = self.train_config.content_or_style
        cos_reg = getattr(self.train_config, "content_or_style_reg", cos)
        if cos != "fixed_cycle" and cos_reg != "fixed_cycle":
            return

        ts_db, seed_db, peaks_db, sigma_db = self.get_runtime_fixed_cycle_params()
        if (
            ts_db is None
            and seed_db is None
            and peaks_db is None
            and sigma_db is None
        ):
            return

        fc_key = (
            tuple(ts_db) if ts_db is not None else None,
            seed_db,
            tuple(peaks_db) if peaks_db is not None else None,
            sigma_db,
        )
        if fc_key == self._last_applied_runtime_fc_key:
            return

        config_matches = True
        if ts_db is not None:
            config_matches = config_matches and tuple(
                self.train_config.fixed_cycle_timesteps or []
            ) == tuple(ts_db)
        if seed_db is not None:
            config_matches = config_matches and self.train_config.fixed_cycle_seed == seed_db
        if peaks_db is not None:
            expected_peaks = list(peaks_db) if len(peaks_db) > 0 else None
            config_matches = (
                config_matches
                and self.train_config.fixed_cycle_weight_peak_timesteps == expected_peaks
            )
        if sigma_db is not None:
            config_matches = (
                config_matches
                and self.train_config.fixed_cycle_weight_sigma == sigma_db
            )
        if config_matches:
            self._last_applied_runtime_fc_key = fc_key
            return

        need_reset_cache = False
        old_ts = tuple(self.train_config.fixed_cycle_timesteps or [])
        old_seed = self.train_config.fixed_cycle_seed

        if ts_db is not None:
            new_ts = tuple(ts_db)
            if new_ts != old_ts:
                need_reset_cache = True
            self.train_config.fixed_cycle_timesteps = list(ts_db)
        if seed_db is not None:
            if seed_db != old_seed:
                need_reset_cache = True
            self.train_config.fixed_cycle_seed = seed_db
        if peaks_db is not None:
            self.train_config.fixed_cycle_weight_peak_timesteps = (
                list(peaks_db) if len(peaks_db) > 0 else None
            )
        if sigma_db is not None:
            self.train_config.fixed_cycle_weight_sigma = sigma_db

        self._last_applied_runtime_fc_key = fc_key
        if need_reset_cache:
            self._reset_fixed_cycle_sampling_cache()
        if is_debug_enabled():
            print_acc(
                "\nruntime fixed_cycle from UI/DB: "
                f"timesteps={self.train_config.fixed_cycle_timesteps}, "
                f"seed={self.train_config.fixed_cycle_seed}, "
                f"weight_peaks={self.train_config.fixed_cycle_weight_peak_timesteps}, "
                f"weight_sigma={self.train_config.fixed_cycle_weight_sigma}"
            )

    def clear_runtime_params(self):
        """Clear all runtime parameters from the RuntimeParams table for this job."""
        if not self.is_ui_trainer:
            return
        
        def _clear():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM RuntimeParams WHERE jobId = ?", (self.job_id,)
                )
        
        _clear()
        if is_debug_enabled():
            print_acc("\nCleared runtime parameters from database")

    def _reset_last_applied_runtime(self):
        """Reset all cached runtime parameter values to None."""
        self._last_applied_runtime_lr = None
        self._last_applied_runtime_gaussian_mean = None
        self._last_applied_runtime_gaussian_std = None
        self._last_applied_runtime_gaussian_mean_2 = None
        self._last_applied_runtime_gaussian_std_2 = None
        self._last_applied_runtime_weight_decay = None
        self._last_applied_runtime_weight_decay_increment = None
        self._last_applied_runtime_weight_decay_mode = None
        self._last_applied_runtime_beta1 = None
        self._last_applied_runtime_beta2 = None
        self._last_applied_runtime_content_or_style = None
        self._last_applied_runtime_timestep_type = None
        self._last_applied_runtime_timestep_weighting = None
        self._last_applied_runtime_network_weights = None
        self._last_applied_runtime_prompts = None
        self._last_applied_runtime_batch_size = None
        self._last_applied_runtime_gradient_accumulation = None
        self._last_applied_runtime_save_every = None
        self._last_applied_runtime_sample_every = None
        self._last_applied_runtime_warmup_steps = None
        self._last_applied_runtime_warmup_boost = None
        self._last_applied_runtime_scale_lr_by_index = None
        self._last_applied_runtime_scale_lr_config = None
        self._last_applied_runtime_min_snr_gamma = None
        self._last_applied_runtime_debug = None
        self._last_applied_runtime_fc_key = None
        self._last_applied_runtime_turbo_prior_steps = None
        self._last_applied_runtime_turbo_t_jitter = None
        self._last_applied_runtime_turbo_teacher_weight = None

    async def _update_key(self, key, value):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # Convert the value to string if it's not already
                    if isinstance(value, str):
                        value_to_insert = value
                    else:
                        value_to_insert = str(value)

                    # Use parameterized query for both the column name and value
                    update_query = f"UPDATE Job SET {key} = ? WHERE id = ?"
                    cursor.execute(
                        update_query, (value_to_insert, self.job_id))
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_key("step", self.step_num))

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_key(key, value))

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.accelerator.is_main_process or not self.is_ui_trainer:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    if info is not None:
                        cursor.execute(
                            "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                            (status, info, self.job_id)
                        )
                    else:
                        cursor.execute(
                            "UPDATE Job SET status = ? WHERE id = ?",
                            (status, self.job_id)
                        )
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        """Non-blocking update of status."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_status(status, info))

    async def wait_for_all_async(self):
        """Wait for all tracked async operations to complete."""
        if not self._async_tasks:
            return

        try:
            await asyncio.gather(*self._async_tasks)
        except Exception as e:
            pass
        finally:
            # Clear the task list after completion
            self._async_tasks.clear()

    def on_error(self, e: Exception):
        super(DiffusionTrainer, self).on_error(e)
        if self.is_ui_trainer:
            if self.accelerator.is_main_process and not self.is_stopping:
                self.update_status("error", str(e))
            self.update_db_key("step", self.last_save_step)
            asyncio.run(self.wait_for_all_async())
            self.thread_pool.shutdown(wait=True)

    def handle_timing_print_hook(self, timing_dict):
        if "train_loop" not in timing_dict:
            print("train_loop not found in timing_dict", timing_dict)
            return
        seconds_per_iter = timing_dict["train_loop"]
        # determine iter/sec or sec/iter
        if seconds_per_iter < 1:
            iters_per_sec = 1 / seconds_per_iter
            self.update_db_key("speed_string", f"{iters_per_sec:.2f} iter/sec")
        else:
            self.update_db_key(
                "speed_string", f"{seconds_per_iter:.2f} sec/iter")

    def done_hook(self):
        super(DiffusionTrainer, self).done_hook()
        if self.is_ui_trainer:
            self.update_status("completed", "Training completed")
            # Wait for all async operations to finish before shutting down
            asyncio.run(self.wait_for_all_async())
            self.thread_pool.shutdown(wait=True)

    def end_step_hook(self):
        super(DiffusionTrainer, self).end_step_hook()
        if self.is_ui_trainer:
            self.update_step()
            self.maybe_stop()
            self.apply_runtime_lr()
            self.apply_runtime_gaussian_params()
            self.apply_runtime_gaussian_peak2_params()
            self.apply_runtime_weight_decay()
            self.apply_runtime_weight_decay_increment()
            self.apply_runtime_weight_decay_mode()
            self.apply_runtime_beta1()
            self.apply_runtime_beta2()
            self.apply_runtime_content_or_style()
            self.apply_runtime_fixed_cycle_params()
            self.apply_runtime_timestep_type()
            self.apply_runtime_turbo_prior_params()
            self.apply_runtime_turbo_teacher_weight()
            self.apply_runtime_timestep_weighting()
            self.apply_runtime_batch_size()
            self.apply_runtime_network_weights()
            self.apply_runtime_gradient_accumulation()
            self.apply_runtime_save_every()
            self.apply_runtime_sample_every()
            self.apply_runtime_prompts()
            self.apply_runtime_warmup_steps()
            self.apply_runtime_warmup_boost()
            self.apply_runtime_scale_lr()
            self.apply_runtime_min_snr_gamma()
            self.apply_runtime_debug()

    def hook_before_model_load(self):
        super().hook_before_model_load()
        if self.is_ui_trainer:
            self.maybe_stop()
            self.update_status("running", "Loading model")

    def before_dataset_load(self):
        super().before_dataset_load()
        if self.is_ui_trainer:
            self.maybe_stop()
            self.update_status("running", "Loading dataset")

    def internal_hook_before_train_loop(self):
        """Clear runtime DB row, reset cached runtime, update job step/status; optional subclasses run before this."""
        if self.is_ui_trainer:
            self.clear_runtime_params()
            self._reset_last_applied_runtime()
            self.maybe_stop()
            self.update_step()
            self.update_status("running", "Training")
            self.timer.add_after_print_hook(self.handle_timing_print_hook)

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.internal_hook_before_train_loop()

    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        if self.is_ui_trainer:
            self.maybe_stop()
            self.sd.add_status_update_hook(self.status_update_hook_func)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        if self.is_ui_trainer:
            self.maybe_stop()
            self.update_status(
                "running", f"Generating images - {img_num + 1}/{total_imgs}")

    def sample(self, step=None, is_first=False):
        self.maybe_stop()
        total_imgs = len(self.sample_config.prompts)
        self.update_status("running", f"Generating images - 0/{total_imgs}")
        super().sample(step, is_first)
        self.maybe_stop()
        self.update_status("running", "Training")

    def save(self, step=None):
        self.maybe_stop()
        self.update_status("running", "Saving model")
        super().save(step)
        self.maybe_stop()
        self.update_status("running", "Training")
