from collections import OrderedDict
import json
import os
import sqlite3
import asyncio
import concurrent.futures
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.accelerator import unwrap_model
from toolkit.data_loader import get_dataloader_datasets
from toolkit.print import print_acc
from toolkit.util.debug import is_debug_enabled
from typing import List, Literal, Optional, Tuple
import threading
import time
import signal

AITK_Status = Literal["running", "stopped", "error", "completed"]


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
            self._last_applied_runtime_min_lr = None
            self._last_applied_runtime_gaussian_mean = None
            self._last_applied_runtime_gaussian_std = None
            self._last_applied_runtime_gaussian_mean_2 = None
            self._last_applied_runtime_gaussian_std_2 = None
            self._last_applied_runtime_weight_decay = None
            self._last_applied_runtime_beta1 = None
            self._last_applied_runtime_beta2 = None
            self._last_applied_runtime_content_or_style = None
            self._last_applied_runtime_timestep_type = None
            self._last_applied_runtime_network_weights: Optional[tuple] = None
            self._last_applied_runtime_batch_size = None
            self._last_applied_runtime_gradient_accumulation = None
            self._last_applied_runtime_save_every = None
            self._last_applied_runtime_sample_every = None
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
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_lr FROM RuntimeParams WHERE jobId = ?", (self.job_id,)
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])

        return _read()

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
            if is_debug_enabled():
                print_acc(f"\nruntime_lr from UI/DB: {value}")
            optimizer.set_lr(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_lr from DB not applied: optimizer has no set_lr (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_lr = value

    def get_runtime_min_lr(self):
        """Read runtime_min_lr from DB (only when is_ui_trainer). Returns float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_min_lr FROM RuntimeParams WHERE jobId = ?", (self.job_id,)
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])

        return _read()

    def apply_runtime_min_lr(self):
        """If runtime_min_lr is set in DB, apply it to the optimizer (e.g. Adafactor)."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_min_lr()
        if value is None:
            return
        if value == self._last_applied_runtime_min_lr:
            return
        optimizer = unwrap_model(self.optimizer)
        while getattr(optimizer, "optimizer", None) is not None:
            optimizer = optimizer.optimizer
        if hasattr(optimizer, "set_min_lr"):
            if is_debug_enabled():
                print_acc(f"\nruntime_min_lr from UI/DB: {value}")
            optimizer.set_min_lr(value)
            self._last_applied_runtime_min_lr = value
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_min_lr from DB not applied: optimizer has no set_min_lr (type: {type(optimizer).__name__})"
                )

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
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_batch_size FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return int(row[0])

        return _read()

    def get_runtime_weight_decay(self):
        """Read runtime_weight_decay from DB (only when is_ui_trainer). Returns float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_weight_decay FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])

        return _read()

    def get_runtime_beta1(self):
        """Read runtime_beta1 from DB (only when is_ui_trainer). Returns float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_beta1 FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])

        return _read()

    def get_runtime_beta2(self):
        """Read runtime_beta2 from DB (only when is_ui_trainer). Returns float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_beta2 FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])

        return _read()

    def apply_runtime_batch_size(self):
        """If runtime_batch_size is set in DB, apply it to train_config and recreate data loaders."""
        if not self.is_ui_trainer:
            return
        batch_size = self.get_runtime_batch_size()
        if batch_size is None:
            return
        if batch_size == self._last_applied_runtime_batch_size:
            return
        
        old_batch_size = self.train_config.batch_size
        self.train_config.batch_size = batch_size
        self._last_applied_runtime_batch_size = batch_size
        
        # Recreate data loaders with new batch_size
        if self.datasets is not None:
            from toolkit.data_loader import get_dataloader_from_datasets
            self.data_loader = get_dataloader_from_datasets(
                self.datasets, 
                self.train_config.batch_size, 
                self.sd, 
                train_config=self.train_config
            )
        
        if self.datasets_reg is not None:
            from toolkit.data_loader import get_dataloader_from_datasets
            self.data_loader_reg = get_dataloader_from_datasets(
                self.datasets_reg, 
                self.train_config.batch_size,
                self.sd, 
                train_config=self.train_config
            )
        
        if is_debug_enabled():
            print_acc(
                f"\nruntime batch_size from UI/DB: {old_batch_size} -> {batch_size}, data loaders recreated"
            )

    def get_runtime_gradient_accumulation(self):
        """Read runtime_gradient_accumulation from DB (only when is_ui_trainer). Returns int or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_gradient_accumulation FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return int(row[0])

        return _read()

    def apply_runtime_gradient_accumulation(self):
        """If runtime_gradient_accumulation is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_gradient_accumulation()
        if value is None:
            return
        if value == self._last_applied_runtime_gradient_accumulation:
            return
        self.train_config.gradient_accumulation = value
        self._last_applied_runtime_gradient_accumulation = value
        if is_debug_enabled():
            print_acc(
                f"\nruntime gradient_accumulation from UI/DB: {value}"
            )

    def get_runtime_save_every(self):
        """Read runtime_save_every from DB (only when is_ui_trainer). Returns int or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_save_every FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return int(row[0])

        return _read()

    def get_runtime_sample_every(self):
        """Read runtime_sample_every from DB (only when is_ui_trainer). Returns int or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_sample_every FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return int(row[0])

        return _read()

    def get_runtime_min_snr_gamma(self):
        """Read runtime_min_snr_gamma from DB (only when is_ui_trainer). Returns float or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_min_snr_gamma FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])

        return _read()

    def get_runtime_debug(self):
        """Read runtime_debug from DB (only when is_ui_trainer). Returns bool or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_debug FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return bool(row[0])

        return _read()

    def apply_runtime_debug(self):
        """If runtime_debug is set in DB, apply it to logging_config.debug."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_debug()
        if value is None or value == self._last_applied_runtime_debug:
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
        
        old_sample_every = self.sample_config.sample_every
        self.sample_config.sample_every = sample_every
        self._last_applied_runtime_sample_every = sample_every
        
        if is_debug_enabled():
            print_acc(
                f"\nruntime sample_every from UI/DB: {old_sample_every} -> {sample_every}"
            )

    def apply_runtime_min_snr_gamma(self):
        """If runtime_min_snr_gamma is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_min_snr_gamma()
        if value is None:
            return
        if value == self._last_applied_runtime_min_snr_gamma:
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
            if is_debug_enabled():
                print_acc(f"\nruntime_weight_decay from UI/DB: {value}")
            optimizer.set_weight_decay(value)
        else:
            if is_debug_enabled():
                print_acc(
                    f"\nruntime_weight_decay from DB not applied: optimizer has no set_weight_decay (type: {type(optimizer).__name__})"
                )
        self._last_applied_runtime_weight_decay = value

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
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_content_or_style FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return str(row[0])

        return _read()

    def apply_runtime_content_or_style(self):
        """If runtime_content_or_style is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_content_or_style()
        if value is None:
            return
        if value == self._last_applied_runtime_content_or_style:
            return
        self.train_config.content_or_style = value
        self.train_config.content_or_style_reg = value
        self._last_applied_runtime_content_or_style = value
        if is_debug_enabled():
            print_acc(f"\nruntime content_or_style from UI/DB: {value}")

    def get_runtime_timestep_type(self):
        """Read runtime_timestep_type from DB (only when is_ui_trainer). Returns str or None."""
        if not self.is_ui_trainer:
            return None

        def _read():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT runtime_timestep_type FROM RuntimeParams WHERE jobId = ?",
                    (self.job_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                return str(row[0])

        return _read()

    def apply_runtime_timestep_type(self):
        """If runtime_timestep_type is set in DB, apply it to train_config."""
        if not self.is_ui_trainer:
            return
        value = self.get_runtime_timestep_type()
        if value is None:
            return
        if value == self._last_applied_runtime_timestep_type:
            return
        self.train_config.timestep_type = value
        self._last_applied_runtime_timestep_type = value
        if is_debug_enabled():
            print_acc(f"\nruntime timestep_type from UI/DB: {value}")

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
                    return [float(x) for x in parsed if isinstance(x, (int, float)) and abs(x) == x and x > 0]
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
        datasets = get_dataloader_datasets(self.data_loader)
        for i, ds in enumerate(datasets):
            if i < len(weights):
                ds.dataset_config.network_weight = weights[i]
        self._last_applied_runtime_network_weights = weights_tuple
        if is_debug_enabled():
            print_acc(f"\nruntime network_weights from UI/DB applied: {list(weights_tuple)}")

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
                    if v != v or abs(v) == float("inf") or v < 0.0 or v > 1000.0:
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
        self._last_applied_runtime_min_lr = None
        self._last_applied_runtime_gaussian_mean = None
        self._last_applied_runtime_gaussian_std = None
        self._last_applied_runtime_gaussian_mean_2 = None
        self._last_applied_runtime_gaussian_std_2 = None
        self._last_applied_runtime_weight_decay = None
        self._last_applied_runtime_beta1 = None
        self._last_applied_runtime_beta2 = None
        self._last_applied_runtime_content_or_style = None
        self._last_applied_runtime_timestep_type = None
        self._last_applied_runtime_network_weights = None
        self._last_applied_runtime_batch_size = None
        self._last_applied_runtime_gradient_accumulation = None
        self._last_applied_runtime_save_every = None
        self._last_applied_runtime_sample_every = None
        self._last_applied_runtime_min_snr_gamma = None
        self._last_applied_runtime_debug = None
        self._last_applied_runtime_fc_key = None

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
            self.apply_runtime_min_lr()
            self.apply_runtime_gaussian_params()
            self.apply_runtime_gaussian_peak2_params()
            self.apply_runtime_weight_decay()
            self.apply_runtime_beta1()
            self.apply_runtime_beta2()
            self.apply_runtime_content_or_style()
            self.apply_runtime_fixed_cycle_params()
            self.apply_runtime_timestep_type()
            self.apply_runtime_network_weights()
            self.apply_runtime_batch_size()
            self.apply_runtime_gradient_accumulation()
            self.apply_runtime_save_every()
            self.apply_runtime_sample_every()
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

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        if self.is_ui_trainer:
            self.clear_runtime_params()
            self._reset_last_applied_runtime()
            self.maybe_stop()
            self.update_step()
            self.update_status("running", "Training")
            self.timer.add_after_print_hook(self.handle_timing_print_hook)

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
