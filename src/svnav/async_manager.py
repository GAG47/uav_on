from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from svnav.types import AsyncTaskType, new_id, now_ts


class AsyncTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass
class AsyncManagerConfig:
    enabled: bool = True

    max_workers: int = 3

    max_inflight_total: int = 6
    max_inflight_task1: int = 1
    max_inflight_gdino: int = 1
    max_inflight_task2: int = 3

    max_completed_results: int = 32

    stale_task1_steps: int = 12
    stale_gdino_steps: int = 10
    stale_task2_steps: int = 14

    verbose: bool = False

    def __post_init__(self) -> None:
        self.max_workers = max(1, int(self.max_workers))
        self.max_inflight_total = max(1, int(self.max_inflight_total))
        self.max_inflight_task1 = max(0, int(self.max_inflight_task1))
        self.max_inflight_gdino = max(0, int(self.max_inflight_gdino))
        self.max_inflight_task2 = max(0, int(self.max_inflight_task2))
        self.max_completed_results = max(1, int(self.max_completed_results))
        self.stale_task1_steps = max(0, int(self.stale_task1_steps))
        self.stale_gdino_steps = max(0, int(self.stale_gdino_steps))
        self.stale_task2_steps = max(0, int(self.stale_task2_steps))


@dataclass
class AsyncTaskRecord:
    async_id: str
    task_type: AsyncTaskType
    episode_id: str
    submit_step: int
    request_id: str
    runner: Callable[[], Any]
    request: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    status: AsyncTaskStatus = AsyncTaskStatus.PENDING
    submitted_at: float = field(default_factory=now_ts)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    future: Optional[Future] = None
    error: Optional[str] = None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return float((self.finished_at - self.started_at) * 1000.0)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "async_id": self.async_id,
            "task_type": self.task_type.value,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "request_id": self.request_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AsyncResultRecord:
    async_id: str
    task_type: AsyncTaskType
    episode_id: str
    submit_step: int
    return_step: int
    request_id: str
    result: Any = None
    request: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    success: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    finished_at: float = field(default_factory=now_ts)

    @property
    def age_steps(self) -> int:
        try:
            return max(0, int(self.return_step) - int(self.submit_step))
        except Exception:
            return 0

    def is_stale(self, config: AsyncManagerConfig) -> bool:
        if self.task_type == AsyncTaskType.TASK1:
            return self.age_steps > config.stale_task1_steps
        if self.task_type == AsyncTaskType.GDINO:
            return self.age_steps > config.stale_gdino_steps
        if self.task_type == AsyncTaskType.TASK2:
            return self.age_steps > config.stale_task2_steps
        return False

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "async_id": self.async_id,
            "task_type": self.task_type.value,
            "episode_id": self.episode_id,
            "submit_step": self.submit_step,
            "return_step": self.return_step,
            "age_steps": self.age_steps,
            "request_id": self.request_id,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "finished_at": self.finished_at,
            "metadata": self.metadata,
        }


class AsyncManager:
    """
    SVNav slow-path asynchronous task manager.

    It only runs slow functions and returns completed results.
    It does not update SemanticMap, TargetEvidence, navigation mode, UAV action,
    or StopGate. All state mutation must happen in ON_Air_SV.py main thread.
    """

    def __init__(self, config: Optional[AsyncManagerConfig] = None) -> None:
        self.config = config or AsyncManagerConfig()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.RLock()

        self._inflight: Dict[str, AsyncTaskRecord] = {}
        self._completed: List[AsyncResultRecord] = []
        self._recent_request_ids: Dict[str, float] = {}

        if self.config.enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="svnav_async",
            )

    def submit_task1(
        self,
        request: Any,
        reasoner: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AsyncTaskRecord]:
        if request is None:
            return None

        episode_id = str(getattr(request, "episode_id", ""))
        submit_step = int(getattr(request, "submit_step", 0))
        request_id = str(getattr(request, "request_id", new_id("task1")))

        def runner() -> Any:
            return reasoner.run(request=request, return_step=submit_step)

        return self._submit(
            task_type=AsyncTaskType.TASK1,
            episode_id=episode_id,
            submit_step=submit_step,
            request_id=request_id,
            runner=runner,
            request=request,
            metadata=metadata or {},
        )

    def submit_gdino(
        self,
        request: Any,
        gdino_client: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AsyncTaskRecord]:
        if request is None:
            return None

        episode_id = str(getattr(request, "episode_id", ""))
        submit_step = int(getattr(request, "submit_step", 0))
        request_id = str(getattr(request, "request_id", new_id("gdino")))

        def runner() -> Any:
            return gdino_client.detect_request(request)

        return self._submit(
            task_type=AsyncTaskType.GDINO,
            episode_id=episode_id,
            submit_step=submit_step,
            request_id=request_id,
            runner=runner,
            request=request,
            metadata=metadata or {},
        )

    def submit_task2(
        self,
        batch: Any,
        target_verifier: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AsyncTaskRecord]:
        if batch is None:
            return None

        episode_id = str(getattr(batch, "episode_id", ""))
        submit_step = int(getattr(batch, "submit_step", 0))
        batch_id = str(getattr(batch, "batch_id", new_id("task2batch")))

        def runner() -> Any:
            return target_verifier.verify_batch(
                batch=batch,
                return_step=submit_step,
            )

        return self._submit(
            task_type=AsyncTaskType.TASK2,
            episode_id=episode_id,
            submit_step=submit_step,
            request_id=batch_id,
            runner=runner,
            request=batch,
            metadata=metadata or {},
        )

    def poll_results(
        self,
        current_step: int,
        episode_id: Optional[str] = None,
    ) -> List[AsyncResultRecord]:
        current_step = int(current_step)
        episode_id = None if episode_id is None else str(episode_id)

        self._collect_finished(current_step=current_step)

        with self._lock:
            kept: List[AsyncResultRecord] = []
            returned: List[AsyncResultRecord] = []

            for record in self._completed:
                if episode_id is not None and record.episode_id != episode_id:
                    kept.append(record)
                    continue
                record.return_step = current_step
                returned.append(record)

            self._completed = kept[-self.config.max_completed_results :]

        return returned

    def cleanup_episode(self, episode_id: str) -> Dict[str, Any]:
        episode_id = str(episode_id)
        removed = []

        with self._lock:
            for async_id, record in list(self._inflight.items()):
                if record.episode_id == episode_id:
                    removed.append(record.to_log_dict())
                    self._inflight.pop(async_id, None)

            self._completed = [
                item for item in self._completed if item.episode_id != episode_id
            ]

        return {
            "episode_id": episode_id,
            "removed_inflight": len(removed),
            "removed_records": removed,
        }

    def shutdown(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python 3.8 ThreadPoolExecutor.shutdown() does not support
                # cancel_futures. UAV_ON commonly runs in Python 3.8 envs.
                executor.shutdown(wait=False)

    def to_log_dict(self) -> Dict[str, Any]:
        self._collect_finished(current_step=-1)

        with self._lock:
            inflight_by_type: Dict[str, int] = {}
            for record in self._inflight.values():
                key = record.task_type.value
                inflight_by_type[key] = inflight_by_type.get(key, 0) + 1

            completed_by_type: Dict[str, int] = {}
            for record in self._completed:
                key = record.task_type.value
                completed_by_type[key] = completed_by_type.get(key, 0) + 1

            return {
                "enabled": self.config.enabled,
                "max_workers": self.config.max_workers,
                "inflight_count": len(self._inflight),
                "inflight_by_type": inflight_by_type,
                "completed_count": len(self._completed),
                "completed_by_type": completed_by_type,
                "inflight": [
                    record.to_log_dict() for record in self._inflight.values()
                ],
            }

    def _submit(
        self,
        task_type: AsyncTaskType,
        episode_id: str,
        submit_step: int,
        request_id: str,
        runner: Callable[[], Any],
        request: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AsyncTaskRecord]:
        if not self.config.enabled or self._executor is None:
            return None

        task_type = AsyncTaskType(task_type)
        episode_id = str(episode_id)
        submit_step = int(submit_step)
        request_id = str(request_id)

        # Important: prepare_inputs() can submit a new Task1 before run()
        # gets a chance to poll completed results. Collect finished futures here
        # as well, otherwise a completed task may keep occupying inflight slots
        # and every later Task1 will be rejected.
        self._collect_finished(current_step=submit_step)

        with self._lock:
            if request_id in self._recent_request_ids:
                return None

            if not self._has_capacity_locked(task_type=task_type, episode_id=episode_id):
                return None

            async_id = new_id("async_{}".format(task_type.value))
            record = AsyncTaskRecord(
                async_id=async_id,
                task_type=task_type,
                episode_id=episode_id,
                submit_step=submit_step,
                request_id=request_id,
                runner=runner,
                request=request,
                metadata=dict(metadata or {}),
            )

            future = self._executor.submit(self._run_record, record)
            record.future = future
            record.status = AsyncTaskStatus.RUNNING
            self._inflight[async_id] = record
            self._recent_request_ids[request_id] = now_ts()

            if self.config.verbose:
                print(
                    "[SVNavAsync] submit type={} episode={} step={} request={}".format(
                        task_type.value,
                        episode_id,
                        submit_step,
                        request_id,
                    )
                )

            return record

    def _run_record(self, record: AsyncTaskRecord) -> AsyncResultRecord:
        record.started_at = now_ts()
        record.status = AsyncTaskStatus.RUNNING

        try:
            result = record.runner()
            record.finished_at = now_ts()
            record.status = AsyncTaskStatus.DONE
            return AsyncResultRecord(
                async_id=record.async_id,
                task_type=record.task_type,
                episode_id=record.episode_id,
                submit_step=record.submit_step,
                return_step=record.submit_step,
                request_id=record.request_id,
                result=result,
                request=record.request,
                metadata=dict(record.metadata or {}),
                success=True,
                error=None,
                latency_ms=record.latency_ms,
                finished_at=record.finished_at,
            )
        except Exception as exc:
            record.finished_at = now_ts()
            record.status = AsyncTaskStatus.FAILED
            record.error = "{}\n{}".format(exc, traceback.format_exc())
            return AsyncResultRecord(
                async_id=record.async_id,
                task_type=record.task_type,
                episode_id=record.episode_id,
                submit_step=record.submit_step,
                return_step=record.submit_step,
                request_id=record.request_id,
                result=None,
                request=record.request,
                metadata=dict(record.metadata or {}),
                success=False,
                error=record.error,
                latency_ms=record.latency_ms,
                finished_at=record.finished_at,
            )

    def _collect_finished(self, current_step: int) -> None:
        finished_ids: List[str] = []

        with self._lock:
            items = list(self._inflight.items())

        for async_id, record in items:
            future = record.future
            if future is None or not future.done():
                continue

            try:
                result_record = future.result()
            except Exception as exc:
                record.finished_at = now_ts()
                record.status = AsyncTaskStatus.FAILED
                record.error = "{}\n{}".format(exc, traceback.format_exc())
                result_record = AsyncResultRecord(
                    async_id=record.async_id,
                    task_type=record.task_type,
                    episode_id=record.episode_id,
                    submit_step=record.submit_step,
                    return_step=record.submit_step,
                    request_id=record.request_id,
                    result=None,
                    request=record.request,
                    metadata=dict(record.metadata or {}),
                    success=False,
                    error=record.error,
                    latency_ms=record.latency_ms,
                    finished_at=record.finished_at,
                )

            if current_step >= 0:
                result_record.return_step = int(current_step)

            finished_ids.append(async_id)

            with self._lock:
                self._completed.append(result_record)
                if len(self._completed) > self.config.max_completed_results:
                    self._completed = self._completed[-self.config.max_completed_results :]

            if self.config.verbose:
                print(
                    "[SVNavAsync] done type={} episode={} submit_step={} "
                    "return_step={} request={} success={} latency_ms={} error={}".format(
                        result_record.task_type.value,
                        result_record.episode_id,
                        result_record.submit_step,
                        result_record.return_step,
                        result_record.request_id,
                        result_record.success,
                        result_record.latency_ms,
                        result_record.error,
                    )
                )

        if finished_ids:
            with self._lock:
                for async_id in finished_ids:
                    self._inflight.pop(async_id, None)

                if len(self._recent_request_ids) > 256:
                    items = sorted(
                        self._recent_request_ids.items(),
                        key=lambda item: item[1],
                    )
                    self._recent_request_ids = dict(items[-128:])

    def _has_capacity_locked(
        self,
        task_type: AsyncTaskType,
        episode_id: str,
    ) -> bool:
        if len(self._inflight) >= self.config.max_inflight_total:
            return False

        same_episode = [
            record
            for record in self._inflight.values()
            if record.episode_id == episode_id and record.task_type == task_type
        ]

        if task_type == AsyncTaskType.TASK1:
            return len(same_episode) < self.config.max_inflight_task1

        if task_type == AsyncTaskType.GDINO:
            return len(same_episode) < self.config.max_inflight_gdino

        if task_type == AsyncTaskType.TASK2:
            return len(same_episode) < self.config.max_inflight_task2

        return True
