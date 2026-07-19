from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self, timeout_seconds: float) -> dict[str, Any]:
        if (
            self.status == "running"
            and self.started_at is not None
            and time.time() - self.started_at > timeout_seconds
        ):
            display_status = "failed"
            error = f"Simulation exceeded {timeout_seconds:g} seconds"
        else:
            display_status = self.status
            error = self.error
        return {
            "job_id": self.job_id,
            "status": display_status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": error,
        }


class JobService:
    def __init__(self, timeout_seconds: float = 60) -> None:
        self.timeout_seconds = timeout_seconds
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mvp-simulation"
        )
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def submit(self, function: Callable[[], dict[str, Any]]) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self.lock:
            self.jobs[job.job_id] = job

        def execute() -> None:
            job.status = "running"
            job.started_at = time.time()
            try:
                job.result = function()
                job.status = "completed"
            except Exception as error:  # surfaced through the API
                job.error = str(error)
                job.status = "failed"
            finally:
                job.finished_at = time.time()

        self.executor.submit(execute)
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
