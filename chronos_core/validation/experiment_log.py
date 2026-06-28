"""
Experiment Log Module
=====================

Records validation run history persistently for the Chronos-Self validation system.
Provides structured logging of experiment runs, comparison between runs,
and a context manager for automatic timing and recording.

Usage:
    from chronos_core.validation.experiment_log import ExperimentLog, ExperimentContext

    log = ExperimentLog()
    with ExperimentContext(log, config={...}, param_changes="", fix_notes="") as ctx:
        # run validation...
        ctx.record = {"metrics": {"overall_passed": True, ...}}
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_experiment_id() -> str:
    """Generate a short, human-readable experiment ID.

    Format: "exp_YYYYMMDD_NNN" where NNN is a zero-padded counter
    derived from the current time to ensure uniqueness within a day.

    Returns:
        A string like "exp_20260628_001".
    """
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y%m%d")
    suffix = now.strftime("%f")[:3]
    return f"exp_{date_part}_{suffix}"


def get_git_commit() -> Optional[str]:
    """Try to retrieve the current git commit hash.

    Returns:
        The full commit hash string, or None if not available
        (e.g., not a git repository, git not installed).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Could not retrieve git commit: %s", exc)
    return None


@dataclass
class ExperimentRecord:
    """A single experiment run record.

    Captures all relevant metadata and results from one validation execution.
    """

    experiment_id: str
    timestamp: str
    duration_s: float
    config: dict
    metrics: dict
    param_changes: Optional[str] = None
    fix_notes: Optional[str] = None
    profilation_data: Optional[dict] = None
    git_commit: Optional[str] = None


class ExperimentLog:
    """Persistent experiment log backed by a JSON file.

    Appends records to a JSON array file, provides history retrieval,
    pairwise comparison, and latest-record access.
    """

    LOG_PATH = "validation_results/experiment_log.json"

    def __init__(self, log_path: str = LOG_PATH):
        """Initialize the experiment log.

        Args:
            log_path: Path to the JSON log file, relative to the
                      working directory. Defaults to
                      "validation_results/experiment_log.json".
        """
        self._log_path = Path(log_path)

    @property
    def log_path(self) -> Path:
        return self._log_path

    def append(self, record: ExperimentRecord) -> None:
        """Append a record to the experiment log file.

        Creates the file and parent directories if they do not exist.
        Appends to the JSON array in-place.

        Args:
            record: The experiment record to persist.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        records: List[dict] = []
        if self._log_path.exists():
            try:
                with open(self._log_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning(
                    "Failed to read existing log file, starting fresh: %s", exc
                )

        records.append(asdict(record))

        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(
            "Appended experiment %s (%.2f s) to %s",
            record.experiment_id,
            record.duration_s,
            self._log_path,
        )

    def get_history(self) -> List[ExperimentRecord]:
        """Read and return all experiment records from the log file.

        Returns:
            A list of ExperimentRecord instances, in insertion order
            (oldest first). Returns an empty list if the file does not
            exist or is malformed.
        """
        if not self._log_path.exists():
            return []

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.error("Failed to read experiment log: %s", exc)
            return []

        records = []
        for item in data:
            try:
                records.append(ExperimentRecord(**item))
            except TypeError as exc:
                logger.warning("Skipping malformed record: %s", exc)
        return records

    def compare(self, id_a: str, id_b: str) -> Dict[str, Any]:
        """Compare two experiment runs by their IDs.

        Computes the difference in metrics and configuration between
        two recorded runs.

        Args:
            id_a: Experiment ID of the first run.
            id_b: Experiment ID of the second run.

        Returns:
            A dictionary with two keys:
                - "metrics_diff": dict mapping each metric key to the
                  signed difference (value_b - value_a).
                - "config_diff": dict containing keys that differ with
                  their values from both runs {"a": ..., "b": ...}.

        Raises:
            ValueError: If either experiment ID is not found.
        """
        history = self.get_history()
        rec_a = next((r for r in history if r.experiment_id == id_a), None)
        rec_b = next((r for r in history if r.experiment_id == id_b), None)

        if rec_a is None:
            raise ValueError(f"Experiment ID '{id_a}' not found in log")
        if rec_b is None:
            raise ValueError(f"Experiment ID '{id_b}' not found in log")

        metrics_diff: Dict[str, Any] = {}
        all_metrics_keys = set(rec_a.metrics.keys()) | set(rec_b.metrics.keys())
        for key in all_metrics_keys:
            val_a = rec_a.metrics.get(key)
            val_b = rec_b.metrics.get(key)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                metrics_diff[key] = round(val_b - val_a, 6)
            else:
                metrics_diff[key] = {"a": val_a, "b": val_b}

        config_diff: Dict[str, Any] = {}
        all_config_keys = set(rec_a.config.keys()) | set(rec_b.config.keys())
        for key in all_config_keys:
            val_a = rec_a.config.get(key)
            val_b = rec_b.config.get(key)
            if val_a != val_b:
                config_diff[key] = {"a": val_a, "b": val_b}

        return {
            "metrics_diff": metrics_diff,
            "config_diff": config_diff,
        }

    def get_latest(self) -> Optional[ExperimentRecord]:
        """Return the most recent experiment record.

        Returns:
            The latest ExperimentRecord, or None if the log is empty.
        """
        history = self.get_history()
        return history[-1] if history else None


class ExperimentContext:
    """Context manager for automatically timing and recording experiments.

    Usage:

        log = ExperimentLog()
        config = {"mode": "full", "level": "P0"}
        with ExperimentContext(log, config, param_changes="tuned lr=1e-4") as ctx:
            # ... run validation ...
            ctx.record = {
                "metrics": {
                    "open_loop_passed": True,
                    "overall_score": 0.92,
                    ...
                }
            }
        # Record is automatically appended on exit.
    """

    def __init__(
        self,
        log: ExperimentLog,
        config: dict,
        param_changes: str = "",
        fix_notes: str = "",
    ):
        """Initialize the context manager.

        Args:
            log: The ExperimentLog instance to write to.
            config: Validation configuration dictionary.
            param_changes: Description of parameter changes since last run.
            fix_notes: Description of bug fixes applied before this run.
        """
        self._log = log
        self._config = config
        self._param_changes = param_changes or None
        self._fix_notes = fix_notes or None
        self._start_time: Optional[float] = None
        self._metrics: Optional[dict] = None

    @property
    def record(self) -> Optional[dict]:
        """Get the current metrics dictionary."""
        return self._metrics

    @record.setter
    def record(self, value: dict) -> None:
        """Set metrics from a dictionary.

        The dictionary should contain a "metrics" key with the actual
        metrics values, or be the metrics dict directly.

        Args:
            value: Either {"metrics": {...}} or {...} directly.
        """
        if isinstance(value, dict) and "metrics" in value:
            self._metrics = value["metrics"]
        else:
            self._metrics = value

    def __enter__(self) -> "ExperimentContext":
        """Record the start time and return self."""
        self._start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Create an ExperimentRecord and append it to the log.

        If an exception occurred inside the context, the duration is
        still recorded but the metrics may be incomplete.
        """
        duration = time.time() - self._start_time if self._start_time else 0.0
        now_iso = datetime.now(timezone.utc).isoformat()

        record = ExperimentRecord(
            experiment_id=generate_experiment_id(),
            timestamp=now_iso,
            duration_s=round(duration, 3),
            config=self._config,
            metrics=self._metrics or {},
            param_changes=self._param_changes,
            fix_notes=self._fix_notes,
            profilation_data=None,
            git_commit=get_git_commit(),
        )

        self._log.append(record)