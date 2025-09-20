"""Tracking, logging and artifact management helpers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import importlib.metadata as importlib_metadata
import numpy as np

logger = logging.getLogger(__name__)

try:  # optional mlflow dependency
    import mlflow  # type: ignore

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    HAS_MLFLOW = False

try:  # optional dvc dependency
    from dvc.exceptions import DvcException
    from dvc.repo import Repo as DvcRepo

    HAS_DVC = True
except Exception:  # pragma: no cover - optional
    DvcException = Exception  # type: ignore
    DvcRepo = None  # type: ignore
    HAS_DVC = False


def serialize_mlflow_param(value: object) -> str:
    """Convert arbitrary parameter values into a string for MLflow logging."""

    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return np.array2string(np.asarray(value), separator=",")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(serialize_mlflow_param(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(
            {str(k): serialize_mlflow_param(v) for k, v in value.items()},
            sort_keys=True,
        )
    return str(value)


def version_artifacts_with_dvc(
    repo_root: Path | None, targets: Sequence[Path]
) -> None:
    """Register the provided targets with a DVC repository if available."""

    if not (HAS_DVC and repo_root):
        return
    repo_root = Path(repo_root).resolve()
    if not (repo_root / ".dvc").exists():
        logger.debug("DVC root %s is not initialized; skipping versioning", repo_root)
        return
    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        if DvcRepo is None:  # pragma: no cover - defensive
            return
        with DvcRepo(str(repo_root)) as repo:  # type: ignore[misc]
            for target in targets:
                target_path = Path(target).resolve()
                if not target_path.exists():
                    continue
                try:
                    rel_target = target_path.relative_to(repo_root)
                except ValueError:
                    logger.debug(
                        "Skipping DVC registration for %s outside of repo %s",
                        target_path,
                        repo_root,
                    )
                    continue
                try:
                    repo.add([str(rel_target)])
                except DvcException:  # pragma: no cover - best effort logging
                    logger.debug(
                        "DVC reported that %s is already tracked; skipping", rel_target
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Failed to add %s to DVC repository %s", rel_target, repo_root
                    )
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to open DVC repository at %s", repo_root)
    finally:
        os.chdir(cwd)


def write_dependency_snapshot(out_dir: Path) -> Path:
    """Record the current Python package versions."""

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        packages = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if result.returncode != 0 and not packages:
            logger.exception(
                "pip freeze returned non-zero exit code %s", result.returncode
            )
    except (
        OSError,
        subprocess.SubprocessError,
    ):  # pragma: no cover - pip not available
        logger.exception("Failed to execute pip freeze; falling back to metadata")
        packages = []
    if not packages:
        packages = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib_metadata.distributions()
        )
    dep_path = out_dir / "dependencies.txt"
    dep_path.write_text("\n".join(packages) + ("\n" if packages else ""))
    return dep_path


__all__ = [
    "HAS_DVC",
    "HAS_MLFLOW",
    "DvcRepo",
    "DvcException",
    "mlflow",
    "serialize_mlflow_param",
    "version_artifacts_with_dvc",
    "write_dependency_snapshot",
]
