from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mcpmft.args import ProjectConfig
from mcpmft.utils.io import atomic_write_text


PROJECT_CONFIG_NAME = "project_config.yaml"


def write_project_config(config: dict[str, Any], output_dir: str | Path) -> None:
    atomic_write_text(
        Path(output_dir) / PROJECT_CONFIG_NAME,
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )


def validate_run_directory(config: ProjectConfig) -> None:
    output_dir = Path(config.train.output_dir).expanduser()
    checkpoint = config.runtime.resume_from_checkpoint
    if checkpoint:
        if not Path(checkpoint).expanduser().is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"Fresh training requires an empty output directory: {output_dir}")
