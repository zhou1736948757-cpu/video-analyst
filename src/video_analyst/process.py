"""Safe subprocess helpers — never use shell=True."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    args: list[str | Path],
    *,
    timeout: float | None = None,
    check: bool = False,
    cwd: Path | None = None,
) -> CommandResult:
    """Run an external command without a shell."""
    str_args = [str(a) for a in args]
    completed = subprocess.run(
        str_args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        shell=False,
        check=False,
    )
    result = CommandResult(
        args=str_args,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if check and not result.ok:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result
