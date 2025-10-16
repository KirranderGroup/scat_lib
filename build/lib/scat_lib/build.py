"""Helper utilities for rebuilding the bundled scattering executables."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set

try:  # Python >=3.9
    from importlib.resources import as_file, files
except ImportError:  # pragma: no cover - Python 3.8 fallback
    from importlib_resources import as_file, files  # type: ignore

DEFAULT_PATTERNS: Sequence[str] = ("Main2.exe", "Zcontr.exe", "*.exe")


def _collect_artifacts(root: Path, patterns: Iterable[str]) -> List[Path]:
    seen: Set[Path] = set()
    results: List[Path] = []
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            results.append(candidate)
    return results


def compile_fortran(dest: Path | None = None, patterns: Sequence[str] | None = None, script: Path | None = None) -> List[Path]:
    """Compile the packaged Fortran sources and copy artifacts into *dest*."""
    package_root = Path(__file__).resolve().parent
    target_dir = dest or package_root / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)

    with as_file(files("scat_lib.fortran_src")) as src_root:
        source_dir = Path(src_root)
        build_script = script or source_dir / "build" / "script_compilation_fortran.sh"
        if not build_script.exists():
            raise FileNotFoundError(f"Cannot locate build script at {build_script}")

        env = os.environ.copy()
        command = ["bash", str(build_script)]
        try:
            subprocess.run(command, cwd=source_dir, env=env, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Required build tool was not found; ensure bash and the Fortran compiler referenced in the script are installed."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Compilation failed with exit code {exc.returncode}") from exc

        artifact_patterns = patterns if patterns else DEFAULT_PATTERNS
        produced = _collect_artifacts(source_dir, artifact_patterns)
        if not produced:
            raise RuntimeError("Compilation succeeded but no matching artifacts were produced.")

        copied: List[Path] = []
        for artifact in produced:
            destination = target_dir / artifact.name
            shutil.copy2(artifact, destination)
            copied.append(destination)
        return copied


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild scat_lib Fortran executables from bundled sources.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Directory where rebuilt executables should be copied (defaults to the package bin/ folder).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern for artifacts to copy after compilation (can be provided multiple times).",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Path to an alternative build script (defaults to the bundled script).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    patterns: Sequence[str] | None = args.pattern if args.pattern else None
    try:
        copied = compile_fortran(dest=args.dest, patterns=patterns, script=args.script)
    except Exception as exc:  # pragma: no cover - CLI error reporting
        print(f"[scat-lib-build] {exc}", file=sys.stderr)
        return 1

    for path in copied:
        print(f"[scat-lib-build] Installed {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
