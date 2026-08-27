"""Capture environment metadata so every result set is reproducible.

The Phase-0 baseline's whole authority rests on being reproducible: a number is
only useful next to the Pyomo commit, package versions, and machine it came
from.  ``collect()`` returns a JSON-serializable dict embedded in every results
file.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any, Dict, Optional


# Packages whose versions materially affect the numbers.
_TRACKED = [
    "Pyomo",
    "numpy",
    "scipy",
    "highspy",
    "gurobipy",
    "linopy",
    "xarray",
    "pandas",
]


def _pkg_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except Exception:
        return None


def _git_commit(repo: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _pyomo_repo_root() -> Optional[str]:
    try:
        import pyomo

        # pyomo/__init__.py -> repo is two levels up
        return os.path.dirname(os.path.dirname(os.path.abspath(pyomo.__file__)))
    except Exception:
        return None


def collect() -> Dict[str, Any]:
    repo = _pyomo_repo_root()
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "pyomo_repo": repo,
        "pyomo_commit": _git_commit(repo) if repo else None,
        "packages": {name: _pkg_version(name) for name in _TRACKED},
    }
    # Note whether Pyomo is an editable/dev install (the point of the baseline).
    try:
        import pyomo

        info["pyomo_file"] = os.path.abspath(pyomo.__file__)
    except Exception:
        info["pyomo_file"] = None
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(collect(), indent=2))
