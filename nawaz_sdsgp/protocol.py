"""Python interface to the Lattigo backend used by the HE-PoE comparison.

The backend sums encrypted PoE statistics. Local GP fitting and prediction
are performed by the notebook before this call."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np


def lattigo_secure_sum(
    party_values: np.ndarray,
    executable: str | Path | None = None,
    workers: int = 4,
    timeout: float = 900.0,
) -> tuple[np.ndarray, dict]:
    """Sum a (parties, slots) array through the Lattigo JSON interface.

    The backend performs collective key generation, encryption, addition, and
    collective decryption. Return the decoded sum and timing metadata."""
    values = np.asarray(party_values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("party_values must have shape (parties, slots)")
    if not np.isfinite(values).all():
        raise ValueError("party_values contains NaN or infinity")
    if workers < 1:
        raise ValueError("workers must be positive")
    if executable is None:
        configured_backend = os.environ.get("NAWAZ_LATTIGO_BACKEND")
        if configured_backend:
            executable = Path(configured_backend)
        else:
            name = "nawaz_lattigo.exe" if os.name == "nt" else "nawaz_lattigo"
            executable = Path(__file__).parent / "lattigo_backend" / name
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(
            f"Lattigo helper not found at {executable}. Build it with "
            "'go build -o nawaz_lattigo.exe .' in nawaz_sdsgp/lattigo_backend."
        )
    process = subprocess.run(
        [str(executable)],
        input=json.dumps({"values": values.tolist(), "workers": int(workers)}),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"Lattigo helper failed with code {process.returncode}:\n{process.stderr}"
        )
    result = json.loads(process.stdout)
    decoded = np.asarray(result.pop("values"), dtype=np.float64)
    if decoded.shape != (values.shape[1],):
        raise RuntimeError(f"Lattigo returned unexpected shape {decoded.shape}")
    result["cryptographically_secure"] = True
    result["backend"] = "Lattigo v6.2.0 multiparty CKKS"
    return decoded, result
