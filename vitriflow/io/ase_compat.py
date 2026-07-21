from __future__ import annotations

"""Small, explicit compatibility shims for supported ASE releases."""

import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence


@lru_cache(maxsize=1)
def lammps_data_atom_style_keyword() -> str:
    """Return the non-deprecated atom-style keyword supported by this ASE.

    ASE 3.23 introduced ``atom_style`` while retaining ``style`` as a
    deprecated alias.  ASE 3.22, which remains in VitriFlow's supported
    dependency range, exposes only ``style``.  Inspecting the actual format
    reader is more reliable than comparing version strings (including vendor
    backports) and avoids emitting a FutureWarning on current ASE.
    """

    from ase.io.lammpsdata import read_lammps_data

    parameters = inspect.signature(read_lammps_data).parameters
    if "atom_style" in parameters:
        return "atom_style"
    if "style" in parameters:
        return "style"
    raise RuntimeError(
        "Installed ASE LAMMPS-data reader exposes neither 'atom_style' nor "
        "the legacy 'style' keyword"
    )


def ase_read_lammps_data(
    path: str | Path,
    *,
    atom_style: str,
    specorder: Optional[Sequence[str]] = None,
    units: Optional[str] = None,
) -> Any:
    """Read LAMMPS data using the correct ASE keyword for the installed API."""

    from ase.io import read as ase_read

    kwargs: dict[str, Any] = {
        "format": "lammps-data",
        lammps_data_atom_style_keyword(): str(atom_style),
    }
    if specorder is not None:
        # ``specorder`` is an ASE LAMMPS *writer* argument, not a reader
        # argument.  The reader's stable API is the explicit type-to-atomic-
        # number mapping ``Z_of_type``.  Passing specorder used to raise a
        # TypeError and silently drive callers into their fallback parser.
        from ase.data import atomic_numbers

        mapping: dict[int, int] = {}
        for type_id, symbol_raw in enumerate(specorder, start=1):
            symbol = str(symbol_raw)
            try:
                atomic_number = int(atomic_numbers[symbol])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unknown chemical symbol in LAMMPS type mapping: {symbol!r}"
                ) from exc
            mapping[int(type_id)] = atomic_number
        kwargs["Z_of_type"] = mapping
    if units is not None:
        kwargs["units"] = str(units)
    return ase_read(str(path), **kwargs)


__all__ = ["ase_read_lammps_data", "lammps_data_atom_style_keyword"]
