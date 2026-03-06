from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from poly_csp.topology.monomers import make_glucose_template
from poly_csp.forcefield.amber_export import export_amber_artifacts


def _template_mol():
    return make_glucose_template("amylose").mol


@pytest.mark.integration
def test_export_residue_aware_backend_missing_tleap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that missing tleap raises a clear error."""
    import poly_csp.forcefield.glycam as glycam_mod

    def _raise_missing(_tools):
        raise RuntimeError(
            "GLYCAM reference extraction requires executables not found on PATH: tleap. "
            "Install AmberTools with GLYCAM06 support."
        )

    monkeypatch.setattr(glycam_mod, "_ensure_required_tools", _raise_missing)

    with pytest.raises(RuntimeError, match="executables"):
        export_amber_artifacts(
            mol=_template_mol(),
            outdir=tmp_path / "amber_missing",
            polymer="amylose",
            dp=1,
        )
