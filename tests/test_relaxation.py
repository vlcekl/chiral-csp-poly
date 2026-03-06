from __future__ import annotations

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.connectors import ConnectorParams
from poly_csp.forcefield.relaxation import RelaxSpec, run_staged_relaxation
from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def test_run_staged_relaxation_parameterizes_connectors_once_per_site(monkeypatch) -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=2))
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=1,
        site="C6",
        selector=selector,
    )

    calls: list[tuple[str, str]] = []

    def fake_parameterize(polymer, selector_template, site, **kwargs):
        calls.append((polymer, site))
        return ConnectorParams(
            polymer=polymer,
            selector_name=selector_template.name,
            site=site,
        )

    monkeypatch.setattr(
        "poly_csp.forcefield.relaxation.parameterize_capped_monomer",
        fake_parameterize,
    )
    monkeypatch.setattr(
        "poly_csp.forcefield.gaff.load_gaff2_selector_forces",
        lambda selector_prmtop_path, mol, selector_template: [],
    )

    spec = RelaxSpec(
        enabled=True,
        positional_k=10.0,
        dihedral_k=0.0,
        hbond_k=0.0,
        n_stages=1,
        max_iterations=5,
        freeze_backbone=False,
        anneal_enabled=False,
    )
    relaxed, summary = run_staged_relaxation(
        mol=mol,
        spec=spec,
        selector=selector,
        selector_prmtop_path="/tmp/mock_selector.prmtop",
    )

    assert relaxed.GetNumAtoms() == mol.GetNumAtoms()
    assert relaxed.GetNumConformers() == 1
    assert calls == [("amylose", "C6")]
    assert summary["enabled"] is True
    assert summary["force_model"] == "modular_gaff2_connectors"
