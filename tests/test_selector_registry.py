from __future__ import annotations

from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.selectors import SelectorRegistry, selector_from_smiles


def test_selector_registry_register_and_get() -> None:
    template = make_35_dmpc_template()
    try:
        SelectorRegistry.register(template)
    except ValueError:
        pass

    got1 = SelectorRegistry.get("35dmpc")
    got2 = SelectorRegistry.get("dmpc_35")
    assert got1.name == "35dmpc"
    assert got2.name == "35dmpc"
    assert got1.attach_atom_idx == template.attach_atom_idx


def test_selector_from_smiles_detects_implicit_h_donors() -> None:
    tpl = selector_from_smiles(
        name="implicit_nh",
        smiles="[*:1][C:2](=[O:3])[NH:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1",
        attach_atom_idx=1,
        attach_dummy_idx=0,
        dihedrals={},
    )
    assert tpl.donors
