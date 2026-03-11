from __future__ import annotations

from rdkit import Chem

from poly_csp.topology.selectors import SelectorRegistry


def test_make_35_dmpc_template_structure() -> None:
    tpl = SelectorRegistry.get("35dmpc")
    assert tpl.name == "35dmpc"
    assert tpl.mol.GetNumAtoms() > 0
    assert tpl.attach_dummy_idx is not None
    assert tpl.attach_atom_idx != tpl.attach_dummy_idx
    assert tpl.linkage_type == "carbamate"
    assert "tau_link" in tpl.dihedrals
    assert len(tpl.donors) >= 1
    assert len(tpl.acceptors) >= 1
    assert set(tpl.connector_local_roles.values()) == {
        "carbonyl_c",
        "carbonyl_o",
        "amide_n",
    }


def test_make_35_dmpc_template_attach_atom_is_carbonyl_c() -> None:
    tpl = SelectorRegistry.get("35dmpc")
    atom = tpl.mol.GetAtomWithIdx(tpl.attach_atom_idx)
    assert atom.GetAtomicNum() == 6
    has_double_o = False
    for bond in atom.GetBonds():
        nbr = bond.GetOtherAtom(atom)
        if nbr.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
    assert has_double_o


def test_make_35_dmpc_template_tracks_carbamate_nh() -> None:
    tpl = SelectorRegistry.get("35dmpc")
    amide_idx = next(
        idx for idx, role in tpl.connector_local_roles.items() if role == "amide_n"
    )
    amide_n = tpl.mol.GetAtomWithIdx(amide_idx)
    assert amide_n.GetTotalNumHs(includeNeighbors=True) == 1
