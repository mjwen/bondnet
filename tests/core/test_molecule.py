from .utils import create_C2H4O1


class TestMoleucleWrapper:
    def test_isomorphic_bonds(self):
        ref_identical = [{(0, 2), (0, 3)}, {(1, 2), (3, 4), (2, 5), (3, 6)}, {(2, 3)}]

        mol = create_C2H4O1()
        iso_identical = [set(g) for g in mol.isomorphic_bonds]

        for v in iso_identical:
            assert v in ref_identical
        for v in ref_identical:
            assert v in iso_identical

    def test_is_atom_in_ring(self):
        mol = create_C2H4O1()

        for atom in [0, 2, 3]:
            assert mol.is_atom_in_ring(atom) is True
        for atom in [1, 4, 5, 6]:
            assert mol.is_atom_in_ring(atom) is False

    def test_is_bond_in_ring(self):
        mol = create_C2H4O1()

        for atom in [(0, 2), (0, 3), (2, 3)]:
            assert mol.is_bond_in_ring(atom) is True
        for atom in [(1, 2), (2, 5), (3, 4), (3, 6)]:
            assert mol.is_bond_in_ring(atom) is False
