from .utils import create_C2H4O1


def test_iso_identical_bonds():
    # {0:3} first because products are ordered
    ref_identical = [{(0, 2), (0, 3)}, {(1, 2), (3, 4), (2, 5), (3, 6)}]

    mol = create_C2H4O1()
    iso_identical = mol.iso_identical_bonds

    for v in iso_identical:
        assert v in ref_identical
    for v in ref_identical:
        assert v in iso_identical
