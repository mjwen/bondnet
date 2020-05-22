from gnn.core.reaction import (
    ReactionsOfSameBond,
    ReactionsMultiplePerBond,
    ReactionExtractorFromMolSet,
    create_reactions_from_reactant,
)
from .utils import (
    create_reactions_nonsymmetric_reactant,
    create_symmetric_molecules,
    create_reactions_symmetric_reactant,
)


class TestReaction:
    def test_atom_mapping(self):
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{0: 0, 1: 2, 2: 1, 3: 3}]
        reaction = A2B[0]
        mapping = reaction.atom_mapping()
        assert mapping == ref_mapping

        # m0 to m2 m4
        ref_mapping = [{0: 0, 1: 2, 2: 1}, {0: 3}]
        reaction = A2BC[0]
        mapping = reaction.atom_mapping()
        assert mapping == ref_mapping

    def test_bond_mapping_int_index(self):
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{0: 1, 1: 2, 2: 3}]
        assert A2B[0].bond_mapping_by_int_index() == ref_mapping

        # m0 to m2 and m3
        ref_mapping = [{0: 1, 1: 0, 2: 2}, {}]
        assert A2BC[0].bond_mapping_by_int_index() == ref_mapping

    def test_bond_mapping_tuple_index(self):
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{(0, 1): (0, 2), (1, 2): (1, 2), (1, 3): (2, 3)}]
        assert A2B[0].bond_mapping_by_tuple_index() == ref_mapping

        # m0 to m2 and m3
        ref_mapping = [{(0, 1): (0, 2), (0, 2): (0, 1), (1, 2): (1, 2)}, {}]
        assert A2BC[0].bond_mapping_by_tuple_index() == ref_mapping

    def test_bond_mapping_sdf_int_index(self):
        """
        m0
        RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 4 4 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 C 0 1 0 0 RAD=3 VAL=2
        M  V30 2 O -1 0 0 0 RAD=2 VAL=1
        M  V30 3 N 1 0 0 0 RAD=2 VAL=2
        M  V30 4 H 1 0 1 0
        M  V30 END ATOM
        M  V30 BEGIN BOND
        M  V30 1 1 1 2
        M  V30 2 1 1 3
        M  V30 3 0 2 3
        M  V30 4 1 3 4
        M  V30 END BOND
        M  V30 END CTAB
        M  END
        $$$$

        m1
             RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 4 3 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 C 0 1 0 0 RAD=2 VAL=1
        M  V30 2 N 1 0 0 0 RAD=2 VAL=2
        M  V30 3 O -1 0 0 0 RAD=3
        M  V30 4 H 1 0 1 0
        M  V30 END ATOM
        M  V30 BEGIN BOND
        M  V30 1 1 1 2
        M  V30 2 0 2 3
        M  V30 3 1 2 4
        M  V30 END BOND
        M  V30 END CTAB
        M  END
        $$$$

        m2
             RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 3 3 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 C 0 1 0 0 RAD=3 VAL=2
        M  V30 2 N -1 0 0 0 RAD=3 VAL=1
        M  V30 3 O 1 0 0 0 RAD=2 VAL=1
        M  V30 END ATOM
        M  V30 BEGIN BOND
        M  V30 1 1 1 2
        M  V30 2 1 1 3
        M  V30 3 0 2 3
        M  V30 END BOND
        M  V30 END CTAB
        M  END
        $$$$

        m3
             RDKit          3D

          0  0  0  0  0  0  0  0  0  0999 V3000
        M  V30 BEGIN CTAB
        M  V30 COUNTS 1 0 0 0 0
        M  V30 BEGIN ATOM
        M  V30 1 H 1 0 1 0 VAL=-1
        M  V30 END ATOM
        M  V30 END CTAB
        M  END
        $$$$


        """
        A2B, A2BC = create_reactions_nonsymmetric_reactant()

        # m0 to m1
        ref_mapping = [{0: 1, 1: 2, 2: 3}]
        assert A2B[0].bond_mapping_by_sdf_int_index() == ref_mapping

        # m0 to m2 m3
        ref_mapping = [{0: 1, 1: 0, 2: 2}, {}]
        assert A2BC[0].bond_mapping_by_sdf_int_index() == ref_mapping


def test_create_reactions_from_reactant():
    def assert_A2B(rxn, products_charge, bond_energy, mol_reservoir):

        A = rxn.reactants[0]
        B = rxn.products[0]
        reactions, molecules = create_reactions_from_reactant(
            A,
            broken_bond=rxn.get_broken_bond(),
            product_charges=products_charge,
            bond_energy=bond_energy,
            mol_reservoir=mol_reservoir,
        )

        assert len(reactions) == 1
        r = reactions[0]
        assert r.get_broken_bond() == rxn.get_broken_bond()
        assert r.reactants[0] == A

        if bond_energy is None and mol_reservoir is not None:
            # reuse the product mol, which has energy
            assert r.get_free_energy() == rxn.get_free_energy()
        else:
            assert r.get_free_energy() == bond_energy

        if mol_reservoir is not None:
            assert len(molecules) == 0
            assert r.products[0] == B
        else:
            assert len(molecules) == 1
            assert r.products[0].charge == B.charge
            assert r.products[0].mol_graph.isomorphic_to(B.mol_graph)

    def assert_A2BC(rxn, products_charge, bond_energy, mol_reservoir):

        A = rxn.reactants[0]
        reactions, molecules = create_reactions_from_reactant(
            A,
            broken_bond=rxn.get_broken_bond(),
            product_charges=products_charge,
            bond_energy=bond_energy,
            mol_reservoir=mol_reservoir,
        )

        assert len(reactions) == len(products_charge)
        for i, r in enumerate(reactions):
            assert r.get_broken_bond() == rxn.get_broken_bond()
            assert r.reactants[0] == A

            p_gen = r.products
            p_ori = rxn.products
            assert p_gen[0].charge == products_charge[i][0]
            assert p_gen[1].charge == products_charge[i][1]
            assert (
                p_gen[0].mol_graph.isomorphic_to(p_ori[0].mol_graph)
                and p_gen[1].mol_graph.isomorphic_to(p_ori[1].mol_graph)
            ) or (
                p_gen[0].mol_graph.isomorphic_to(p_ori[1].mol_graph)
                and p_gen[1].mol_graph.isomorphic_to(p_ori[0].mol_graph)
            )

        if mol_reservoir is None:
            assert len(molecules) == 2 * len(products_charge)

    A2B, A2BC = create_reactions_symmetric_reactant()

    rxn = A2B[0]
    charge = rxn.products[0].charge
    assert_A2B(rxn, [[charge]], rxn.get_free_energy(), [rxn.products[0]])
    assert_A2B(rxn, [[charge]], rxn.get_free_energy(), None)
    assert_A2B(rxn, [[charge]], None, [rxn.products[0]])
    assert_A2B(rxn, [[charge]], None, None)

    rxn = A2BC[0]
    assert_A2BC(rxn, [[0, 0]], rxn.get_free_energy(), rxn.products)
    assert_A2BC(rxn, [[0, 0]], rxn.get_free_energy(), None)
    assert_A2BC(rxn, [[0, 0], [-1, 0]], None, rxn.products)
    assert_A2BC(rxn, [[0, 0], [-1, 0]], None, None)


class TestReactionsOfSameBond:
    @staticmethod
    def assert_mol(m, formula, charge):
        assert m.formula == formula
        assert m.charge == charge

    def test_create_complement_reactions_and_order_reactions(self):

        A2B, A2BC = create_reactions_symmetric_reactant()

        # A->B style
        rxn = A2B[0]
        reactant = rxn.reactants[0]

        ### test complement reactions
        # provide reaction
        rsb = ReactionsOfSameBond(reactant, A2B)
        comp_rxns, comp_mols = rsb.create_complement_reactions()
        assert len(comp_rxns) == 0
        assert len(comp_mols) == 0

        # do not provide reaction, and create itself
        rsb = ReactionsOfSameBond(reactant, broken_bond=rxn.get_broken_bond())
        comp_rxns, comp_mols = rsb.create_complement_reactions()
        assert len(comp_rxns) == 1
        assert len(comp_mols) == 1

        ### test order reactions
        ordered_rxns = rsb.order_reactions(complement_reactions=True)
        assert len(ordered_rxns) == 1
        assert ordered_rxns[0] == comp_rxns[0]

        # A->B+C
        A2BC = A2BC[:2]  # breaks same bonds
        reactant = A2BC[0].reactants[0]

        ### test complement reactions
        rsb = ReactionsOfSameBond(reactant, A2BC)
        comp_rxns, comp_mols = rsb.create_complement_reactions()
        assert len(comp_rxns) == 1
        assert len(comp_mols) == 2
        rxn = comp_rxns[0]
        self.assert_mol(rxn.reactants[0], "C1H2O2", 0)
        self.assert_mol(rxn.products[0], "C1H1O2", 1)
        self.assert_mol(rxn.products[1], "H1", -1)

        ### test order reactions
        ordered_rxns = rsb.order_reactions(complement_reactions=False)
        assert len(ordered_rxns) == 2
        assert ordered_rxns[0] == A2BC[0]
        assert ordered_rxns[1] == A2BC[1]

        ordered_rxns = rsb.order_reactions(complement_reactions=True)
        assert len(ordered_rxns) == 3
        assert ordered_rxns[0] == A2BC[0]
        assert ordered_rxns[1] == A2BC[1]
        assert ordered_rxns[2] == comp_rxns[0]


class TestReactionsMultiplePerBond:
    def test_group_by_bond(self):
        # create reactions of same reactant
        A2B, A2BC = create_reactions_symmetric_reactant()
        reactions = A2B + A2BC[:4]
        reactant = reactions[0].reactants[0]
        rmb = ReactionsMultiplePerBond(reactant, reactions)

        # one reactions per iso bond group
        rsb_group = rmb.group_by_bond(find_one=False)
        for rsb in rsb_group:
            # A2B
            if rsb.broken_bond == (3, 4):
                assert len(rsb.reactions) == 1
                assert rsb.reactions[0] == A2B[0]
            # A2BC
            elif rsb.broken_bond == (1, 2) or rsb.broken_bond == (0, 1):
                assert len(rsb.reactions) == 2
                assert rsb.reactions[0] == A2BC[0]
                assert rsb.reactions[1] == A2BC[1]
            else:
                assert rsb.reactions == []

        # all reactions for iso bond group
        rsb_group = rmb.group_by_bond(find_one=True)
        for rsb in rsb_group:
            # A2B
            if rsb.broken_bond == (3, 4):
                assert len(rsb.reactions) == 1
                assert rsb.reactions[0] == A2B[0]
            # A2BC; reactions with broken bond (1,2) are isomorphic to those (0,1)
            elif rsb.broken_bond == (0, 1):
                assert len(rsb.reactions) == 2
                assert rsb.reactions[0] == A2BC[0]
                assert rsb.reactions[1] == A2BC[1]
            else:
                assert rsb.reactions == []

    def test_order_reactions(self):
        # create reactions of same reactant
        A2B, A2BC = create_reactions_symmetric_reactant()
        reactions = A2B + A2BC[:4]
        reactant = reactions[0].reactants[0]
        rmb = ReactionsMultiplePerBond(reactant, reactions)

        ordered_rxns = rmb.order_reactions(
            complement_reactions=False, one_per_iso_bond_group=False
        )
        assert len(ordered_rxns) == 5
        # A2B
        assert ordered_rxns[0] == A2B[0]
        # A2BC
        # they have the same energy
        assert ordered_rxns[1] == ordered_rxns[2]
        assert ordered_rxns[3] == ordered_rxns[4]
        assert ordered_rxns[1] == A2BC[0] or ordered_rxns[1] == A2BC[1]
        assert ordered_rxns[2] == A2BC[0] or ordered_rxns[2] == A2BC[1]
        assert ordered_rxns[3] == A2BC[2] or ordered_rxns[3] == A2BC[3]
        assert ordered_rxns[4] == A2BC[2] or ordered_rxns[4] == A2BC[3]

        ordered_rxns = rmb.order_reactions(
            complement_reactions=False, one_per_iso_bond_group=True
        )
        assert len(ordered_rxns) == 3
        # A2B
        assert ordered_rxns[0] == A2B[0]
        # A2BC
        # they have the same energy
        assert ordered_rxns[1] == A2BC[0] or ordered_rxns[1] == A2BC[1]
        assert ordered_rxns[2] == A2BC[2] or ordered_rxns[2] == A2BC[3]

        ordered_rxns = rmb.order_reactions(
            complement_reactions=True, one_per_iso_bond_group=False
        )
        assert len(ordered_rxns) == 9
        # A2B
        assert ordered_rxns[0] == A2B[0]
        # A2BC
        # they have the same energy
        assert ordered_rxns[1] == ordered_rxns[2]
        assert ordered_rxns[3] == ordered_rxns[4]
        assert ordered_rxns[1] == A2BC[0] or ordered_rxns[1] == A2BC[1]
        assert ordered_rxns[2] == A2BC[0] or ordered_rxns[2] == A2BC[1]
        assert ordered_rxns[3] == A2BC[2] or ordered_rxns[3] == A2BC[3]
        assert ordered_rxns[4] == A2BC[2] or ordered_rxns[4] == A2BC[3]

        ordered_rxns = rmb.order_reactions(
            complement_reactions=True, one_per_iso_bond_group=True
        )
        assert len(ordered_rxns) == 5
        # A2B
        assert ordered_rxns[0] == A2B[0]
        # A2BC
        # they have the same energy
        assert ordered_rxns[1] == A2BC[0] or ordered_rxns[1] == A2BC[1]
        assert ordered_rxns[2] == A2BC[2] or ordered_rxns[2] == A2BC[3]


def test_extract_reactions_from_mol_set():
    def assert_rxns(rxns, ref, size):
        assert (len(rxns)) == size
        for r in rxns:
            assert r in ref
        for r in ref:
            assert r in rxns

    def assert_one(find_one):
        ref_A2B, ref_A2BC = create_reactions_symmetric_reactant()
        ref_size = 6
        if find_one:
            # remove reactions of isomorphic bond
            ref_A2BC = ref_A2BC[:2] + ref_A2BC[4:]
            ref_size = 4

        mols = create_symmetric_molecules()
        extractor = ReactionExtractorFromMolSet(mols)
        A2B = extractor.extract_A_to_B_style_reaction(find_one)
        A2BC = extractor.extract_A_to_B_C_style_reaction(find_one)

        assert_rxns(A2B, ref_A2B, 1)
        assert_rxns(A2BC, ref_A2BC, ref_size)

    assert_one(True)
    assert_one(False)
