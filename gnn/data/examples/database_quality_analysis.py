import os
import copy
from collections import defaultdict
import numpy as np
import subprocess
from gnn.data.utils import TexWriter
from gnn.utils import pickle_dump, pickle_load, expand_path


def plot_mol_graph():
    def plot_one(m, prefix):
        fname = os.path.join(prefix, "{}.png".format(m.id))
        fname = expand_path(fname)
        m.draw(fname, show_atom_idx=True)
        subprocess.run(["convert", fname, "-trim", "-resize", "100%", fname])

    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    mols = pickle_load(filename)

    for m in mols:

        # mol builder
        prefix = "~/Applications/db_access/mol_builder/png_critic_builder"
        plot_one(m, prefix)

        # babel builder with extender
        m.convert_to_babel_mol_graph(use_metal_edge_extender=True)
        prefix = "~/Applications/db_access/mol_builder/png_extend_builder"
        plot_one(m, prefix)

        # babel builder
        m.convert_to_babel_mol_graph(use_metal_edge_extender=False)
        prefix = "~/Applications/db_access/mol_builder/png_babel_builder"
        plot_one(m, prefix)


def compare_connectivity_across_graph_builder(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
    # filename="~/Applications/db_access/mol_builder/molecules_n200.pkl",
    tex_file="~/Applications/db_access/mol_builder/tex_mol_connectivity.tex",
    only_different=True,
):
    """
    Plot the mol connectivity and same how different they are.
    """

    mols = pickle_load(filename)

    # ###############
    # # filter on charge
    # ###############
    # new_mols = []
    # for m in mols:
    #     if m.charge == 0:
    #         new_mols.append(m)
    # mols = new_mols

    mols_differ_graph = defaultdict(list)

    # write tex file
    tex_file = expand_path(tex_file)
    with open(tex_file, "w") as f:
        f.write(TexWriter.head())
        f.write(
            "On each page, we plot three mols (top to bottom) from mol builder, "
            "babel builder, and babel builder with metal edge extender.\n"
        )

        for m in mols:

            mol_keeper = []

            # mol builder
            m.make_picklable()
            mol_keeper.append(copy.deepcopy(m))

            # babel builder with extender
            m.convert_to_babel_mol_graph(use_metal_edge_extender=True)
            m.make_picklable()
            mol_keeper.append(copy.deepcopy(m))

            # babel builder
            m.convert_to_babel_mol_graph(use_metal_edge_extender=False)
            m.make_picklable()
            mol_keeper.append(copy.deepcopy(m))

            do_plot = True
            if only_different:
                mol_builder_graph = mol_keeper[0].mol_graph
                ob_extender_graph = mol_keeper[1].mol_graph
                if mol_builder_graph.isomorphic_to(ob_extender_graph):
                    do_plot = False

            if do_plot:
                f.write(TexWriter.newpage())
                f.write(TexWriter.verbatim("formula: " + m.formula))
                f.write(TexWriter.verbatim("charge: " + str(m.charge)))
                f.write(
                    TexWriter.verbatim("spin multiplicity: " + str(m.spin_multiplicity))
                )
                f.write(TexWriter.verbatim("free energy: " + str(m.free_energy)))
                f.write(TexWriter.verbatim("id: " + m.id))

                for i, m in enumerate(mol_keeper):
                    if i == 0:
                        p = "png_critic_builder"
                    elif i == 1:
                        p = "png_extend_builder"
                    elif i == 2:
                        p = "png_babel_builder"
                    fname = os.path.join(
                        "~/Applications/db_access/mol_builder", p, "{}.png".format(m.id)
                    )

                    f.write(TexWriter.single_figure(fname))
                    f.write(TexWriter.verbatim("=" * 80))

                mols_differ_graph["critic_builder"].append(mol_keeper[0])
                mols_differ_graph["extend_builder"].append(mol_keeper[1])
                mols_differ_graph["babel_builder"].append(mol_keeper[2])

        f.write(TexWriter.tail())

        filename = "~/Applications/db_access/mol_builder/molecules_critic_builder.pkl"
        pickle_dump(mols_differ_graph["critic_builder"], filename)
        filename = "~/Applications/db_access/mol_builder/molecules_extend_builder.pkl"
        pickle_dump(mols_differ_graph["extend_builder"], filename)
        filename = "~/Applications/db_access/mol_builder/molecules_babel_buidler.pkl"
        pickle_dump(mols_differ_graph["babel_builder"], filename)


def check_mol_valence(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
    # filename="~/Applications/db_access/mol_builder/molecules_critic_builder.pkl"
):
    """
    Check the valence of each atom, without considering their bonding to Li,
    since elements can form coordination bond with Li.

    For Li itself, we experiment with the bonds it has and see how it works.
    """

    def get_atom_bonds(m):
        """
        Returns:
            A list of tuple (atom species, bonded atom species),
            where `bonded_atom_species` is a list.
            Each tuple represents an atom and its bonds.
        """
        res = [(attr["specie"], []) for _, attr in m.atoms]
        for a1, a2, _ in m.bonds:
            s1 = m.atoms[a1]["specie"]
            s2 = m.atoms[a2]["specie"]
            res[a1][1].append(s2)
            res[a2][1].append(s1)
        return res

    Li_allowed = [1, 2]
    allowed_charge = {
        "H": [1],
        "C": [1, 2, 3, 4],
        "O": [1, 2],
        "F": [1],
        "P": [1, 2, 3, 5],
        "Li": Li_allowed,
    }

    failed = []
    reason = []
    mols = pickle_load(filename)
    for m in mols:
        bonds = get_atom_bonds(m)
        for atom_specie, bonded_atom_species in bonds:

            if len(bonded_atom_species) == 0 and len(bonds) == 1:
                print("@@@@@Error@@@@@@ not connected atom in mol:", m.id)

            removed_Li = [s for s in bonded_atom_species if s != "Li"]
            num_bonds = len(removed_Li)

            if num_bonds == 0:  # fine since we removed Li bonds
                continue

            if num_bonds not in allowed_charge[atom_specie]:
                failed.append(m)
                reason.append([atom_specie, num_bonds])

    print("@@@ Failed `check_mol_valence()`")
    print("@@@ number of entries failed:", len(failed))
    print("idx    id    atom specie    num bonds (without considering Li)")
    for i, (m, r) in enumerate(zip(failed, reason)):
        print(i, m.id, r)
    filename = "~/Applications/db_access/mol_builder/failed_check_mol_valence.pkl"
    pickle_dump(failed, filename)


def check_bond_species(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
    # filename="~/Applications/db_access/mol_builder/molecules_critic_builder.pkl"
):
    """
    Check the species of atoms associated with a bond. Certain bond (e.g. Li-H) fail
    the check.

    """

    def get_bond_species(m):
        """
        Returns:
            A list of the two species associated with each bonds in the molecule.
        """
        res = []
        for a1, a2, _ in m.bonds:
            s1 = m.atoms[a1]["specie"]
            s2 = m.atoms[a2]["specie"]
            res.append(sorted([s1, s2]))
        return res

    not_allowed = [("Li", "H"), ("Li", "Li")]
    not_allowed = [sorted(i) for i in not_allowed]

    failed = []
    reason = []
    mols = pickle_load(filename)
    for m in mols:
        bond_species = get_bond_species(m)
        for b in bond_species:
            if b in not_allowed:
                failed.append(m)
                reason.append(b)

    print("@@@ Failed `check_bond_species()`")
    print("@@@ number of entries failed:", len(failed))
    print("index    id     reason")
    for i, (m, r) in enumerate(zip(failed, reason)):
        print(i, m.id, r)
    filename = "~/Applications/db_access/mol_builder/failed_check_bond_species.pkl"
    pickle_dump(failed, filename)


def check_bond_length(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
    # filename="~/Applications/db_access/mol_builder/molecules_critic_builder.pkl"
):
    """
    Check the species of atoms associated with a bond. Certain bond (e.g. Li-H) fail
    the check.

    """

    def get_bond_lengthes(m):
        """
        Returns:
            A list of tuple (species, length), where species are the two species
            associated with a bond and length is the corresponding bond length.
        """
        res = []
        for a1, a2, _ in m.bonds:
            s1 = m.species[a1]
            s2 = m.species[a2]
            c1 = np.asarray(m.coords[a1])
            c2 = np.asarray(m.coords[a2])
            length = np.linalg.norm(c1 - c2)
            res.append((tuple(sorted([s1, s2])), length))
        return res

    #
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies
    # https://www.chem.tamu.edu/rgroup/connell/linkfiles/bonds.pdf
    #
    bond_length_limit = {
        # H
        ("H", "H"): 0.74,
        ("H", "C"): 1.09,
        ("H", "O"): 0.96,
        ("H", "Li"): None,
        ("H", "F"): 0.92,
        ("H", "P"): 1.44,
        # C
        ("C", "C"): 1.54,
        ("C", "O"): 1.43,
        ("C", "Li"): None,
        ("C", "F"): 1.35,
        ("C", "P"): 1.84,
        # O
        ("O", "O"): 1.48,
        ("O", "Li"): None,
        ("O", "F"): 1.42,
        ("O", "P"): 1.63,
        # Li
        ("Li", "Li"): None,
        ("Li", "F"): None,
        ("Li", "P"): None,
        # F
        ("F", "F"): 1.42,
        ("F", "P"): 1.54,
        # P
        ("P", "P"): 2.21,
    }

    # multiply by 1.2 to give relax the rule a bit
    tmp = dict()
    for k, v in bond_length_limit.items():
        if v is not None:
            v *= 1.2
        tmp[tuple(sorted(k))] = v
    bond_length_limit = tmp

    failed = []
    reason = []
    mols = pickle_load(filename)
    for m in mols:
        bond_species = get_bond_lengthes(m)
        for b, length in bond_species:
            limit = bond_length_limit[b]
            if limit is not None and length > limit:
                failed.append(m)
                reason.append("{}  {} ({})".format(b, length, limit))

    print("@@@ Failed `check_bond_length()`")
    print("@@@ number of entries failed:", len(failed))
    print("index    id     bond     length (limit)")
    for i, (m, r) in enumerate(zip(failed, reason)):
        print(i, m.id, r)
    filename = "~/Applications/db_access/mol_builder/failed_check_bond_length.pkl"
    pickle_dump(failed, filename)


if __name__ == "__main__":

    # plot_mol_graph()
    # compare_connectivity_across_graph_builder()
    # check_mol_valence()
    # check_bond_species()
    check_bond_length()
