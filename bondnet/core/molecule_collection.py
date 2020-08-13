import copy
import shutil
import logging
import numpy as np
import itertools
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt
from bondnet.core.molwrapper import create_rdkit_mol_from_mol_graph
from bondnet.analysis.utils import TexWriter
from bondnet.utils import pickle_dump, pickle_load, to_path, create_directory

logger = logging.getLogger(__name__)


class MoleculeCollection:
    """
    A collection of molecules and operations on them.

    Args:
        molecules (list): a sequence of :meth:`MoleculeWrapper` molecules.
    """

    def __init__(self, molecules):
        self.molecules = molecules

    @classmethod
    def from_file(cls, filename):
        molecules = pickle_load(to_path(filename))
        logger.info(f"{len(molecules)} molecules loaded from file: {filename}")

        return cls(molecules)

    def to_file(self, filename):
        pickle_dump(self.molecules, filename)

    def print_single_atom_property(self):

        single_atom = defaultdict(list)
        for m in self.molecules:
            if m.num_atoms == 1:
                single_atom[m.formula].append((m.charge, m.free_energy, m.id))

        print("# formula    charge    free energy    id")
        for k, v in single_atom.items():
            print(k)
            for c, e, i in v:
                print(f"             {c}        {e}    {i}")

    def get_species(self):
        """
        Returns:
             list: species in all molecules.
        """
        species = set()
        for m in self.molecules:
            species.update(m.species)

        return list(species)

    def get_molecule_counts_by_charge(self):
        """
        Returns:
            dict: {charge, counts} molecule counts by charge.
        """
        counts = defaultdict(int)
        for m in self.molecules:
            counts[m.charge] += 1

        return counts

    def plot_molecules(self, prefix=Path.cwd()):
        """
        Plot molecules to .png and write .sdf and .pdb files.

        Args:
            prefix (Path): directory path for the created files.
        """

        prefix = to_path(prefix)
        if prefix.exists():
            if not prefix.is_dir():
                raise ValueError(
                    f"Expect `prefix` be a path to a directory, but got {prefix}"
                )
        else:
            create_directory(prefix, path_is_directory=True)
        for i in ["png", "sdf", "pdb"]:
            create_directory(prefix.joinpath(f"mol_{i}"), path_is_directory=True)
            create_directory(prefix.joinpath(f"mol_{i}_id"), path_is_directory=True)

        for m in self.molecules:

            fname1 = prefix.joinpath(
                "mol_png/{}_{}_{}_{}.png".format(
                    m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
                ),
            )
            m.draw(filename=fname1, show_atom_idx=True)
            fname2 = prefix.joinpath(
                "mol_png_id/{}_{}_{}_{}.png".format(
                    m.id, m.formula, m.charge, str(m.free_energy).replace(".", "dot")
                ),
            )
            shutil.copyfile(fname1, fname2)

            for ext in ["sdf", "pdb"]:
                fname1 = prefix.joinpath(
                    "mol_{}/{}_{}_{}_{}.{}".format(
                        ext,
                        m.formula,
                        m.charge,
                        m.id,
                        str(m.free_energy).replace(".", "dot"),
                        ext,
                    ),
                )
                m.write(fname1, format=ext)
                fname2 = prefix.joinpath(
                    "mol_{}_id/{}_{}_{}_{}.{}".format(
                        ext,
                        m.id,
                        m.formula,
                        m.charge,
                        str(m.free_energy).replace(".", "dot"),
                        ext,
                    ),
                )
                create_directory(fname2)
                shutil.copyfile(fname1, fname2)

    def plot_atom_distance_histogram(
        self, bond=True, filename="atom_dist_hist.pdf", **kwargs
    ):
        """
        Plot histogram of atom distances (bond lengths).

        Args:
            filename (Path): path to the created plot.
            bond (bool): If `True`, only consider bond length; otherwise consider all
                atom pair distances.
            kwargs: keyword args passed to matplotlib.pyplot.hist.
        """

        def plot_hist(data, filename, **kwargs):
            fig = plt.figure()
            ax = fig.gca()
            ax.hist(data, **kwargs)

            ax.set_xlabel("Atom distance length")
            ax.set_ylabel("counts")

            fig.savefig(filename, bbox_inches="tight")

        def get_distances(m, bond):
            if bond:
                dist = [
                    np.linalg.norm(m.coords[u] - m.coords[v]) for (u, v), _ in m.bonds
                ]
            else:
                dist = [
                    np.linalg.norm(m.coords[u] - m.coords[v])
                    for u, v in itertools.combinations(range(m.num_atoms), 2)
                ]
            return dist

        data = [get_distances(m, bond) for m in self.molecules]
        data = np.concatenate(data)

        print("\n\n### atom distance min={}, max={}".format(min(data), max(data)))

        filename = to_path(filename)
        plot_hist(data, filename, **kwargs)

    def filter_by_species(self, species):
        """
        Remove molecules containing the given species.

        Args:
            species (list of str): atom specie string.
        """

        def check_one(m, species):
            for s in species:
                if s in m.species:
                    return True, s
            return False, None

        succeed = []
        fail = []
        reason = []

        for m in self.molecules:
            do_fail, rsn = check_one(m, species)
            if do_fail:
                fail.append(m)
                reason.append(rsn)
            else:
                succeed.append(m)

        print("#" * 80)
        print("### Molecules removed by `filter_by_species()`")
        print("### number of entries failed:", len(fail))
        print("index    id     species")
        for i, (m, r) in enumerate(zip(fail, reason)):
            print(i, m.id, r)

        self.molecules = succeed

        return succeed

    def filter_by_bond_species(
        self, not_allowed=[("Li", "H"), ("Li", "Li"), ("Mg", "Mg"), ("H", "Mg")],
    ):
        """
        Remove molecules with bonds formed between species that typically violate
        chemical rules.

        Args:
            not_allowed (list of tuple): bonds violate chemical rules.
        """

        succeed = []
        fail = []
        reason = []

        for m in self.molecules:
            do_fail, rsn = check_bond_species(m, not_allowed)
            if do_fail:
                fail.append(m)
                reason.append(rsn)
            else:
                succeed.append(m)

        print("#" * 80)
        print("### Molecules removed by `filter_by_bond_species()`")
        print("### number of entries failed:", len(fail))
        print("index    id     reason")
        for i, (m, r) in enumerate(zip(fail, reason)):
            print(i, m.id, r)

        self.molecules = succeed

        return succeed

    def filter_by_bond_length(self, bond_length_limit=None):
        """
        Remove molecules with large bond lengths. The default bond lengths are given in
        function `check_bond_length()`.

        Args:
            bond_length_limit (dict or None): {(specie, specie): limit}. Limit of bond
                length between two species. If `None`, use the internally defined one.
        """

        succeed = []
        fail = []
        reason = []

        for m in self.molecules:
            do_fail, rsn = check_bond_length(m, bond_length_limit)
            if do_fail:
                fail.append(m)
                reason.append(rsn)
            else:
                succeed.append(m)

        print("#" * 80)
        print("### Molecules removed by `filter_by_bond_length()`")
        print("### number of entries failed:", len(fail))
        print("index    id     bond     length (limit)")
        for i, (m, r) in enumerate(zip(fail, reason)):
            print(i, m.id, r)

        self.molecules = succeed

        return succeed

    def filter_by_connectivity(self, allowed_connectivity=None, exclude_species=None):
        """
        Filter molecules by their connectivity. Molecules having connectivity larger
        than allowed are removed.

        Args:
            allowed_connectivity (dict or None): {specie, [connectivity]}. Allowed
            connectivity by specie. If None, use internally defined connectivity.
            exclude_species (list of str or None): bond formed with species given in
                this list are ignored when counting the connectivity of an atom.
        """

        succeed = []
        fail = []
        reason = []

        for m in self.molecules:
            do_fail, rsn = check_connectivity(m, allowed_connectivity, exclude_species)
            if do_fail:
                fail.append(m)
                reason.append(rsn)
            else:
                succeed.append(m)

        print("#" * 80)
        print("### Molecules removed by `filter_by_connectivity()`")
        print("### number of entries failed:", len(fail))
        print("idx    id    atom specie    num bonds (without considering Li)")
        for i, (m, r) in enumerate(zip(fail, reason)):
            print(i, m.id, r)

        self.molecules = succeed

        return succeed

    def filter_by_rdkit_sanitize(self):
        """
        Check whether a molecule can be converted to a rdkit molecule. If not, remove it.
        """

        succeed = []
        fail = []
        reason = []

        for m in self.molecules:
            try:
                create_rdkit_mol_from_mol_graph(m.mol_graph, force_sanitize=True)
                succeed.append(m)
            except Exception as e:
                fail.append(m)
                reason.append(str(e))

        print("#" * 80)
        print("### Molecules removed by `filter_by_rdkit_sanitize()`")
        print("### number of entries failed:", len(fail))
        print("idx    id    failing_reason")
        for i, (m, r) in enumerate(zip(fail, reason)):
            print(i, m.id, r)

        self.molecules = succeed

        return succeed

    def filter_by_property(self, key, value):
        """
        Filter molecules by a property. The property is specified in `key` and the
            value is specified in `value`. Molecules with `m.key = value` are kept and
            others are removed.
            
        Args:
            key (str): molecule property to filter
            value: value the property
        """
        succeed = []

        for m in self.molecules:
            if getattr(m, key) == value:
                succeed.append(m)

        self.molecules = succeed

        return succeed

    # TODO this needs to be factored.
    def write_group_isomorphic_to_file(self, filename):
        """Write molecules statistics"""

        def group_isomorphic(molecules):
            """
            Group molecules
            Args:
                molecules: a list of Molecules.

            Returns:
                A list of list, with inner list of isomorphic molecules.
            """
            groups = []
            for m in molecules:
                find_iso = False
                for g in groups:
                    iso_m = g[0]
                    if m.mol_graph.isomorphic_to(iso_m.mol_graph):
                        g.append(m)
                        find_iso = True
                        break
                if not find_iso:
                    groups.append([m])
            return groups

        groups = group_isomorphic(self.molecules)

        # statistics or charges of mols
        charges = defaultdict(int)
        for m in self.molecules:
            charges[m.charge] += 1

        # statistics of isomorphic mols
        sizes = defaultdict(int)
        for g in groups:
            sizes[len(g)] += 1

        # statistics of charge combinations
        charge_combinations = defaultdict(int)
        for g in groups:
            chg = [m.charge for m in g]
            for ij in itertools.combinations(chg, 2):
                ij = tuple(sorted(ij))
                charge_combinations[ij] += 1

        create_directory(filename)
        with open(to_path(filename), "w") as f:
            f.write("Number of molecules: {}\n\n".format(len(self.molecules)))
            f.write("Molecule charge state statistics.\n")
            f.write("# charge state     number of molecules:\n")
            for k, v in charges.items():
                f.write("{}    {}\n".format(k, v))

            f.write("Number of isomorphic groups: {}\n\n".format(len(groups)))
            f.write(
                "Molecule isomorphic group size statistics. (i.e. the number of "
                "isomorphic molecules that have a specific number of charge state\n"
            )
            f.write("# size     number of molecules:\n")
            for k, v in sizes.items():
                f.write("{}    {}\n".format(k, v))

            f.write("# charge combinations     number:\n")
            for k, v in charge_combinations.items():
                f.write("{}    {}\n".format(k, v))

            for g in groups:
                for m in g:
                    f.write("{}_{}_{}    ".format(m.formula, m.id, m.charge))
                f.write("\n")

    def compare_connectivity_across_graph_builder(
        self,
        union_plot_path,
        babel_plot_path,
        extender_plot_path,
        critic_plot_path,
        only_different=True,
        tex_file="tex_mol_connectivity_comparison.tex",
    ):
        """
        Write the connectivity (with plot) of molecules obtained from different methods
        into a tex file. This is for easier comparison of the connectivity.

        Args:
            union_plot_path (Path): directory where plots for the molecules with
                connectivity determined using the union of all methods are stored.
            babel_plot_path (Path): directory where plots for the molecules with
                connectivity determined using babel are stored. Similar for
                `extender_plot_path` and `critic_plot_path`.
            only_different (bool): If `True`, write out molecules only the connectivity
                obtained from the methods are different.
        tex_file (Path): path to the output .tex file.
        """

        union_plot_path = to_path(union_plot_path)
        babel_plot_path = to_path(babel_plot_path)
        extender_plot_path = to_path(extender_plot_path)
        critic_plot_path = to_path(critic_plot_path)

        # keep record of molecules of which the babel mol graph and critic mol graph are
        # different
        mols_differ_graph = []
        for m in self.molecules:

            # mol builder
            m1 = copy.deepcopy(m)

            # babel builder
            m.convert_to_babel_mol_graph(use_metal_edge_extender=False)
            m2 = copy.deepcopy(m)

            # babel builder with extender
            m.convert_to_babel_mol_graph(use_metal_edge_extender=True)
            m3 = copy.deepcopy(m)

            # critic
            m.convert_to_critic_mol_graph()
            m4 = copy.deepcopy(m)

            if not only_different or not m3.mol_graph.isomorphic_to(m4.mol_graph):
                mols_differ_graph.append([m1, m2, m3, m4])

        # write tex file
        tex_file = to_path(tex_file)
        with open(tex_file, "w") as f:
            f.write(TexWriter.head())
            f.write(
                "On each page, we plot 4 mols (top to bottom) from: the union of metal "
                "extender and critic, babel without extender, babel with extender and the "
                "critic builder.\n"
            )

            for i, mols in enumerate(mols_differ_graph):
                m = mols[0]

                # molecule info
                f.write(TexWriter.newpage())
                f.write("formula: " + m.formula + "\n\n")
                f.write("charge: " + str(m.charge) + "\n\n")
                f.write("spin multiplicity: " + str(m.spin_multiplicity) + "\n\n")
                f.write("free energy: " + str(m.free_energy) + "\n\n")
                f.write("id: " + m.id + "\n\n")

                # edge distances
                f.write("atom pair distances:\n\n")

                for a1, a2 in itertools.combinations(range(m.num_atoms), 2):
                    dist = np.linalg.norm(m.coords[a1] - m.coords[a2])
                    f.write("{} {}: {:.3f}\n\n".format(a1 + 1, a2 + 1, dist))

                # comparing edge differences between builder
                babel_bonds = set(
                    [(a1 + 1, a2 + 1) for (a1, a2), _ in mols[1].bonds.items()]
                )
                extender_bonds = set(
                    [(a1 + 1, a2 + 1) for (a1, a2), _ in mols[2].bonds.items()]
                )
                critic_bonds = set(
                    [(a1 + 1, a2 + 1) for (a1, a2), _ in mols[3].bonds.items()]
                )

                intersection = extender_bonds.intersection(critic_bonds)
                extender_not_in_critic = extender_bonds - intersection
                critic_not_in_extender = critic_bonds - intersection

                f.write("extender added to babel: ")
                for b in extender_bonds - babel_bonds:
                    f.write("{} ".format(b))
                f.write("\n\n")
                f.write("extender bond not in critic: ")
                for b in extender_not_in_critic:
                    f.write("{} ".format(b))
                f.write("\n\n")
                f.write("critic bond not in extender: ")
                for b in critic_not_in_extender:
                    f.write("{} ".format(b))
                f.write("\n\n")

                # add mol graph png
                for j, m in enumerate(mols):
                    if j == 0:
                        fname = union_plot_path.joinpath(f"{m.id}.png")
                    elif j == 1:
                        fname = babel_plot_path.joinpath(f"{m.id}.png")
                    elif j == 2:
                        fname = extender_plot_path.joinpath(f"{m.id}.png")
                    elif j == 3:
                        fname = critic_plot_path.joinpath(f"{m.id}.png")

                    f.write(TexWriter.single_figure(fname, figure_size=0.2))
                    f.write(TexWriter.verbatim("=" * 80))

            f.write(TexWriter.tail())

        filename = "~/Applications/db_access/mol_builder/molecules_union_builder.pkl"
        mols = [i[0] for i in mols_differ_graph]
        pickle_dump(mols, filename)
        filename = "~/Applications/db_access/mol_builder/molecules_babel_builder.pkl"
        mols = [i[1] for i in mols_differ_graph]
        pickle_dump(mols, filename)
        filename = "~/Applications/db_access/mol_builder/molecules_extender_builder.pkl"
        mols = [i[2] for i in mols_differ_graph]
        pickle_dump(mols, filename)
        filename = "~/Applications/db_access/mol_builder/molecules_critic_builder.pkl"
        mols = [i[3] for i in mols_differ_graph]
        pickle_dump(mols, filename)

        print(
            "### mol graph comparison. number of molecules {}, different mol graphs by "
            "babel extender builder and critic builder: {}".format(
                len(self.molecules), len(mols_differ_graph)
            )
        )

    def __len__(self):
        return len(self.molecules)


def check_connectivity(mol, allowed_connectivity=None, exclude=None):
    """
    Check the connectivity of each atom in a mol, without considering their bonding to
    metal element (e.g. Li), which forms coordinate bond with other atoms.
    """

    def get_neighbor_species(m):
        """
        Returns:
            A list of tuple (atom species, bonded atom species),
            where `bonded_atom_species` is a list.
            Each tuple represents an atom and its bonds.
        """
        res = [(s, []) for s in m.species]
        for (a1, a2), _ in m.bonds.items():
            s1 = m.species[a1]
            s2 = m.species[a2]
            res[a1][1].append(s2)
            res[a2][1].append(s1)
        return res

    if allowed_connectivity is None:
        allowed_connectivity = {
            "H": [1],
            "C": [1, 2, 3, 4],
            "O": [1, 2],
            "F": [1],
            "P": [1, 2, 3, 5, 6],  # 6 for LiPF6
            "N": [1, 2, 3, 4, 5],
            "S": [1, 2, 3, 4, 5, 6],
            "Cl": [1],
            # metal
            "Li": [1, 2, 3],
            "Mg": [1, 2, 3, 4, 5],
        }

    neigh_species = get_neighbor_species(mol)

    do_fail = False
    reason = []

    for a_s, n_s in neigh_species:

        if exclude is not None:
            num_bonds = len([s for s in n_s if s not in exclude])
        else:
            num_bonds = len(n_s)

        if num_bonds == 0:  # fine since we removed metal coordinate bonds
            continue

        if num_bonds not in allowed_connectivity[a_s]:
            reason.append("{} {}".format(a_s, num_bonds))
            do_fail = True

    return do_fail, reason


def check_bond_species(mol, not_allowed=[("Li", "H"), ("Li", "Li")]):
    """
    Check the species of atoms associated with a bond.
    Bonds provided in `not_allowed` fail the check.
    """

    def get_bond_species(m):
        """
        Returns:
            A list of the two species associated with each bonds in the molecule.
        """
        res = []
        for (a1, a2), _ in m.bonds.items():
            s1 = m.species[a1]
            s2 = m.species[a2]
            res.append(sorted([s1, s2]))
        return res

    not_allowed = [sorted(i) for i in not_allowed]

    bond_species = get_bond_species(mol)

    do_fail = False
    reason = []
    for b in bond_species:
        if b in not_allowed:
            reason.append(str(b))
            do_fail = True

    return do_fail, reason


def check_bond_length(mol, bond_length_limit=None):
    """
    Check the length of bonds. If larger than allowed length, it fails.

    """

    def get_bond_lengths(m):
        """
        Returns:
            A list of tuple (species, length), where species are the two species
            associated with a bond and length is the corresponding bond length.
        """
        res = []
        for (a1, a2), _ in m.bonds.items():
            s1 = m.species[a1]
            s2 = m.species[a2]
            c1 = np.asarray(m.coords[a1])
            c2 = np.asarray(m.coords[a2])
            length = np.linalg.norm(c1 - c2)
            res.append((tuple(sorted([s1, s2])), length))
        return res

    #
    # bond lengths references:
    # http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
    # page 29 https://slideplayer.com/slide/17256509/
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies
    #
    # unit: Angstrom
    #

    if bond_length_limit is None:
        li_len = 2.8
        mg_len = 2.8
        bond_length_limit = {
            # H
            # ("H", "H"): 0.74,
            ("H", "H"): None,
            ("H", "C"): 1.09,
            ("H", "O"): 0.96,
            # ("H", "F"): 0.92,
            ("H", "F"): None,
            ("H", "P"): 1.44,
            ("H", "N"): 1.01,
            ("H", "S"): 1.34,
            # ("H", "Cl"): 1.27,
            ("H", "Cl"): None,
            ("H", "Li"): li_len,
            ("H", "Mg"): mg_len,
            # C
            ("C", "C"): 1.54,
            ("C", "O"): 1.43,
            ("C", "F"): 1.35,
            ("C", "P"): 1.84,
            ("C", "N"): 1.47,
            ("C", "S"): 1.81,
            ("C", "Cl"): 1.77,
            ("C", "Li"): li_len,
            ("C", "Mg"): mg_len,
            # O
            ("O", "O"): 1.48,
            ("O", "F"): 1.42,
            ("O", "P"): 1.63,
            ("O", "N"): 1.44,
            ("O", "S"): 1.51,
            ("O", "Cl"): 1.64,
            ("O", "Li"): li_len,
            ("O", "Mg"): mg_len,
            # F
            # ("F", "F"): 1.42,
            ("F", "F"): None,
            ("F", "P"): 1.54,
            ("F", "N"): 1.39,
            ("F", "S"): 1.58,
            # ("F", "Cl"): 1.66,
            ("F", "Cl"): None,
            ("F", "Li"): li_len,
            ("F", "Mg"): mg_len,
            # P
            ("P", "P"): 2.21,
            ("P", "N"): 1.77,
            ("P", "S"): 2.1,
            ("P", "Cl"): 204,
            ("P", "Li"): li_len,
            ("P", "Mg"): mg_len,
            # N
            ("N", "N"): 1.46,
            ("N", "S"): 1.68,
            ("N", "Cl"): 1.91,
            ("N", "Li"): li_len,
            ("N", "Mg"): mg_len,
            # S
            ("S", "S"): 2.04,
            ("S", "Cl"): 201,
            ("S", "Li"): li_len,
            ("S", "Mg"): mg_len,
            # Cl
            # ("Cl", "Cl"): 1.99,
            ("Cl", "Cl"): None,
            ("Cl", "Li"): li_len,
            ("Cl", "Mg"): mg_len,
            # Li
            ("Li", "Li"): li_len,
            # Mg
            ("Mg", "Mg"): mg_len,
        }

        # multiply by 1.2 to relax the rule a bit
        tmp = dict()
        for k, v in bond_length_limit.items():
            if v is not None:
                v *= 1.2
            tmp[tuple(sorted(k))] = v
        bond_length_limit = tmp

    do_fail = False
    reason = []

    bond_species = get_bond_lengths(mol)
    for b, length in bond_species:
        limit = bond_length_limit[b]
        if limit is not None and length > limit:
            reason.append("{}  {} ({})".format(b, length, limit))
            do_fail = True

    return do_fail, reason
