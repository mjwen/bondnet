import itertools
import os
import glob
import numpy as np
from collections import defaultdict
import matplotlib as mpl
from matplotlib import pyplot as plt
from gnn.data.reaction import ReactionExtractor
from pprint import pprint
from gnn.data.utils import TexWriter
from gnn.utils import pickle_load, expand_path, create_directory
from gnn.data.feature_analyzer import read_sdf, read_label


def eg_buckets():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge", "spin_multiplicity"])
    pprint(extractor.buckets)
    extractor.bucket_molecules(keys=["formula"])
    pprint(extractor.buckets)


def eg_extract_A_to_B():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_A_to_B_style_reaction()
    extractor.to_file(filename="~/Applications/db_access/mol_builder/reaction_A2B.pkl")


def eg_extract_A_to_B_C():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_A_to_B_C_style_reaction()
    extractor.to_file(filename="~/Applications/db_access/mol_builder/reactions_A2BC.pkl")


def eg_extract_one_bond_break():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets:", len(extractor.buckets))

    extractor.extract_one_bond_break()

    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    extractor.to_file(filename)


def subselect_reactions():
    filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    extractor = ReactionExtractor.from_file(filename)
    reactions = []

    # for rxn in extractor.reactions:
    #    if rxn.reactants[0].formula == "C5H8Li1O3":
    #        reactions.append(rxn)

    ##############
    # filter reactant charge = 0
    ##############
    extractor.filter_reactions_by_reactant_charge(charge=0)

    # ##############
    # # filter C-C bond
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"))

    extractor.reactions = reactions
    extractor.molecules = extractor._get_molecules_from_reactions(reactions)

    # filename = "~/Applications/db_access/mol_builder/reactions_C5H8Li1O3.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_charge0.pkl"
    extractor.to_file(filename)


def reactants_bond_energies_to_file():
    filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_C5H8Li1O3.pkl"
    extractor = ReactionExtractor.from_file(filename)

    # ##############
    # # filter reactant charge = 0
    # ##############
    # extractor.filter_reactions_by_reactant_charge(charge=0)
    #
    # ##############
    # # filter C-C bond and order
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"))

    filename = "~/Applications/db_access/mol_builder/bond_energies.yaml"
    # filename = "~/Applications/db_access/mol_builder/bond_energies_n200.yaml"
    extractor.write_bond_energies(filename)


def reactant_broken_bond_fraction():
    """
    Get the fraction of bonds that broken with respect to all the bonds in a reactant.

    Note, this requires that when extracting the reactions, all the reactions
    related to a reactant (ignore symmetry) should be extracted.

    """

    filename = "~/Applications/db_access/mol_builder/reactions_having_all.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"

    extractor = ReactionExtractor.from_file(filename)
    groups = extractor.group_by_reactant_bond_and_charge()

    num_bonds = dict()
    frac = dict()
    for reactant, reactions in groups.items():
        val = 0
        for bond, rxns in reactions.items():
            if rxns:
                val += 1
        frac[reactant] = val / len(reactions)
        num_bonds[reactant] = len(reactions)

    frac_list = [v for k, v in frac.items()]
    num_bonds_list = [v for k, v in num_bonds.items()]

    print("### number of bonds in dataset (mean):", np.mean(num_bonds_list))
    print("### number of bonds in dataset (median):", np.median(num_bonds_list))
    print("### broken bond ratio in dataset (mean):", np.mean(frac_list))
    print("### broken bond ratio in dataset (median):", np.median(frac_list))


def bond_energy_difference_in_molecule_nth_lowest():

    """
    Get the nth lowest bond energy difference in molecules.

    """

    def hist_analysis(data, xmin=0, xmax=1, num_bins=10, frac_all_data=True):
        hist, bin_edge = np.histogram(data, bins=num_bins, range=(xmin, xmax))
        outside = 0
        for i in data:
            if i > xmax:
                outside += 1

        if frac_all_data:
            total = len(data)
        else:
            total = sum(hist)

        print("Bond energy difference histogram")
        print("energy        counts      %")
        for i, n in enumerate(hist):
            print(
                "{:.2f}--{:.2f}:    {}    {:.1f}%".format(
                    bin_edge[i], bin_edge[i + 1], n, n / total * 100
                )
            )
        print("> {}:          {}    {:.1f}%".format(xmax, outside, outside / total * 100))

    filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"

    ########################
    # nth lowest
    ########################
    all_nth = [1, 2, 3, 4]

    extractor = ReactionExtractor.from_file(filename)
    groups = extractor.group_by_reactant_bond_keep_lowest_energy_across_products_charge()

    for nth in all_nth:
        bond_energy_diff = dict()
        for rsr in groups:
            energies = [rxn.get_reaction_free_energy() for rxn in rsr.reactions]
            e_diff = [abs(i - j) for i, j in itertools.combinations(energies, 2)]

            # get nth lowest
            bond_energy_diff[rsr.reactant] = sorted(e_diff)[(nth - 1) : nth]
        diff = [v for k, v in bond_energy_diff.items()]
        diff = np.concatenate(diff)

        hist_analysis(diff, xmin=0, xmax=1, num_bins=10, frac_all_data=True)
        hist_analysis(diff, xmin=0, xmax=0.1, num_bins=10, frac_all_data=False)


def plot_reaction_energy_difference_arcoss_reactant_charge(
    filename="~/Applications/db_access/mol_builder/reactions.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_C5H8Li1O3.pkl",
):
    """
    Plot a histogram showing the energy difference of the same reaction between
    different charges.

    e.g.
    A(+1) -> B(+1) + C(0)    G1
    A(0) -> B(0) + C(0)      G2
    Delta G = G2-G2

    we plot delta G
    """

    def plot_hist(data, filename, s1, s2, c1, c2):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(data, 20)

        ax.set_xlabel("E diff. species:{} {}; charge {} {}".format(s1, s2, c1, c2))
        ax.set_ylabel("counts")

        fig.savefig(filename, bbox_inches="tight")

    def extract_one(extractor, s1=None, s2=None):
        results = extractor.group_by_reactant_charge()
        for charge in [(-1, 0), (-1, 1), (0, 1)]:
            reactions = results[charge]
            energies = []
            for rxn1, rxn2 in reactions:
                e_diff = rxn2.get_reaction_free_energy() - rxn1.get_reaction_free_energy()
                energies.append(e_diff)

            outname = "~/Applications/db_access/mol_builder/reactant_e_diff/"
            outname += "reactant_e_diff_{}{}_{}_{}.pdf".format(s1, s2, *charge)
            outname = expand_path(outname)
            create_directory(outname)
            plot_hist(energies, outname, s1, s2, *charge)

    # all species together
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.reactions
    extract_one(extractor)

    # species pairs
    species = ["H", "Li", "C", "O", "F", "P"]
    for s1, s2 in itertools.combinations_with_replacement(species, 2):
        extractor = ReactionExtractor(molecules=None, reactions=all_reactions)
        extractor.filter_reactions_by_bond_type_and_order(bond_type=[s1, s2])
        extract_one(extractor, s1, s2)


def plot_bond_type_heat_map(
    filename="~/Applications/db_access/mol_builder/reactions.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
):
    """
    Generate a heatmap to show the statistics of the bond type (species of the two
    atoms).
    """

    def plot_heat_map(matrix, labels, filename="heat_map.pdf", cmap=mpl.cm.viridis):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap=cmap, vmin=np.min(matrix), vmax=np.max(matrix))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(labels)), minor=False)
        ax.set_yticks(np.arange(len(labels)), minor=False)

        # label them with the respective list entries
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(labels, minor=False)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)

        # colorbar
        plt.colorbar(im)
        fig.savefig(filename, bbox_inches="tight")

    def plot_one(reactions, charge):

        num_bonds = defaultdict(float)
        for rxn in reactions:
            bt = tuple(sorted(rxn.get_broken_bond_attr()["species"]))
            num_bonds[bt] += 1
            # if bt == ("F", "F"):
            #     print("@@@", rxn.as_dict())

        species = ["H", "Li", "C", "O", "F", "P"]
        data = np.zeros((len(species), len(species))).astype(np.int32)
        for s1, s2 in itertools.combinations_with_replacement(species, 2):
            idx1 = species.index(s1)
            idx2 = species.index(s2)
            key = tuple(sorted([s1, s2]))
            data[idx1, idx2] = num_bonds[key]
            data[idx2, idx1] = num_bonds[key]

        # plot
        filename = "~/Applications/db_access/mol_builder/bond_type_count_{}.pdf".format(
            charge
        )
        filename = expand_path(filename)
        plot_heat_map(data, species, filename)

        # table
        table = TexWriter.beautifultable(data, species, species, " ")
        print("number of reactions", len(reactions))
        print("table for charges: {}".format(charge))
        print(table)

    # prepare data
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.get_reactions_with_lowest_energy()

    # all charges
    plot_one(all_reactions, "all")

    # for specific charge
    for charge in [-1, 0, 1]:
        reactions = []
        for rxn in all_reactions:
            if rxn.reactants[0].charge == charge:
                reactions.append(rxn)
        plot_one(reactions, charge)


def plot_bond_energy_hist(
    filename="~/Applications/db_access/mol_builder/reactions.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
):
    """
    Plot histogram of bond energy of a specific type of bond (two atoms forming the
    bond) and at a specific charge.
    """

    def plot_hist(data, filename, s1, s2, ch, xmin, xmax):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(data, 20, range=(xmin, xmax))

        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Bond energy. species:{} {}; charge {}".format(s1, s2, ch))
        ax.set_ylabel("counts")

        fig.savefig(filename, bbox_inches="tight")

    def plot_one(all_reactions, s1, s2):
        charge = [None, -1, 0, 1]

        for ch in charge:
            if ch is None:
                reactions = all_reactions
            else:
                reactions = []
                for rxn in all_reactions:
                    if rxn.reactants[0].charge == ch:
                        reactions.append(rxn)

            energies = []
            for rxn in reactions:
                energies.append(rxn.get_reaction_free_energy())

            if ch is None:
                if len(energies) == 0:
                    xmin = 0.0
                    xmax = 1.0
                else:
                    xmin = np.min(energies) - 0.5
                    xmax = np.max(energies) + 0.5

            outname = "~/Applications/db_access/mol_builder/bond_energy_hist/"
            outname += "bond_energy_histogram_species_{}{}_charge_{}.pdf".format(
                s1, s2, ch
            )
            outname = expand_path(outname)
            create_directory(outname)
            plot_hist(energies, outname, s1, s2, ch, xmin, xmax)

    # prepare data
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.get_reactions_with_lowest_energy()
    print(
        "@@@ total number of reactions: {}, lowest energy reactions: {}".format(
            len(extractor.reactions), len(all_reactions)
        )
    )

    # all species
    plot_one(all_reactions, None, None)

    # species pairs
    species = ["H", "Li", "C", "O", "F", "P"]
    for s1, s2 in itertools.combinations_with_replacement(species, 2):
        reactions = []
        for rxn in all_reactions:
            bt = tuple(sorted(rxn.get_broken_bond_attr()["species"]))
            if bt == tuple(sorted((s1, s2))):
                reactions.append(rxn)
        plot_one(reactions, s1, s2)


def plot_all_bond_length_hist(
    filename="~/Applications/db_access/mol_builder/reactions.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
):
    """
    This includes all bonds in molecules.
    """

    def plot_hist(data, filename):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(data, 20)

        ax.set_xlabel("Bond length")
        ax.set_ylabel("counts")

        fig.savefig(filename, bbox_inches="tight")

    def get_length_of_bond(rxn):
        reactant = rxn.reactants[0]
        coords = reactant.coords
        dist = [np.linalg.norm(coords[u] - coords[v]) for u, v, _ in reactant.bonds]
        return dist

    # prepare data
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.get_reactions_with_lowest_energy()
    data = [get_length_of_bond(rxn) for rxn in all_reactions]
    data = np.concatenate(data)

    print("\n\n@@@ all bond length min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/db_access/mol_builder/bond_length_all.pdf"
    filename = expand_path(filename)
    plot_hist(data, filename)


def plot_broken_bond_length_hist(
    filename="~/Applications/db_access/mol_builder/reactions.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
):
    """
    This includes only the broken bonds, not all the bonds.
    """

    def plot_hist(data, filename):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(data, 20)

        ax.set_xlabel("Bond length")
        ax.set_ylabel("counts")

        fig.savefig(filename, bbox_inches="tight")

    def get_length_of_broken_bond(rxn):
        coords = rxn.reactants[0].coords
        u, v = rxn.get_broken_bond()
        dist = np.linalg.norm(coords[u] - coords[v])
        return dist

    # prepare data
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.get_reactions_with_lowest_energy()
    data = [get_length_of_broken_bond(rxn) for rxn in all_reactions]

    print("\n\n@@@ broken bond length min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/db_access/mol_builder/bond_length_broken.pdf"
    filename = expand_path(filename)
    plot_hist(data, filename)


def create_struct_label_dataset_mol_based():
    filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_charge0.pkl"

    extractor = ReactionExtractor.from_file(filename)
    extractor.create_struct_label_dataset_mol_based(
        struct_file="~/Applications/db_access/mol_builder/struct_mol_based.sdf",
        label_file="~/Applications/db_access/mol_builder/label_mol_based.txt",
        feature_file="~/Applications/db_access/mol_builder/feature_mol_based.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_n200.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_n200.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_n200.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_charge0.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_charge0.txt",
    )


def create_struct_label_dataset_bond_based_lowest_energy():
    filename = "~/Applications/db_access/mol_builder/reactions_having_all.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    extractor = ReactionExtractor.from_file(filename)

    ##############
    # filter by reactant attributes
    ##############
    # extractor.filter_reactions_by_reactant_attribute(
    #     key="id", values=["5e2a05838eab11f1fa104e29", "5e2a05d28eab11f1fa107899"]
    # )
    # extractor.filter_reactions_by_reactant_attribute(
    #     key="formula", values=["C3H4O3", "C3H3O3"]
    # )
    # extractor.filter_reactions_by_reactant_attribute(key="charge", values=[1])

    # ##############
    # # filter C-C bond
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"))

    extractor.create_struct_label_dataset_bond_based(
        struct_file="~/Applications/db_access/mol_builder/struct.sdf",
        label_file="~/Applications/db_access/mol_builder/label.txt",
        feature_file="~/Applications/db_access/mol_builder/feature.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_n200.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_n200.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_n200.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_charge1.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_charge1.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_charge1.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_charge0_CC.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_charge0_CC.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_charge0_CC.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_observe.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_observe.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_observe.yaml",
    )


def create_struct_label_dataset_bond_based_0_charge():
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    extractor = ReactionExtractor.from_file(filename)

    # ##############
    # # filter by reactant attributes
    # ##############
    # extractor.filter_reactions_by_reactant_attribute(
    #     key="id", values=["5e2a05838eab11f1fa104e29", "5e2a05d28eab11f1fa107899"]
    # )
    # extractor.filter_reactions_by_reactant_attribute(
    #     key="formula", values=["C3H4O3", "C3H3O3"]
    # )
    # extractor.filter_reactions_by_reactant_attribute(key="charge", values=[1])

    # ##############
    # # filter C-C bond
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"))

    extractor.create_struct_label_dataset_bond_based(
        struct_file="~/Applications/db_access/mol_builder/struct_0_charge_all.sdf",
        label_file="~/Applications/db_access/mol_builder/label_0_charge_all.txt",
        feature_file="~/Applications/db_access/mol_builder/feature_0_charge_all.yaml",
        lowest_across_product_charge=False,
    )


def write_reaction_sdf_mol_png():
    """
    Write reactions to file, including its sdf file and png graphs.
    """

    label_file = "~/Applications/db_access/mol_builder/label_observe.txt"
    struct_file = "~/Applications/db_access/mol_builder/struct_observe.sdf"

    png_dir = "~/Applications/db_access/mol_builder/mol_png"
    tex_file = "~/Applications/db_access/mol_builder/reactions_sdf_energy_mol_png.tex"

    labels = read_label(label_file)
    structs = read_sdf(struct_file)
    all_pngs = glob.glob(os.path.join(expand_path(png_dir), "*.png"))

    tex_file = expand_path(tex_file)
    with open(tex_file, "w") as f:

        f.write(TexWriter.head())

        for rxn in labels:
            reactant = rxn["reactants"]
            products = rxn["products"]
            raw = rxn["raw"]

            f.write(TexWriter.newpage())

            # sdf info
            f.write(TexWriter.verbatim(structs[reactant]))

            # label info
            f.write(TexWriter.verbatim(TexWriter.resize_string(raw)))

            # figure
            filename = None
            for name in all_pngs:
                if reactant in name:
                    filename = name
                    break
            if filename is None:
                raise Exception(
                    "cannot find png file for {} in {}".format(reactant, png_dir)
                )
            f.write(TexWriter.single_figure(filename))

            f.write(r"\begin{equation*}\downarrow\end{equation*}")

            for i, p in enumerate(products):
                if i > 0:
                    f.write(r"\begin{equation*}+\end{equation*}")
                filename = None
                for name in all_pngs:
                    if p in name:
                        filename = name
                        break
                if filename is None:
                    raise Exception(
                        "cannot find png file for {} in {}".format(p, png_dir)
                    )
                f.write(TexWriter.single_figure(filename))

        # tail
        f.write(TexWriter.tail())


if __name__ == "__main__":
    # eg_buckets()
    # eg_extract_A_to_B()
    # eg_extract_A_to_B_C()
    # eg_extract_one_bond_break()
    # subselect_reactions()

    # plot_reaction_energy_difference_arcoss_reactant_charge()
    # plot_bond_type_heat_map()
    # plot_bond_energy_hist()
    # plot_broken_bond_length_hist()
    # plot_all_bond_length_hist()

    # reactant_broken_bond_fraction()
    # bond_energy_difference_in_molecule_nth_lowest()

    # reactants_bond_energies_to_file()
    # create_struct_label_dataset_mol_based()
    # create_struct_label_dataset_bond_based_lowest_energy()
    create_struct_label_dataset_bond_based_0_charge()

    # write_reaction_sdf_mol_png()
