import itertools
import os
import glob
import numpy as np
from collections import defaultdict
from pprint import pprint
import matplotlib as mpl
from matplotlib import pyplot as plt
from gnn.database.utils import TexWriter
from gnn.database.reaction import ReactionExtractor, ReactionsMultiplePerBond
from gnn.utils import pickle_load, expand_path, create_directory
from gnn.data.feature_analyzer import read_sdf, read_label


def eg_buckets():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    buckets = extractor.bucket_molecules(keys=["formula", "charge", "spin_multiplicity"])
    pprint(buckets)
    buckets = extractor.bucket_molecules(keys=["formula"])
    pprint(buckets)


def eg_extract_A_to_B():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.extract_A_to_B_style_reaction(find_one=False)

    filename = "~/Applications/db_access/mol_builder/reaction_n200.pkl"
    extractor.to_file(filename)


def eg_extract_A_to_B_C():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.extract_A_to_B_C_style_reaction(fine_one=False)

    filename = "~/Applications/db_access/mol_builder/reactions_A2BC.pkl"
    extractor.to_file(filename)


def eg_extract_one_bond_break():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.extract_one_bond_break(find_one=False)

    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    extractor.to_file(filename)


def subselect_reactions():
    filename = "~/Applications/db_access/mol_builder/reactions_qc_ws.pkl"
    extractor = ReactionExtractor.from_file(filename)

    ##############
    # filter reactant charge = 0
    ##############
    # extractor.filter_reactions_by_reactant_attribute(key="charge", values=[0])

    ##############
    # filter charge 0 reactant and products
    ##############
    reactions = []
    for rxn in extractor.reactions:
        zero_charge = True
        for m in rxn.reactants + rxn.products:
            if m.charge != 0:
                zero_charge = False
                break
        if zero_charge:
            reactions.append(rxn)
    extractor.reactions = reactions

    # ##############
    # # filter C-C bond
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"))

    reactions = extractor.reactions
    extractor.molecules = extractor._get_molecules_from_reactions(reactions)
    # filename = "~/Applications/db_access/mol_builder/reactions_C5H8Li1O3.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_qc_ws_charge0.pkl"
    extractor.to_file(filename)


def bond_energies_to_file():
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
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

    # filename = "~/Applications/db_access/mol_builder/bond_energies.yaml"
    filename = "~/Applications/db_access/mol_builder/bond_energies_n200.yaml"
    extractor.write_bond_energies(filename)


def reactant_broken_bond_fraction():
    """
    Get the fraction of bonds that broken with respect to all the bonds in a reactant.

    Note, this requires that when extracting the reactions, all the reactions
    related to a reactant (ignore symmetry) should be extracted.

    """

    filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    # filename = "~/Applications/db_access_newest/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access_newest/mol_builder/reactions_qc.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_qc_ws.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_qc_ws_charge0.pkl"

    extractor = ReactionExtractor.from_file(filename)
    groups = extractor.group_by_reactant()

    num_bonds = []
    frac = []
    for reactant, rxns in groups.items():
        rmb = ReactionsMultiplePerBond(reactant, rxns)
        rsbs = rmb.group_by_bond()
        tot = len(rsbs)
        bond_has_rxn = [True if len(x.reactions) > 0 else False for x in rsbs]
        num_bonds.append(tot)
        frac.append(sum(bond_has_rxn) / tot)

    print("### number of bonds in dataset (mean):", np.mean(num_bonds))
    print("### number of bonds in dataset (median):", np.median(num_bonds))
    print("### broken bond ratio in dataset (mean):", np.mean(frac))
    print("### broken bond ratio in dataset (median):", np.median(frac))


def bond_label_fraction(top_n=2):
    """
    Get the fraction of each type of label 0 (low energy), 1 (high energy),
    and 2 (unknown) in the dataset.

    Note, this requires that when extracting the reactions, all the reactions
    related to a reactant (ignore symmetry) should be extracted.

    Note, here we analyze on the 0,0,0 type reactions.

    """
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_qc.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_qc_ws.pkl"
    filename = "~/Applications/db_access/mol_builder/reactions_qc_ws_charge0.pkl"

    extractor = ReactionExtractor.from_file(filename)
    groups = extractor.group_by_reactant_charge_0()

    num_bonds = []
    frac = defaultdict(list)
    for rsr in groups:
        label0 = 0
        label1 = 0
        label2 = 0
        all_None = True
        for bond, data in rsr.order_reactions().items():
            if data["order"] is None:
                label2 += 1
            else:
                all_None = False
                if data["order"] < top_n:
                    label0 += 1
                else:
                    label1 += 1
        if all_None:
            print(
                "reactant {} {} has not broken bond reaction; should never happen".format(
                    rsr.reactant.id, rsr.reactant.formula
                )
            )
            continue

        n = len(rsr.order_reactions())
        num_bonds.append(n)
        frac["label0"].append(label0 / n)
        frac["label1"].append(label1 / n)
        frac["label2"].append(label2 / n)

    print("### number of bonds in dataset (mean):", np.mean(num_bonds))
    print("### number of bonds in dataset (median):", np.median(num_bonds))
    print("### label0 bond ratio in dataset (mean):", np.mean(frac["label0"]))
    print("### label0 bond ratio in dataset (mean):", np.median(frac["label0"]))
    print("### label1 bond ratio in dataset (mean):", np.mean(frac["label1"]))
    print("### label1 bond ratio in dataset (mean):", np.median(frac["label1"]))
    print("### label2 bond ratio in dataset (mean):", np.mean(frac["label2"]))
    print("### label2 bond ratio in dataset (mean):", np.median(frac["label2"]))


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
    groups = extractor.group_by_reactant_lowest_energy()

    for nth in all_nth:
        bond_energy_diff = dict()
        for rsr in groups:
            energies = [rxn.get_free_energy() for rxn in rsr.reactions]
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
        results = extractor.group_by_reactant_charge_pair()
        for charge in [(-1, 0), (-1, 1), (0, 1)]:
            reactions = results[charge]
            energies = []
            for rxn1, rxn2 in reactions:
                e_diff = rxn2.get_free_energy() - rxn1.get_free_energy()
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
                energies.append(rxn.get_free_energy())

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
    # filename="~/Applications/db_access/mol_builder/reactions.pkl",
    filename="~/Applications/db_access/mol_builder/reactions_qc.pkl",
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
        ax.set_yscale("log")

        fig.savefig(filename, bbox_inches="tight")

    def get_length_of_broken_bond(rxn):
        coords = rxn.reactants[0].coords
        u, v = rxn.get_broken_bond()
        dist = np.linalg.norm(coords[u] - coords[v])

        if dist > 3.0:
            print(
                "Some bonds are suspicious. id:",
                rxn.reactants[0].id,
                "bonds:",
                u,
                v,
                "energy",
                rxn.reactants[0].free_energy,
            )

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


def create_struct_label_dataset_bond_based_regression(
    # filename="~/Applications/db_access/mol_builder/reactions_qc.pkl",
    filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
    lowest_energy=False,
):

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

    extractor.create_struct_label_dataset_bond_based_regressssion(
        # struct_file="~/Applications/db_access/mol_builder/struct.sdf",
        # label_file="~/Applications/db_access/mol_builder/label.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature.yaml",
        struct_file="~/Applications/db_access/mol_builder/struct_n200.sdf",
        label_file="~/Applications/db_access/mol_builder/label_n200.txt",
        feature_file="~/Applications/db_access/mol_builder/feature_n200.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_qc.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_qc.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_qc.yaml",
        lowest_across_product_charge=lowest_energy,
    )


def create_struct_label_dataset_bond_based_classification(
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl",
    filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_qc.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_qc_ws.pkl",
    lowest_energy=False,
    top_n=2,
):

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

    extractor.create_struct_label_dataset_bond_based_classification(
        # struct_file="~/Applications/db_access/mol_builder/struct.sdf",
        # label_file="~/Applications/db_access/mol_builder/label.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature.yaml",
        struct_file="~/Applications/db_access/mol_builder/struct_clfn_n200.sdf",
        label_file="~/Applications/db_access/mol_builder/label_clfn_n200.txt",
        feature_file="~/Applications/db_access/mol_builder/feature_clfn_n200.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_clfn_qc_ws.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_clfn_qc_ws.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_clfn_qc_ws.yaml",
        lowest_across_product_charge=lowest_energy,
        top_n=top_n,
    )


def create_struct_label_dataset_reaction_based_classification(
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_qc.pkl",
    filename="~/Applications/db_access/mol_builder/reactions_qc_ws.pkl",
    lowest_energy=False,
    top_n=2,
):

    extractor = ReactionExtractor.from_file(filename)

    extractor.create_struct_label_dataset_reaction_based(
        # struct_file="~/Applications/db_access/mol_builder/struct_rxn_clfn_n200.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_rxn_clfn_n200.yaml",
        # feature_file="~/Applications/db_access/mol_builder/feature_rxn_clfn_n200.yaml",
        struct_file="~/Applications/db_access/mol_builder/struct_rxn_clfn_qc_ws.sdf",
        label_file="~/Applications/db_access/mol_builder/label_rxn_clfn_qc_ws.yaml",
        feature_file="~/Applications/db_access/mol_builder/feature_rxn_clfn_qc_ws.yaml",
        top_n=top_n,
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
    eg_extract_one_bond_break()
    # subselect_reactions()

    # plot_reaction_energy_difference_arcoss_reactant_charge()
    # plot_bond_type_heat_map()
    # plot_bond_energy_hist()
    # plot_broken_bond_length_hist()
    # plot_all_bond_length_hist()

    # reactant_broken_bond_fraction()
    # bond_label_fraction()
    # bond_energy_difference_in_molecule_nth_lowest()

    # bond_energies_to_file()
    # create_struct_label_dataset_mol_based()
    # create_struct_label_dataset_bond_based_regression()
    # create_struct_label_dataset_bond_based_classification()
    # create_struct_label_dataset_reaction_based_classification()

    # write_reaction_sdf_mol_png()
