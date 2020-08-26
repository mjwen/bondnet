import itertools
import numpy as np
from collections import defaultdict
from pprint import pprint
from matplotlib import pyplot as plt
from bondnet.core.reaction import ReactionsMultiplePerBond, ReactionExtractorFromMolSet
from bondnet.core.reaction_collection import ReactionCollection
from bondnet.utils import pickle_load, to_path, create_directory


def eg_buckets(molecule_file):
    molecules = pickle_load(molecule_file)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractorFromMolSet(molecules)
    buckets = extractor.bucket_molecules(keys=["formula", "charge", "spin_multiplicity"])
    pprint(buckets)
    buckets = extractor.bucket_molecules(keys=["formula"])
    pprint(buckets)


def eg_extract_A_to_B(molecule_file, reaction_file="reactions.pkl"):
    molecules = pickle_load(molecule_file)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractorFromMolSet(molecules)
    extractor.extract_A_to_B_style_reaction(find_one=False)

    extractor.to_file(reaction_file)


def eg_extract_A_to_B_C(molecule_file, reaction_file="reactions.pkl"):

    molecules = pickle_load(molecule_file)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractorFromMolSet(molecules)
    extractor.extract_A_to_B_C_style_reaction(find_one=False)

    extractor.to_file(reaction_file)


def eg_extract_one_bond_break(molecule_file, reaction_file="reactions.pkl"):
    molecules = pickle_load(molecule_file)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractorFromMolSet(molecules)
    extractor.extract_one_bond_break(find_one=False)

    extractor.to_file(reaction_file)


def get_reactions_with_lowest_energy(extractor):
    """
    Get the reactions by removing higher energy ones. Higher energy is compared
    across product charge.

    Returns:
        list: a sequence of :class:`Reaction`.
    """
    groups = extractor.group_by_reactant_lowest_energy()
    reactions = []
    for rsr in groups:
        reactions.extend(rsr.reactions)
    return reactions


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

    extractor = ReactionCollection.from_file(filename)
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

    extractor = ReactionCollection.from_file(filename)
    groups = extractor.group_by_reactant_charge_0()

    num_bonds = []
    frac = defaultdict(list)
    for ropb in groups:
        counts = np.asarry([0, 0, 0])
        all_none = True
        for i, rxn in enumerate(ropb.order_reactions()):
            energy = rxn.get_free_energy()
            if energy is None:
                counts[2] += 1
            else:
                all_none = False
                if i < top_n:
                    counts[1] += 1
                else:
                    counts[0] += 1
        if all_none:
            print(
                "reactant {} {} has not broken bond reaction; should never happen".format(
                    ropb.reactant.id, ropb.reactant.formula
                )
            )
            continue

        n = len(ropb.order_reactions())
        num_bonds.append(n)
        frac = counts / n

    print("### number of bonds in dataset (mean):", np.mean(num_bonds))
    print("### number of bonds in dataset (median):", np.median(num_bonds))
    for i, fr in enumerate(frac):
        print(f"### label{i} bond ratio in dataset (mean): {np.mean(fr)}")
        print(f"### label{i} bond ratio in dataset (mean): {np.median(fr)}")


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

    extractor = ReactionCollection.from_file(filename)
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
            create_directory(outname)

            outname = to_path(outname)
            plot_hist(energies, outname, s1, s2, *charge)

    # all species together
    extractor = ReactionCollection.from_file(filename)
    all_reactions = extractor.reactions
    extract_one(extractor)

    # species pairs
    species = ["H", "Li", "C", "O", "F", "P"]
    for s1, s2 in itertools.combinations_with_replacement(species, 2):
        extractor = ReactionCollection(molecules=None, reactions=all_reactions)
        extractor.filter_reactions_by_bond_type_and_order(bond_type=[s1, s2])
        extract_one(extractor, s1, s2)


def create_struct_label_dataset_mol_based():
    filename = "~/Applications/db_access/mol_builder/reactions.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_n200.pkl"
    # filename = "~/Applications/db_access/mol_builder/reactions_charge0.pkl"

    extractor = ReactionCollection.from_file(filename)
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
):

    extractor = ReactionCollection.from_file(filename)

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

    extractor.create_struct_label_dataset_bond_based_regression(
        struct_file="~/Applications/db_access/mol_builder/struct_n200.sdf",
        label_file="~/Applications/db_access/mol_builder/label_n200.yaml",
        feature_file="~/Applications/db_access/mol_builder/feature_n200.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_qc.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_qc.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_qc.yaml",
        group_mode="charge_0",
        one_per_iso_bond_group=True,
    )


def create_struct_label_dataset_bond_based_classification(
    # filename = "~/Applications/db_access/mol_builder/reactions.pkl",
    filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_qc.pkl",
    # filename="~/Applications/db_access/mol_builder/reactions_qc_wib.pkl",
):

    extractor = ReactionCollection.from_file(filename)

    extractor.create_struct_label_dataset_bond_based_classification(
        struct_file="~/Applications/db_access/mol_builder/struct_bond_clfn_n200.sdf",
        label_file="~/Applications/db_access/mol_builder/label_bond_clfn_n200.txt",
        feature_file="~/Applications/db_access/mol_builder/feature_bond_clfn_n200.yaml",
        # struct_file="~/Applications/db_access/mol_builder/struct_bond_clfn_qc.sdf",
        # label_file="~/Applications/db_access/mol_builder/label_bond_clfn_qc.txt",
        # feature_file="~/Applications/db_access/mol_builder/feature_bond_clfn_qc.yaml",
        group_mode="charge_0",
        top_n=2,
        complement_reactions=False,
        one_per_iso_bond_group=True,
    )


def create_struct_label_dataset_reaction_based_regression(
    filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
):

    extractor = ReactionCollection.from_file(filename)

    extractor.create_struct_label_dataset_reaction_based_regression(
        struct_file="~/Applications/db_access/mol_builder/struct_rxn_rgrn_n200.sdf",
        label_file="~/Applications/db_access/mol_builder/label_rxn_rgrn_n200.yaml",
        feature_file="~/Applications/db_access/mol_builder/feature_rxn_rgrn_n200.yaml",
        group_mode="all",
        one_per_iso_bond_group=True,
    )


def create_struct_label_dataset_reaction_based_classification(
    filename="~/Applications/db_access/mol_builder/reactions_n200.pkl",
):

    extractor = ReactionCollection.from_file(filename)

    extractor.create_struct_label_dataset_reaction_based_classification(
        struct_file="~/Applications/db_access/mol_builder/struct_rxn_clfn_n200.sdf",
        label_file="~/Applications/db_access/mol_builder/label_rxn_clfn_n200.yaml",
        feature_file="~/Applications/db_access/mol_builder/feature_rxn_clfn_n200.yaml",
        group_mode="all",
        top_n=2,
        complement_reactions=True,
        one_per_iso_bond_group=True,
    )


def create_struct_label_dataset_reaction_network_based_regression(
    reaction_file,
    struct_file="struct_rxn_ntwk_rgrn.sdf",
    label_file="label_rxn_ntwk_rgrn.yaml",
    feature_file="feature_rxn_ntwk_rgrn.yaml",
):
    extractor = ReactionCollection.from_file(reaction_file)
    extractor.create_struct_label_dataset_reaction_network_based_regression(
        struct_file,
        label_file,
        feature_file,
        group_mode="all",
        one_per_iso_bond_group=True,
    )


def create_struct_label_dataset_reaction_network_based_classification(
    reaction_file,
    struct_file="struct_rxn_ntwk_clfn.sdf",
    label_file="label_rxn_ntwk_clfn.yaml",
    feature_file="feature_rxn_ntwk_clfn.yaml",
):
    extractor = ReactionCollection.from_file(reaction_file)

    extractor.create_struct_label_dataset_reaction_network_based_classification(
        struct_file,
        label_file,
        feature_file,
        group_mode="all",
        top_n=2,
        complement_reactions=True,
        one_per_iso_bond_group=True,
    )


def create_input_file_without_bond_mapping(
    reaction_file,
    mol_file="molecules.sdf",
    mol_attr_file="molecule_attributes.yaml",
    rxn_file="reactions.yaml",
):
    rxn_coll = ReactionCollection.from_file(reaction_file)
    rxn_coll.create_input_files(
        mol_file, mol_attr_file, rxn_file,
    )


if __name__ == "__main__":

    working_dir = to_path("~/Applications/db_access/mol_builder")

    n200 = True
    if n200:
        m = "molecules_n200_qc.pkl"
        r = "reactions_n200_qc.pkl"
        size = "_n200"
    else:
        m = "molecules_qc.pkl"
        r = "reactions_qc.pkl"
        size = ""
    molecule_file = working_dir.joinpath(m)
    reaction_file = working_dir.joinpath(r)

    #
    # extract reactions
    #
    # eg_buckets()
    # eg_extract_A_to_B(molecule_file, reaction_file)
    # eg_extract_A_to_B_C(molecule_file, reaction_file)
    eg_extract_one_bond_break(molecule_file, reaction_file)

    # #
    # # analysis
    # #
    # rxn_coll = ReactionCollection.from_file(reaction_file)
    #
    # print("Counts of broken bond type:")
    # pprint(rxn_coll.get_counts_by_broken_bond_type())
    #
    # print("Counts of reactant charge:")
    # pprint(rxn_coll.get_counts_by_reactant_charge())
    #
    # print("Counts of reactant and product charge:")
    # pprint(rxn_coll.get_counts_by_reaction_charge())
    #
    # rxn_coll.plot_heatmap_of_counts_by_broken_bond_type(
    #     working_dir.joinpath("heatmap_counts_by_bond_type.pdf")
    # )
    #
    # rxn_coll.plot_bar_of_counts_by_reactant_charge(
    #     working_dir.joinpath("barplot_counts_by_reactant_charge.pdf")
    # )
    #
    # rxn_coll.plot_bar_of_counts_by_reaction_charge(
    #     working_dir.joinpath("barplot_counts_by_reaction_charge.pdf")
    # )
    #
    # rxn_coll.plot_histogram_of_reaction_energy(
    #     working_dir.joinpath("hist_reaction_energy.pdf")
    # )
    #
    # rxn_coll.plot_histogram_of_broken_bond_length(
    #     working_dir.joinpath("hist_broken_bond_length.pdf")
    # )

    #
    # write out files for fitting code
    #
    # struct_file = working_dir.joinpath(f"struct_rxn_ntwk_rgrn{size}.sdf")
    # label_file = working_dir.joinpath(f"label_rxn_ntwk_rgrn{size}.yaml")
    # feature_file = working_dir.joinpath(f"feature_rxn_ntwk_rgrn{size}.yaml")
    # create_struct_label_dataset_reaction_network_based_regression(
    #     reaction_file, struct_file, label_file, feature_file
    # )

    reaction_file = "~/Applications/db_access/mol_builder/reactions_1000.pkl"
    create_input_file_without_bond_mapping(
        reaction_file,
        working_dir.joinpath(f"molecules.sdf"),
        working_dir.joinpath(f"molecule_attributes.yaml"),
        working_dir.joinpath(f"reactions.yaml"),
    )
