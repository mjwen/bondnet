import itertools
import numpy as np
from collections import defaultdict
import matplotlib as mpl
from matplotlib import pyplot as plt
from gnn.data.reaction import ReactionExtractor
from pprint import pprint
from gnn.utils import pickle_load, expand_path, create_directory
from gnn.data.utils import TexWriter


def eg_buckets():
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge", "spin_multiplicity"])
    pprint(extractor.buckets)
    extractor.bucket_molecules(keys=["formula"])
    pprint(extractor.buckets)


def eg_extract_A_to_B():
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_A_to_B_style_reaction()
    extractor.to_file(
        filename="~/Applications/mongo_db_access/extracted_mols/reaction_A2B.pkl"
    )


def eg_extract_A_to_B_C():
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets", len(extractor.buckets))

    extractor.extract_A_to_B_C_style_reaction()
    extractor.to_file(
        filename="~/Applications/mongo_db_access/extracted_mols/reactions_A2BC.pkl"
    )


def eg_extract_one_bond_break():
    # filename = "~/Applications/mongo_db_access/extracted_mols/molecules.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/molecules_n200.pkl"
    molecules = pickle_load(filename)
    print("number of moles:", len(molecules))

    extractor = ReactionExtractor(molecules)
    extractor.bucket_molecules(keys=["formula", "charge"])
    print("number of buckets:", len(extractor.buckets))

    extractor.extract_one_bond_break()

    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    extractor.to_file(filename)


def subselect_reactions():
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
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

    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_C5H8Li1O3.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_charge0.pkl"
    extractor.to_file(filename)


def reactants_bond_energies_to_file():
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions_C5H8Li1O3.pkl"
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

    filename = "~/Applications/mongo_db_access/extracted_mols/bond_energies.yaml"
    # filename = "~/Applications/mongo_db_access/extracted_mols/bond_energies_n200.yaml"
    extractor.write_bond_energies(filename)


def plot_reaction_energy_difference_arcoss_reactant_charge(
    filename="~/Applications/mongo_db_access/extracted_mols/reactions.pkl",
    # filename="~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl",
    # filename="~/Applications/mongo_db_access/extracted_mols/reactions_C5H8Li1O3.pkl",
):
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

            outname = "~/Applications/mongo_db_access/extracted_mols/reactant_e_diff/"
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
    filename="~/Applications/mongo_db_access/extracted_mols/reactions.pkl",
    # filename="~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl",
):
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
        filename = "~/Applications/mongo_db_access/extracted_mols/bond_type_count_{}.pdf".format(
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
    filename="~/Applications/mongo_db_access/extracted_mols/reactions.pkl",
    # filename="~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl",
):
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

            outname = "~/Applications/mongo_db_access/extracted_mols/bond_energy_hist/"
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
    filename="~/Applications/mongo_db_access/extracted_mols/reactions.pkl",
    # filename="~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl",
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

    # TODO this should be moved to Reaction (coords should be moved to WrapperMolecule)
    def get_length_of_bond(rxn):
        reactant = rxn.reactants[0]
        coords = np.asarray([s.coords for s in reactant.pymatgen_mol.sites])
        dist = [np.linalg.norm(coords[u] - coords[v]) for u, v, _ in reactant.bonds]
        return dist

    # prepare data
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.get_reactions_with_lowest_energy()
    data = [get_length_of_bond(rxn) for rxn in all_reactions]
    data = np.concatenate(data)

    print("\n\n@@@ all bond length min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/mongo_db_access/extracted_mols/bond_length_all.pdf"
    filename = expand_path(filename)
    plot_hist(data, filename)


def plot_broken_bond_length_hist(
    filename="~/Applications/mongo_db_access/extracted_mols/reactions.pkl",
    # filename="~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl",
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

    # TODO this should be moved to Reaction (coords should be moved to WrapperMolecule)
    def get_length_of_broken_bond(rxn):
        coords = np.asarray([s.coords for s in rxn.reactants[0].pymatgen_mol.sites])
        u, v = rxn.get_broken_bond()
        dist = np.linalg.norm(coords[u] - coords[v])
        return dist

    # prepare data
    extractor = ReactionExtractor.from_file(filename)
    all_reactions = extractor.get_reactions_with_lowest_energy()
    data = [get_length_of_broken_bond(rxn) for rxn in all_reactions]

    print("\n\n@@@ broken bond length min={}, max={}".format(min(data), max(data)))
    filename = "~/Applications/mongo_db_access/extracted_mols/bond_length_broken.pdf"
    filename = expand_path(filename)
    plot_hist(data, filename)


def create_struct_label_dataset_mol_based():
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_charge0.pkl"
    extractor = ReactionExtractor.from_file(filename)
    extractor.create_struct_label_dataset_mol_based(
        struct_name="~/Applications/mongo_db_access/extracted_mols/struct.sdf",
        label_name="~/Applications/mongo_db_access/extracted_mols/label.txt",
        feature_name="~/Applications/mongo_db_access/extracted_mols/feature.yaml",
        # struct_file="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature_n200.yaml",
        # struct_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0.txt",
    )


def create_struct_label_dataset_bond_based():
    filename = "~/Applications/mongo_db_access/extracted_mols/reactions.pkl"
    # filename = "~/Applications/mongo_db_access/extracted_mols/reactions_n200.pkl"
    extractor = ReactionExtractor.from_file(filename)

    ##############
    # filter reactant charge = 0
    ##############
    extractor.filter_reactions_by_reactant_charge(charge=0)

    # ##############
    # # filter C-C bond
    # ##############
    # extractor.filter_reactions_by_bond_type_and_order(bond_type=("C", "C"))

    extractor.create_struct_label_dataset_bond_based(
        # struct_file="~/Applications/mongo_db_access/extracted_mols/struct.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature.yaml",
        # struct_file="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature_n200.yaml",
        struct_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0.sdf",
        label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0.txt",
        feature_file="~/Applications/mongo_db_access/extracted_mols/feature_charge0.yaml",
        # struct_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0_CC.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0_CC.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature_charge0_CC.yaml",
    )


if __name__ == "__main__":
    # eg_buckets()
    # eg_extract_A_to_B()
    # eg_extract_A_to_B_C()
    # eg_extract_one_bond_break()
    subselect_reactions()

    # plot_reaction_energy_difference_arcoss_reactant_charge()
    # plot_bond_type_heat_map()
    # plot_bond_energy_hist()
    # plot_broken_bond_length_hist()
    # plot_all_bond_length_hist()

    # reactants_bond_energies_to_file()
    # eg_create_struct_label_dataset_mol_based()
    # create_struct_label_dataset_bond_based()
