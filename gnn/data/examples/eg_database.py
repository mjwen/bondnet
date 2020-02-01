import os
import subprocess
from gnn.data.database import DatabaseOperation
from gnn.data.utils import TexWriter
from gnn.utils import pickle_dump, pickle_load, yaml_dump, expand_path


def pickle_db_entries():
    entries = DatabaseOperation.query_db_entries(
        db_collection="mol_builder", num_entries=200
    )
    # entries = DatabaseOperation.query_db_entries(db_collection="smd", num_entries=200)

    filename = "~/Applications/db_access/mol_builder/database_n200.pkl"
    pickle_dump(entries, filename)


def pickle_molecules():
    db_collection = "mol_builder"
    entries = DatabaseOperation.query_db_entries(
        db_collection=db_collection, num_entries=None
    )

    mols = DatabaseOperation.to_molecules(entries, db_collection=db_collection)
    filename = "~/Applications/db_access/mol_builder/molecules_unfiltered.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200_unfiltered.pkl"
    pickle_dump(mols, filename)

    mols = DatabaseOperation.filter_molecules(mols, connectivity=True, isomorphism=True)
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    pickle_dump(mols, filename)


def print_mol_property():
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    m = mols[10]

    # get all attributes
    for key, val in vars(m).items():
        print("{}: {}".format(key, val))

    # @property attributes
    properties = [
        "charge",
        "spin_multiplicity",
        "atoms",
        "bonds",
        "species",
        "coords",
        "formula",
        "composition_dict",
        "weight",
    ]
    for prop in properties:
        print("{}: {}".format(prop, getattr(m, prop)))


def plot_molecules():
    plot_prefix = "~/Applications/db_access/mol_builder"

    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename="~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    for m in mols:

        fname = os.path.join(
            plot_prefix,
            "mol_png/{}_{}_{}_{}.png".format(
                m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
            ),
        )
        m.draw(fname, show_atom_idx=True)
        fname = expand_path(fname)
        subprocess.run(["convert", fname, "-trim", "-resize", "100%", fname])

        fname = os.path.join(
            plot_prefix,
            "mol_pdb/{}_{}_{}_{}.pdb".format(
                m.formula, m.charge, m.id, str(m.free_energy).replace(".", "dot")
            ),
        )
        m.write(fname, file_format="pdb")


def write_group_isomorphic_to_file():
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    filename = "~/Applications/db_access/mol_builder/isomorphic_mols.txt"
    DatabaseOperation.write_group_isomorphic_to_file(mols, filename)


def write_dataset():
    filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    # #######################
    # # filter charge 0 mols
    # #######################
    # new_mols = []
    # for m in mols:
    #     if m.charge == 0:
    #         new_mols.append(m)
    # mols = new_mols

    # mols = mols[1 : len(mols) // 2]
    # mols = mols[1 : len(mols) // 4]
    # mols = mols[len(mols) // 4 : len(mols) // 2]
    # mols = mols[len(mols) // 4 : len(mols) * 3 // 8]
    # mols = mols[len(mols) // 4 : len(mols) * 5 // 16]
    # mols = mols[len(mols) * 5 // 16 : len(mols) * 6 // 16]
    # mols = mols[len(mols) * 10 // 32 : len(mols) * 11 // 32]
    # mols = mols[len(mols) * 11 // 32 : len(mols) * 12 // 32]
    # mols = mols[len(mols) * 22 // 64 : len(mols) * 23 // 64]
    # mols = mols[len(mols) * 23 // 64 : len(mols) * 24 // 64]
    # mols = mols[len(mols) * 46 // 128 : len(mols) * 47 // 128]
    # mols = mols[len(mols) * 92 // 256 : len(mols) * 93 // 256]
    # mols = mols[len(mols) * 184 // 512 : len(mols) * 185 // 512]
    # mols = mols[len(mols) * 368 // 1024 : len(mols) * 369 // 1024]
    # mols = mols[len(mols) * 369 // 1024 : len(mols) * 370 // 1024]
    # mols = mols[len(mols) * 738 // 2048 : len(mols) * 739 // 2048]
    # mols = mols[len(mols) * 739 // 2048 : len(mols) * 740 // 2048]

    struct_file = "~/Applications/db_access/mol_builder/struct_mols.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_mols.csv"
    feature_file = "~/Applications/db_access/mol_builder/feature_mols.yaml"
    DatabaseOperation.write_sdf_csv_dataset(mols, struct_file, label_file, feature_file)


def get_single_atom_energy():
    filename = "~/Applications/db_access/mol_builder/molecules_unfiltered.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules.pkl"
    # filename = "~/Applications/db_access/mol_builder/molecules_n200.pkl"
    mols = pickle_load(filename)

    formula = ["H1", "Li1", "C1", "O1", "F1", "P1"]
    print("# formula    free energy    charge")
    for m in mols:
        if m.formula in formula:
            print(m.formula, m.free_energy, m.charge)


def compare_connectivity_mol_builder_and_babel_builder(
    filename="~/Applications/db_access/mol_builder/molecules.pkl",
    # filename="~/Applications/db_access/mol_builder/molecules_n200.pkl",
    tex_file="~/Applications/db_access/mol_builder/mol_connectivity.tex",
):

    # write tex file
    tex_file = expand_path(tex_file)
    with open(tex_file, "w") as f:
        f.write(TexWriter.head())
        f.write(
            "On each page, we plot three mols (top to bottom) from mol builder, "
            "babel builder, and babel builder with metal edge extender.\n"
        )

        mols = pickle_load(filename)
        for m in mols:

            f.write(TexWriter.newpage())
            f.write(TexWriter.verbatim("formula:" + m.formula))
            f.write(TexWriter.verbatim("charge:" + str(m.charge)))
            f.write(TexWriter.verbatim("spin multiplicity:" + str(m.spin_multiplicity)))
            f.write(TexWriter.verbatim("id:" + m.id))

            # mol builder
            fname = "~/Applications/db_access/mol_builder/png_mol_builder/{}.png".format(
                m.id
            )
            fname = expand_path(fname)
            m.draw(fname, show_atom_idx=True)
            subprocess.run(["convert", fname, "-trim", "-resize", "100%", fname])
            f.write(TexWriter.single_figure(fname))
            f.write(TexWriter.verbatim("=" * 80))

            # babel builder
            m.convert_to_babel_mol_graph(use_metal_edge_extender=False)
            fname = "~/Applications/db_access/mol_builder/png_babel_builder/{}.png".format(
                m.id
            )
            fname = expand_path(fname)
            m.draw(fname, show_atom_idx=True)
            subprocess.run(["convert", fname, "-trim", "-resize", "100%", fname])
            f.write(TexWriter.single_figure(fname))
            f.write(TexWriter.verbatim("=" * 80))

            # babel builder with extender
            m.convert_to_babel_mol_graph(use_metal_edge_extender=True)
            fname = "~/Applications/db_access/mol_builder/png_extend_builder/{}.png".format(
                m.id
            )
            fname = expand_path(fname)
            m.draw(fname, show_atom_idx=True)
            subprocess.run(["convert", fname, "-trim", "-resize", "100%", fname])
            f.write(TexWriter.single_figure(fname))

        f.write(TexWriter.tail())


if __name__ == "__main__":
    # pickle_db_entries()
    # pickle_molecules()
    # print_mol_property()
    # plot_molecules()

    write_dataset()
    # write_group_isomorphic_to_file()
    # get_single_atom_energy()

    # compare_connectivity_mol_builder_and_babel_builder()
