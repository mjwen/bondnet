import torch
import yaml
import bondnet
import tarfile
import shutil
import tempfile
import os
from pathlib import Path
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.data.dataset import ReactionNetworkDataset
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizerMinimum,
    AtomFeaturizerFull,
    BondAsNodeFeaturizerMinimum,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
from bondnet.core.rdmol import read_rdkit_mols_from_file
from bondnet.data.utils import get_dataset_species
from bondnet.prediction.google_drive import download_file_from_google_drive
from bondnet.utils import (
    load_checkpoints,
    check_exists,
    to_path,
    yaml_load,
)

MODEL_INFO = {
    "mesd": {"date": ["20200808"]},  # default to the last (should be the latest)
    "pubchem": {"date": ["20200531"]},
}

GOOGLE_DRIVE_MODEL_INFO = {
    "mg": {
        "date": ["20200826"],
        "file_id": ["1N_ZKbvNhpStwnY4bCO41q_q6ixzVN7Vx"],  # should corresponds to data
    },
    "phosphorus": {"date": ["20200826"]},
}


def get_model_path(model_name: str) -> os.PathLike:
    """
    Find correct path to the directory storing the model.

    A `model_name` can be provided in various format:
    1. `dataset/data`, in this case the model will be returned directly.
    2. `dataset`, in this case the latest model corresponds to the dataset is returned.
    3. it can also be a path to a local directory where the model info are stored.

    Args:
        model_name: name of the model, e.g. pubchem

    Returns:
        model_path: path storing the model
    """
    # download model stored at Google Drive if haven't yet
    if model_name.lower() in GOOGLE_DRIVE_MODEL_INFO:
        prefix = model_name.lower()

        # directory to store the model
        model_dir = to_path("bondnet_pretrained_model__").joinpath(prefix)

        # if not exist, download it
        if not model_dir.is_dir():
            date = GOOGLE_DRIVE_MODEL_INFO[prefix]["date"][-1]
            file_id = GOOGLE_DRIVE_MODEL_INFO[prefix]["file_id"][-1]
            download_model(file_id, date, model_dir)
        else:
            print(
                f"Found existing model files at {model_dir} for your requested model "
                f"`{model_name}`. We'll reuse it."
            )

        model_name = model_dir

    # model provided as a path
    if to_path(model_name).is_dir():
        model_path = to_path(model_name)

    # model stored with code
    else:
        model_name = model_name.strip().lower()
        if "/" in model_name:
            prefix, date = model_name.split("/")
            if date not in MODEL_INFO[prefix]["date"]:
                raise ValueError(
                    f"expect model date to be one of { MODEL_INFO[prefix]['date'] }, "
                    f"but got {date}."
                )
        else:
            prefix = model_name
            date = MODEL_INFO[prefix]["date"][-1]

        model_path = to_path(bondnet.__file__).parent.joinpath(
            "prediction", "pretrained", prefix, date
        )

    return model_path


def get_model_info(model_path):
    path = model_path.joinpath("model_info.yaml")
    return yaml_load(path)


def load_model(model_path, pretrained=True):

    # NOTE cannot use bondnet.utils.yaml_load, which uses the safe_loader.
    # see: https://github.com/yaml/pyyaml/issues/266
    with open(model_path.joinpath("train_args.yaml"), "r") as f:
        model_args = yaml.load(f, Loader=yaml.Loader)

    model = GatedGCNReactionNetwork(
        in_feats=model_args.feature_size,
        embedding_size=model_args.embedding_size,
        gated_num_layers=model_args.gated_num_layers,
        gated_hidden_size=model_args.gated_hidden_size,
        gated_num_fc_layers=model_args.gated_num_fc_layers,
        gated_graph_norm=model_args.gated_graph_norm,
        gated_batch_norm=model_args.gated_batch_norm,
        gated_activation=model_args.gated_activation,
        gated_residual=model_args.gated_residual,
        gated_dropout=model_args.gated_dropout,
        num_lstm_iters=model_args.num_lstm_iters,
        num_lstm_layers=model_args.num_lstm_layers,
        set2set_ntypes_direct=model_args.set2set_ntypes_direct,
        fc_num_layers=model_args.fc_num_layers,
        fc_hidden_size=model_args.fc_hidden_size,
        fc_batch_norm=model_args.fc_batch_norm,
        fc_activation=model_args.fc_activation,
        fc_dropout=model_args.fc_dropout,
        outdim=1,
        conv="GatedGCNConv",
    )

    if pretrained:
        load_checkpoints(
            {"model": model},
            map_location=torch.device("cpu"),
            filename=model_path.joinpath("checkpoint.pkl"),
        )

    return model


def load_dataset(model_path, molecules, labels, extra_features):
    state_dict_filename = model_path.joinpath("dataset_state_dict.pkl")
    _check_species(molecules, state_dict_filename)
    _check_charge(model_path, extra_features)

    dataset = ReactionNetworkDataset(
        grapher=_get_grapher(model_path),
        molecules=molecules,
        labels=labels,
        extra_features=extra_features,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=state_dict_filename,
    )

    return dataset


def _check_species(molecules, state_dict_filename):
    if isinstance(molecules, (str, Path)):
        check_exists(molecules)
        mols = read_rdkit_mols_from_file(molecules)
    else:
        mols = molecules

    species = get_dataset_species(mols)

    supported_species = torch.load(str(state_dict_filename))["species"]
    not_supported = []
    for s in species:
        if s not in supported_species:
            not_supported.append(s)
    if not_supported:
        not_supported = ",".join(not_supported)
        supported = ",".join(supported_species)
        raise ValueError(
            f"Model trained with a dataset having species: {supported}; Cannot make "
            f"predictions for molecule containing species: {not_supported}."
        )


def _check_charge(model_path, features):
    if isinstance(features, (str, Path)):
        features = yaml_load(features)
    charges = set([x["charge"] for x in features])

    model_info = get_model_info(model_path)
    allowed_charge = set(model_info["allowed_charge"])

    if not charges.issubset(allowed_charge):
        raise ValueError(
            f"Supported molecular charges include {allowed_charge}; "
            f"but the dataset contains molecules of charge {charges}."
        )


def _get_grapher(model_path):
    model_info = get_model_info(model_path)
    allowed_charge = model_info["allowed_charge"]

    featurizer_set = model_info["featurizer_set"]

    if featurizer_set == "full":
        atom_featurizer = AtomFeaturizerFull()
        bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
        # This is used by the pubchem dataset, which only allows charges of 0.
        # We still pass None to allowed_charges because we do not use any charge info
        # in training.
        global_featurizer = GlobalFeaturizer(allowed_charges=None)

    elif featurizer_set == "minimum":
        atom_featurizer = AtomFeaturizerMinimum()
        bond_featurizer = BondAsNodeFeaturizerMinimum(length_featurizer=None)
        global_featurizer = GlobalFeaturizer(allowed_charges=allowed_charge)

    else:
        raise ValueError(
            f"Unsupported featurizer set: {featurizer_set}. Cannot load grapher."
        )

    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )

    return grapher


def download_model(file_id, date, directory):
    """
    Download a shared tarball file of the model from Google Drive with `file_id`,
    untar it and move it to `directory`.

    For info on how to find the file_id, see
    https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
    """

    with tempfile.TemporaryDirectory() as dirpath:

        fname = "pretrained_model.tar.gz"
        fname2 = fname.split(".")[0]
        fname = to_path(dirpath).joinpath(fname)
        fname2 = to_path(dirpath).joinpath(fname2)

        print(
            "Start downloading pretrained model from Google Drive; this may take a while."
        )
        download_file_from_google_drive(file_id, fname)

        if not tarfile.is_tarfile(fname):
            model_path = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            raise RuntimeError(
                f"Failed downloading model from Google Drive. You can try download the "
                f"model manually at: {model_path}, untar it, and pass the path to the "
                f"model to bondnet to use it; i.e. do something like: "
                f"$ bondnet --model <path_to_model> ..."
            )

        tf = tarfile.open(fname, "r:gz")
        tf.extractall(fname2)

        # copy content to the given directory
        # note, we need to joinpath date because the download file from Google Drive
        # with extract to a directory named date
        shutil.copytree(fname2.joinpath(date), to_path(directory))

        print(f"Finish downloading pretrained model; placed at: {to_path(directory)}")
