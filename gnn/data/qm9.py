"""
QM9 dataset.
"""

import pandas as pd
import numpy as np
import logging
from collections import OrderedDict
from gnn.data.electrolyte import ElectrolyteMoleculeDataset

logger = logging.getLogger(__name__)


class QM9Dataset(ElectrolyteMoleculeDataset):
    """
    The QM9 dataset.
    """

    def _read_label_file(self):
        """
        Returns:
            rst (2D array): shape (N, M), where N is the number of lines (excluding the
                header line), and M is the number of columns (exluding the first index
                column).
            extensive (list): size (M), indicating whether the corresponding data in
                rst is extensive property or not.
        """

        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()

        h2e = 27.211396132  # Hatree to eV
        k2e = 0.0433634  # kcal/mol to eV

        # supported property
        supp_prop = OrderedDict()
        supp_prop["A"] = {"uc": 1.0, "extensive": True}
        supp_prop["B"] = {"uc": 1.0, "extensive": True}
        supp_prop["C"] = {"uc": 1.0, "extensive": True}
        supp_prop["mu"] = {"uc": 1.0, "extensive": True}
        supp_prop["alpha"] = {"uc": 1.0, "extensive": True}
        supp_prop["homo"] = {"uc": h2e, "extensive": False}
        supp_prop["lumo"] = {"uc": h2e, "extensive": False}
        supp_prop["gap"] = {"uc": h2e, "extensive": False}
        supp_prop["r2"] = {"uc": 1.0, "extensive": True}
        supp_prop["zpve"] = {"uc": h2e, "extensive": True}
        supp_prop["u0"] = {"uc": h2e, "extensive": True}
        supp_prop["u298"] = {"uc": h2e, "extensive": True}
        supp_prop["h298"] = {"uc": h2e, "extensive": True}
        supp_prop["g298"] = {"uc": h2e, "extensive": True}
        supp_prop["cv"] = {"uc": 1.0, "extensive": True}
        supp_prop["u0_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["u298_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["h298_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["g298_atom"] = {"uc": k2e, "extensive": True}

        if self.properties is not None:

            for prop in self.properties:
                if prop not in supp_prop:
                    raise ValueError(
                        "Property '{}' not supported. Supported ones are: {}".format(
                            prop, supp_prop.keys()
                        )
                    )
            supp_list = list(supp_prop.keys())
            indices = [supp_list.index(p) for p in self.properties]
            rst = rst[:, indices]
            convert = [supp_prop[p]["uc"] for p in self.properties]
            extensive = [supp_prop[p]["extensive"] for p in self.properties]
        else:
            convert = [v["uc"] for k, v in supp_prop.items()]
            extensive = [v["extensive"] for k, v in supp_prop.items()]

        if self.unit_conversion:
            rst = np.multiply(rst, convert)

        return rst, extensive
