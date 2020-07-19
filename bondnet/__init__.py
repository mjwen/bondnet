__version__ = "0.0.1"

import logging
import os

logging.basicConfig(
    filename="bondnet.log",
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    level=logging.INFO,
)

os.environ["DGLBACKEND"] = "pytorch"
