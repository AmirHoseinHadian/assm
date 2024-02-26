import os
import numpy as np
import pandas as pd

__dir__ = os.path.abspath(os.path.dirname(__file__))


def load_example_dataset():
    """Load sample dataset from Krajbich et al., (2010) for testing and tutorials.

    Parameters
    ----------

    hierarchical_levels : int
         Set to 1 for individual data and to 2 for grouped data.

    n_alternatives : int
        When 2, the dataset of https://doi.org/10.1038/nn.2635
        is loaded.

    """
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "data", "sample_data.csv")
    data = pd.read_csv(data_path, index_col=0)

    return data
