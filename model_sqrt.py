import numpy as np
import csv
"""
This module implements models for chosing an appropriate sampling rate.
"""

def load_model_flexible_v2(model_file):
    """
    This function loads all model parameters from a chosen csv-file.
    """
    with open(model_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        a = np.zeros(37, dtype=np.float64)
        b = np.zeros(37, dtype=np.float64)
        c = np.zeros(37, dtype=np.float64)
        d = np.zeros(37, dtype=np.float64)

        i = 0
        for line in csv_reader:
            a[i], b[i] = np.float64(line["V1"]), np.float64(line["V2"])
            c[i], d[i] = np.float64(line["V3"]), np.float64(line["V4"])
            i += 1

        a[36], b[36], c[36], d[36] = -0.007560676,  1.347482858,  3.281456663,  2.067671992
    return(a, b, c, d)


# connectivity in this model is between 0 and 36 and corresponds to
# the expected max clique size in terms that 0 has a max clique size of at least (0.05 * n)
# and an increase of 1 in connectivity increases max clique size from at least (x*n) to (x+0.025)*n
# so that the maximum least max clique size that can be expected lies at (0.925*n).
def model_flexible_v2(model_pars_b, model_pars_a, n, connectivity):
    """
    This function takes two model parameter sets (return values of load_model_flexible_v2)
    It chooses the right parameters from the set and applies the (resulting) model to a specific n, returning a sampling rate (NOT number of edges!)
    """
    if n < 100:
        model_pars = model_pars_b
    else:
        model_pars = model_pars_a
    a = model_pars[0]
    b = model_pars[1]
    c = model_pars[2]
    d = model_pars[3]

    edges = a[connectivity] * n * np.log(n) + b[connectivity] * n + c[connectivity] * np.sqrt(n) + d[connectivity]
    max_edges = n * (n-1) / 2

    return edges / max_edges
