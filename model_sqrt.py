import numpy as np
import csv

def model_sqrt(n, quantile):
    # Parameters fitted for 99.9% quantile of necessary edges for connectivity
    if quantile == 0.999:
        a, b, c = 3.484291, -2.983427, -12.853034
    elif quantile == 0.99:
        a, b, c = 2.314929,  -1.525401, -10.056070
    elif quantile == 0.95:
        a, b, c = 1.501966, -1.310822, -5.243723
    elif quantile == 0.9:
        a, b, c = 1.137098, -1.063811, -3.959385
    elif quantile == 0.8:
        a, b, c = 0.7609406, -0.9257601, -2.3286719
    elif quantile == 0.7:
        a, b, c = 0.5239999, -0.7869253, -1.5918337
    elif quantile == 0.6:
        a, b, c = 0.3446035, -0.7454735, -0.6189292
    elif quantile == 0.5:
        a, b, c = 0.1913904, -0.6876843,  0.1371878

    edges = (0.5 * n * np.log(n)) + (a * n) + b * np.sqrt(n) + c
    max_edges = n * (n-1) / 2

    return edges / max_edges

def model_flexible(n, connectivity):
    # defined only for 0.999%-quantile
    # connectivity is the expected minimum connectivity rate
    if connectivity == 0.9:
        a, b, c, d = -0.007560676,  1.347482858,  3.281456663,  2.067671992
    elif connectivity == 0.5:
        a, b, c, d = 0.03601871,  0.36608633,  4.61962889, -9.06528983
    edges = a * n * np.log(n) + b * n + c * np.sqrt(n) + d
    max_edges = n * (n-1) / 2


    return edges / max_edges

# connectivity in this model is between 1 and 35 and corresponds to
# the expected max clique size in terms that 0 has a max clique size of at least (0.05 * n)
# and an increase of 1 in connectivity increases max clique size from at least (x*n) to (x+0.025)*n
# so that the maximum least max clique size that can be expected lies at (0.9*n).
def model_flexible_v2(n, connectivity):
    if connectivity == 35:
        a, b, c, d = -0.007560676,  1.347482858,  3.281456663,  2.067671992
    else:
        with open('model_parameters.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_ind = 1
            for line in csv_reader:
                if line_ind != connectivity:
                    line_ind += 1
                else:
                    a, b = np.float64(line["a"]), np.float64(line["b"])
                    c, d = np.float64(line["c"]), np.float64(line["d"])
                    break

    edges = a * n * np.log(n) + b * n + c * np.sqrt(n) + d
    max_edges = n * (n-1) / 2

    return edges / max_edges
