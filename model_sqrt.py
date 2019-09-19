import numpy as np

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