import Compute_Fisher_Noise
import joblib
from joblib import Parallel, delayed
import numpy as np

# Defining array of multipoles
compute_array = np.arange(0, 4999)

# Compute fisher elements with no noise
Parallel(n_jobs = joblib.cpu_count(), verbose = 10)(delayed(Compute_Fisher.compute_fisher)(j, 0) for j in compute_array)

# Compute fisher elements with noise
Parallel(n_jobs = joblib.cpu_count(), verbose = 10)(delayed(Compute_Fisher.compute_fisher)(j, 1) for j in compute_array)
