import Compute_Fisher_Noise
import joblib
from joblib import Parallel, delayed
import numpy as np
import time

#compute_array = np.arange(3743, 4999)

#Parallel(n_jobs = joblib.cpu_count(), verbose = 10)(delayed(Compute_Fisher.compute_fisher)(j) for j in compute_array)

Compute_Fisher_Noise.compute_fisher(0)