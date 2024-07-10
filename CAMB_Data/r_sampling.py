# %%
import numpy as np

import pandas as pd

from pathlib import Path
import sys, platform, os

# %% [markdown]
# First, we sample the comoving coordinate $r$ according to Table 1 of Liguori et al. PRD, 76, 105016 (2007) [https://arxiv.org/pdf/0708.3786]. $r$ is measured in Mpc.

# %%
# Sample r
r_max = 14400

r_array = np.array([r_max])

r = r_max

while r > 105:

    if 12632 < r <= 14400:

        r -= 3.5
        r_array = np.append(r_array, r)

    elif 10007 < r <= 12632:

        r -= 105
        r_array = np.append(r_array, r)

    elif 9377 < r <= 10007:

        r -= 35
        r_array = np.append(r_array, r)

    elif 105 < r <= 9377:

        r -= 105
        r_array = np.append(r_array, r)


# %%
# Writing to CSV

r_array_dataframe = pd.DataFrame(r_array)
r_array_dataframe.to_csv('r_Samples.csv')


