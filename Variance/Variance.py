# %%
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import interpolate
from scipy import special
import sympy
from sympy.physics import wigner

import pandas as pd

from pathlib import Path
import sys, platform, os

import joblib
from joblib import Parallel, delayed

import itertools

import wigners
import py3nj

import pyautogui
import time
import subprocess

# %%
fisher_array = np.zeros((7,4999, 4))
fisher_array_sorted = np.zeros((7,4999, 4))

TCMB = 2.7255
TCMB_sq = TCMB**2

fisher_array[0] = (TCMB_sq)**3 * np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_local+local.csv', header = None))
fisher_array[1] = (TCMB_sq)**3 * np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_equilateral+equilateral.csv', header = None))
fisher_array[2] = (TCMB_sq)**3 * np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_orthogonal+orthogonal.csv', header = None))
fisher_array[3] = (TCMB_sq)**(3/2) * np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_local+ISW.csv', header = None))
fisher_array[4] = (TCMB_sq)**(3/2) * np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_equilateral+ISW.csv', header = None))
fisher_array[5] = (TCMB_sq)**(3/2) * np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_orthogonal+ISW.csv', header = None))
fisher_array[6] = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Variance\Fisher_Elements_ISW+ISW.csv', header = None))

for i in range(7):

    temp_array = fisher_array[i]
    fisher_array_sorted[i] = temp_array[temp_array[:,0].argsort(kind = 'mergesort')]

# %%
type_array = ['local', 'equilateral', 'orthogonal', 'ISW']
bispectra_order_array = np.array([[0, 0], [1, 1], [2, 2], [0, 3], [1, 3], [2, 3], [3,3]])


for i in range(len(bispectra_order_array)):

        address = 'Fisher_Elements_Sorted' + '_' + type_array[bispectra_order_array[i, 0]] + '+' + type_array[bispectra_order_array[i, 1]] + '.csv'

        df = pd.DataFrame(fisher_array_sorted[i])
        df.to_csv(address, index = False, header = False)

# %% [markdown]
# Calculate Fisher

# %%
max_index = 4999

Fisher = np.zeros((7,3))
Variance = np.zeros(3)

for i in range(7):

    for j in range(max_index):

        for k in range(1,4):

            Fisher[i,k-1] += fisher_array_sorted[i,j,k]


for q in range(3):

    Variance[q] = 1 / np.sqrt(Fisher[q,2])

print(Fisher)

print(Variance)

# %% [markdown]
# Marginalization

# %%
F_local_ISW = np.array([[Fisher[0,2], Fisher[3,2]], [Fisher[3,2], Fisher[6,2]]])
F_equilateral_ISW = np.array([[Fisher[1,2], Fisher[4,2]], [Fisher[4,2], Fisher[6,2]]])
F_orthogonal_ISW = np.array([[Fisher[2,2], Fisher[5,2]], [Fisher[5,2], Fisher[6,2]]])

print(F_local_ISW)


Variance_lensed = np.zeros(3)

Variance_lensed[0] = np.sqrt(np.linalg.inv(F_local_ISW)[0, 0])
Variance_lensed[1] = np.sqrt(np.linalg.inv(F_equilateral_ISW)[0, 0])
Variance_lensed[2] = np.sqrt(np.linalg.inv(F_orthogonal_ISW)[0, 0])

Fisher_lensed = np.zeros(3)

Fisher_lensed = (1 / Variance_lensed) ** 2

print(Fisher_lensed)

print(Variance_lensed)

print(F_local_ISW[0,1] / F_local_ISW[0,0])
print(F_equilateral_ISW[0,1] / F_equilateral_ISW[0,0])
print(F_orthogonal_ISW[0,1] / F_orthogonal_ISW[0,0])


