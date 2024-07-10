# %%
import numpy as np
from scipy import integrate
from scipy import special

import pandas as pd

from pathlib import Path

import joblib
from joblib import Parallel, delayed


# %%
# Reading r Samples

r_array = np.array(pd.read_csv(r'/home2/s4674693/CAMB/r_Samples.csv', index_col = 0))[:,0]

# Reading Transfer Functions

T_transfer_function = np.array(pd.read_csv(r'/home2/s4674693/CAMB/T_Transfer_Function.csv', index_col=0))

E_transfer_function = np.array(pd.read_csv(r'/home2/s4674693/CAMB/E_Transfer_Function.csv', index_col=0))

ell_values = np.array(pd.read_csv(r'/home2/s4674693/CAMB/Transfer_Function_Ell_Values.csv', index_col=0))[:,0]

k_values = np.array(pd.read_csv(r'/home2/s4674693/CAMB/Transfer_Function_k_Values.csv', index_col=0))[:,0]

# %%
# Defining the Alpha Function
def compute_alpha(ell_index, r_values, source):
    '''
    Computes the alpha parameter.
    '''

    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Alpha_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Alpha_Function_E.csv'

    alpha_array = np.zeros((1,len(r_values)+1), 'float64')

    for i in range(len(r_values)):

        alpha_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values * 
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]), x = k_values)

    alpha_array[0, 0] = ell_values[ell_index]
    df = pd.DataFrame(alpha_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)



# Defining the Beta Function
def compute_beta(ell_index, r_values, source):
    '''
    Computes the beta parameter.
    '''

    A_s = 2.09681e-9
    n_s = 0.9652
    k_pivot = 0.05

    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Beta_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Beta_Function_E.csv'

    beta_array = np.zeros((1, len(r_values)+1), 'float64')

    P_k = (2 * (np.pi**2) * A_s * np.power(k_values / k_pivot, n_s - 1)) / (np.power(k_values, 3))

    for i in range(len(r_values)):

        beta_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values *
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]) * P_k, x = k_values)
    
    beta_array[0, 0] = ell_values[ell_index]
    df = pd.DataFrame(beta_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)



# Defining the Delta Function
def compute_delta(ell_index, r_values, source):
    '''
    Computes the delta parameter.
    '''

    A_s = 2.09681e-9
    n_s = 0.9652
    k_pivot = 0.05

    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Delta_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Delta_Function_E.csv'

    delta_array = np.zeros((1, len(r_values)+1), 'float64')
    
    P_k = (2 * (np.pi**2) * A_s * np.power(k_values / k_pivot, n_s - 1)) / (np.power(k_values, 3))

    for i in range(len(r_values)):

        delta_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values * 
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]) * np.power(P_k, 2/3), x = k_values)
        
    delta_array[0, 0] = ell_values[ell_index]
    df = pd.DataFrame(delta_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)



# Defining the Gamma Function
def compute_gamma(ell_index, r_values, source):
    '''
    Computes the gamma parameter.
    '''

    A_s = 2.09681e-9
    n_s = 0.9652
    k_pivot = 0.05

    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Gamma_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Gamma_Function_E.csv'

    gamma_array = np.zeros((1, len(r_values)+1), 'float64')
    
    P_k = (2 * (np.pi**2) * A_s * np.power(k_values / k_pivot, n_s - 1)) / (np.power(k_values, 3))

    for i in range(len(r_values)):

        gamma_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values * 
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]) * np.power(P_k, 1/3), x = k_values)
        
    gamma_array[0, 0] = ell_values[ell_index]
    df = pd.DataFrame(gamma_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)

# %%
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_alpha)(i, r_array, 0) 
                                             for i in range(4999))

Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_alpha)(i, r_array, 1) 
                                             for i in range(4999))

# %%
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_beta)(i, r_array, 0) 
                                             for i in range(4999))

Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_beta)(i, r_array, 1) 
                                             for i in range(4999))

# %%
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_delta)(i, r_array, 0) 
                                             for i in range(4999))

Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_delta)(i, r_array, 1) 
                                             for i in range(4999))

# %%
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_gamma)(i, r_array, 0) 
                                             for i in range(4999))

Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_gamma)(i, r_array, 1) 
                                             for i in range(4999))

