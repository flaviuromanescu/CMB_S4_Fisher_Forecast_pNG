import numpy as np
from scipy import integrate
from scipy import special
import pandas as pd
import joblib
from joblib import Parallel, delayed

# Declare home PATH (ex: M:\Folder1\Folder2 (Windows) or /Folder1/Folder2 (Linux))
# If working on Windows, change path separators to '\' instead of '/'
PATH = r'insert_path_here'

# Reading r Samples
r_array = np.array(pd.read_csv(PATH + r'/CAMB_Data/r_Samples.csv', index_col = 0))[:,0]

# Reading Transfer Functions
T_transfer_function = np.array(pd.read_csv(PATH + r'/CAMB_Data/T_Transfer_Function.csv', index_col=0))
E_transfer_function = np.array(pd.read_csv(PATH + r'CAMB_Data/E_Transfer_Function.csv', index_col=0))
ell_values = np.array(pd.read_csv(PATH + r'/CAMB_Data/Transfer_Function_Ell_Values.csv', index_col=0))[:,0]
k_values = np.array(pd.read_csv(PATH + r'/CAMB_Data/Transfer_Function_k_Values.csv', index_col=0))[:,0]

# Defining the functions

# Defining the Alpha Function
def compute_alpha(ell_index, r_values, source):
    '''
    Computes the alpha function and saves it to a.csv file in a non-sorted order.

    Inputs:
    ell_index        (int)            index in the ell_values array
    r_values         (np.ndarray)     array of comoving distances r
    source           (int)            0 -> Temperature; 1 -> Polarization

    Output:
    N/A
    '''

    # Set correct address according to CMB source
    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Alpha_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Alpha_Function_E.csv'

    
    # Generate empty array
    alpha_array = np.zeros((1,len(r_values)+1), 'float64')

    # Looping over r_values and integrating
    for i in range(len(r_values)):

        alpha_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values * 
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]), x = k_values)

    # Index values by the multipole moment l
    alpha_array[0, 0] = ell_values[ell_index]

    # Save to csv
    df = pd.DataFrame(alpha_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)



# Defining the Beta Function
def compute_beta(ell_index, r_values, source):
    '''
    Computes the beta function and saves it to a.csv file in a non-sorted order.

    Inputs:
    ell_index        (int)            index in the ell_values array
    r_values         (np.ndarray)     array of comoving distances r
    source           (int)            0 -> Temperature; 1 -> Polarization

    Output:
    N/A
    '''

    # Set cosmological parameters used for the primordial power spectrum
    A_s = 2.09681e-9
    n_s = 0.9652
    k_pivot = 0.05

    # Set correct address according to CMB source
    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Beta_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Beta_Function_E.csv'

    # Generate empty array
    beta_array = np.zeros((1, len(r_values)+1), 'float64')
    
    # Compute primordial power spectrum
    P_k = (2 * (np.pi**2) * A_s * np.power(k_values / k_pivot, n_s - 1)) / (np.power(k_values, 3))
    
    # Looping over r_values and integrating
    for i in range(len(r_values)):

        beta_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values *
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]) * P_k, x = k_values)

    # Index values by the multipole moment l
    beta_array[0, 0] = ell_values[ell_index]

    # Save to csv
    df = pd.DataFrame(beta_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)



# Defining the Delta Function
def compute_delta(ell_index, r_values, source):
    '''
    Computes the delta function and saves it to a.csv file in a non-sorted order.

    Inputs:
    ell_index        (int)            index in the ell_values array
    r_values         (np.ndarray)     array of comoving distances r
    source           (int)            0 -> Temperature; 1 -> Polarization

    Output:
    N/A
    '''

    # Set cosmological parameters used for the primordial power spectrum
    A_s = 2.09681e-9
    n_s = 0.9652
    k_pivot = 0.05

    # Set correct address according to CMB source
    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Delta_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Delta_Function_E.csv'

    # Generate empty array
    delta_array = np.zeros((1, len(r_values)+1), 'float64')
    
    # Compute primordial power spectrum
    P_k = (2 * (np.pi**2) * A_s * np.power(k_values / k_pivot, n_s - 1)) / (np.power(k_values, 3))

    # Looping over r_values and integrating
    for i in range(len(r_values)):

        delta_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values * 
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]) * np.power(P_k, 2/3), x = k_values)

    # Index values by the multipole moment l
    delta_array[0, 0] = ell_values[ell_index]

    # Save to csv
    df = pd.DataFrame(delta_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)



# Defining the Gamma Function
def compute_gamma(ell_index, r_values, source):
    '''
    Computes the gamma function and saves it to a.csv file in a non-sorted order.

    Inputs:
    ell_index        (int)            index in the ell_values array
    r_values         (np.ndarray)     array of comoving distances r
    source           (int)            0 -> Temperature; 1 -> Polarization

    Output:
    N/A
    '''

    # Set cosmological parameters used for the primordial power spectrum
    A_s = 2.09681e-9
    n_s = 0.9652
    k_pivot = 0.05

    # Set correct address according to CMB source
    if source == 0:

        transfer_function = T_transfer_function
        filename = 'Gamma_Function_T.csv'
    
    else:

        transfer_function = E_transfer_function
        filename = 'Gamma_Function_E.csv'

    # Generate empty array
    gamma_array = np.zeros((1, len(r_values)+1), 'float64')
    
    # Compute primordial power spectrum
    P_k = (2 * (np.pi**2) * A_s * np.power(k_values / k_pivot, n_s - 1)) / (np.power(k_values, 3))

    # Looping over r_values and integrating
    for i in range(len(r_values)):

        gamma_array[0,i+1] = (2 / np.pi) * integrate.trapezoid(transfer_function[:, ell_index] * k_values * k_values * 
                                                           special.spherical_jn(int(ell_values[ell_index]), k_values * r_values[i]) * np.power(P_k, 1/3), x = k_values)

    # Index values by the multipole moment l
    gamma_array[0, 0] = ell_values[ell_index]

    # Save to csv
    df = pd.DataFrame(gamma_array)
    df.to_csv(filename, mode = 'a', index = False, header = False)


# Computing the functions in parallel according to the number of cores available on the system up to l_max = 5000

# Alpha T
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_alpha)(i, r_array, 0) 
                                             for i in range(4999))

# Alpha E
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_alpha)(i, r_array, 1) 
                                             for i in range(4999))

# Beta T
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_beta)(i, r_array, 0) 
                                             for i in range(4999))

# Beta E
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_beta)(i, r_array, 1) 
                                             for i in range(4999))

# Delta T
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_delta)(i, r_array, 0) 
                                             for i in range(4999))

# Delta E
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_delta)(i, r_array, 1) 
                                             for i in range(4999))

# Gamma T
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_gamma)(i, r_array, 0) 
                                             for i in range(4999))

# Gamma E
Parallel(n_jobs = joblib.cpu_count(), verbose = 5)(delayed(compute_gamma)(i, r_array, 1) 
                                             for i in range(4999))


# The functions were computed in parallel and thus are not sorted in terms of l. 

# Reading alpha, beta, gamma, delta csv files
alpha_function_E = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Alpha_Function_E.csv', header = None))
alpha_function_T = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Alpha_Function_T.csv', header = None))

beta_function_E = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Beta_Function_E.csv', header = None))
beta_function_T = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Beta_Function_T.csv', header = None))

delta_function_E = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Delta_Function_E.csv', header = None))
delta_function_T = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Delta_Function_T.csv', header = None))

gamma_function_E = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Gamma_Function_E.csv', header = None))
gamma_function_T = np.array(pd.read_csv(PATH + r'/AlphaBetaGammaDelta/Gamma_Function_T.csv', header = None))


# Sorting by the first column (l values)
alpha_function_T_sorted = alpha_function_T[alpha_function_T[:, 0].argsort(kind = 'mergesort')]
alpha_function_E_sorted = alpha_function_E[alpha_function_E[:, 0].argsort(kind = 'mergesort')]

beta_function_T_sorted = beta_function_T[beta_function_T[:, 0].argsort(kind = 'mergesort')]
beta_function_E_sorted = beta_function_E[beta_function_E[:, 0].argsort(kind = 'mergesort')]

delta_function_T_sorted = delta_function_T[delta_function_T[:, 0].argsort(kind = 'mergesort')]
delta_function_E_sorted = delta_function_E[delta_function_E[:, 0].argsort(kind = 'mergesort')]

gamma_function_T_sorted = gamma_function_T[gamma_function_T[:, 0].argsort(kind = 'mergesort')]
gamma_function_E_sorted = gamma_function_E[gamma_function_E[:, 0].argsort(kind = 'mergesort')]


# Saving sorted functions to csv
alpha_function_T_df = pd.DataFrame(alpha_function_T_sorted)
alpha_function_T_df.to_csv('Alpha_Function_T_Sorted.csv', index = False, header = False)

alpha_function_E_df = pd.DataFrame(alpha_function_E_sorted)
alpha_function_E_df.to_csv('Alpha_Function_E_Sorted.csv', index = False, header = False)


beta_function_T_df = pd.DataFrame(beta_function_T_sorted)
beta_function_T_df.to_csv('Beta_Function_T_Sorted.csv', index = False, header = False)

beta_function_E_df = pd.DataFrame(beta_function_E_sorted)
beta_function_E_df.to_csv('Beta_Function_E_Sorted.csv', index = False, header = False)


delta_function_T_df = pd.DataFrame(delta_function_T_sorted)
delta_function_T_df.to_csv('Delta_Function_T_Sorted.csv', index = False, header = False)

delta_function_E_df = pd.DataFrame(delta_function_E_sorted)
delta_function_E_df.to_csv('Delta_Function_E_Sorted.csv', index = False, header = False)


gamma_function_T_df = pd.DataFrame(gamma_function_T_sorted)
gamma_function_T_df.to_csv('Gamma_Function_T_Sorted.csv', index = False, header = False)

gamma_function_E_df = pd.DataFrame(gamma_function_E_sorted)
gamma_function_E_df.to_csv('Gamma_Function_E_Sorted.csv', index = False, header = False)
