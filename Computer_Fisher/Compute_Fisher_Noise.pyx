import numpy as np
import cython
cimport numpy as cnp

from scipy import integrate
import pandas as pd

from pathlib import Path

import joblib
from joblib import Parallel, delayed

import itertools

import py3nj

cnp.import_array()

cdef cnp.ndarray ell_values = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\Transfer_Function_Ell_Values.csv', index_col=0), dtype = np.intc)[:,0]

cdef cnp.ndarray k_values = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\Transfer_Function_k_Values.csv', index_col=0), dtype = np.double)[:,0]

cdef cnp.ndarray r_array = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\r_Samples.csv', index_col = 0), dtype = np.double)[:,0]

# Alpha Functions
cdef cnp.ndarray alpha_function_E = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Alpha_Function_E_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray alpha_function_T = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Alpha_Function_T_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray alpha_function = np.array([alpha_function_T, alpha_function_E], dtype = np.double)

# Beta Functions
cdef cnp.ndarray beta_function_E = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Beta_Function_E_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray beta_function_T = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Beta_Function_T_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray beta_function = np.array([beta_function_T, beta_function_E], dtype = np.double)

# Delta Functions
cdef cnp.ndarray delta_function_E = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Delta_Function_E_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray delta_function_T = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Delta_Function_T_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray delta_function = np.array([delta_function_T, delta_function_E], dtype = np.double)

# Gamma Functions
cdef cnp.ndarray gamma_function_E = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Gamma_Function_E_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray gamma_function_T = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Alpha_Beta_Gamma_Delta\Gamma_Function_T_Sorted.csv', header = None), dtype = np.double)[:,1:]
cdef cnp.ndarray gamma_function = np.array([gamma_function_T, gamma_function_E], dtype = np.double)

# Power Spectra
cdef cnp.ndarray TT_power =  np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\TT_Power_Spectrum.csv'), dtype = np.double)
cdef cnp.ndarray EE_power =  np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\EE_Power_Spectrum.csv'), dtype = np.double)
cdef cnp.ndarray TE_power =  np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\TE_Power_Spectrum.csv'), dtype = np.double)

cdef cnp.ndarray PP_power =  np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\PP_Power_Spectrum.csv'), dtype = np.double)
cdef cnp.ndarray PT_power =  np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\PT_Power_Spectrum.csv'), dtype = np.double)
cdef cnp.ndarray PE_power =  np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Transfer_Functions\PE_Power_Spectrum.csv'), dtype = np.double)

cdef cnp.ndarray CMB_S4_Noise = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Noise_Calculation\CMB_S4_Final_Noise.csv', header = None), dtype = np.double)
cdef cnp.ndarray CMB_S4_Noise_TT = CMB_S4_Noise[:,1]
cdef cnp.ndarray CMB_S4_Noise_EE = CMB_S4_Noise[:,2]

cdef cnp.ndarray cross_lens_power_spectra = np.array([PT_power, PE_power, PP_power], dtype = np.double)[:,:,1]

# Combined power spectra array and inverse covariance array
cdef cnp.ndarray power_spectra = np.zeros((2,2,len(TT_power)), dtype = np.double)
cdef cnp.ndarray inverse_covariance_matrix = np.zeros((2,2,len(TT_power)), dtype = np.double)

cdef int l_min_index = 0
cdef int q

for q in range(len(TT_power)):

    power_spectra[0,0,q] = TT_power[q,1] + CMB_S4_Noise_TT[q]
    power_spectra[1,1,q] = EE_power[q,1] + CMB_S4_Noise_EE[q]
    power_spectra[1,0,q] = TE_power[q,1]
    power_spectra[0,1,q] = TE_power[q,1]

    if q >=2:
        inverse_covariance_matrix[:,:,q] = np.linalg.inv(power_spectra[:,:,q])


options = [0,1]
cdef cnp.ndarray X_options = np.array(list(itertools.product(options, repeat = 6)), dtype = np.intc)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double f_coefficient(int source, int l1_index, int l2_index, int l3_index):

    cdef int l1 = ell_values[l1_index]
    cdef int l2 = ell_values[l2_index]
    cdef int l3 = ell_values[l3_index]

    if source == 0:

        return ((l2 * (l2 + 1) + l3 * (l3 + 1) - l1 * (l1 + 1)) / 2)
    
    if source == 1:

        return (((l2 * (l2 + 1) + l3 * (l3 + 1) - l1 * (l1 + 1)) / 2) * 
                py3nj.wigner3j(int(l1 * 2), int(l2 * 2), int(l3 * 2), 4, 0, -4)) / (py3nj.wigner3j(int(l1 * 2), int(l2 * 2), int(l3 * 2), 0, 0, 0))



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double bispectrum(int type, int source1, int source2, int source3, int l1_index, int l2_index, int l3_index):
    """
    type:
    0 -> local
    1 -> equilateral
    2 -> orthogonal
    3 -> ISW
    """

    cdef int a = source1
    cdef int b = source2
    cdef int c = source3

    cdef int l1 = ell_values[l1_index]
    cdef int l2 = ell_values[l2_index]
    cdef int l3 = ell_values[l3_index]

    cdef double value = 0

    if type == 0:

        value = (6.0 / 5.0) * integrate.trapezoid(r_array * r_array *
                                            (beta_function[a, l3_index] * beta_function[b, l2_index] * alpha_function[c, l1_index] + 
                                            beta_function[b, l2_index] * beta_function[c, l1_index] * alpha_function[a, l3_index] + 
                                            beta_function[c, l1_index] * beta_function[a, l3_index] * alpha_function[b, l2_index]), x = r_array)
        
    if type == 1:

        value = (18.0 / 5.0) * integrate.trapezoid(r_array * r_array * 
                                            (-beta_function[a, l3_index] * beta_function[b, l2_index] * alpha_function[c, l1_index] - 
                                            beta_function[b, l2_index] * beta_function[c, l1_index] * alpha_function[a, l3_index] - 
                                            beta_function[c, l1_index] * beta_function[a, l3_index] * alpha_function[b, l2_index] -
                                            2.0 * delta_function[c, l1_index] * delta_function[b, l2_index] * delta_function[a, l3_index] +
                                            beta_function[c, l1_index] * gamma_function[b, l2_index] * delta_function[a, l3_index] +
                                            beta_function[c, l1_index] * gamma_function[a, l3_index] * delta_function[b, l2_index] +
                                            beta_function[b, l2_index] * gamma_function[c, l1_index] * delta_function[a, l3_index] +
                                            beta_function[b, l2_index] * gamma_function[a, l3_index] * delta_function[c, l1_index] +
                                            beta_function[a, l3_index] * gamma_function[c, l1_index] * delta_function[b, l2_index] +
                                            beta_function[a, l3_index] * gamma_function[b, l2_index] * delta_function[c, l1_index]), x = r_array)
    
    if type == 2:

        value = (18.0 / 5.0) * integrate.trapezoid(r_array * r_array * 
                                            (-3.0 * beta_function[a, l3_index] * beta_function[b, l2_index] * alpha_function[c, l1_index] - 
                                            3.0 * beta_function[b, l2_index] * beta_function[c, l1_index] * alpha_function[a, l3_index] - 
                                            3.0 * beta_function[c, l1_index] * beta_function[a, l3_index] * alpha_function[b, l2_index] -
                                            8.0 * delta_function[c, l1_index] * delta_function[b, l2_index] * delta_function[a, l3_index] +
                                            3.0 * beta_function[c, l1_index] * gamma_function[b, l2_index] * delta_function[a, l3_index] +
                                            3.0 * beta_function[c, l1_index] * gamma_function[a, l3_index] * delta_function[b, l2_index] +
                                            3.0 * beta_function[b, l2_index] * gamma_function[c, l1_index] * delta_function[a, l3_index] +
                                            3.0 * beta_function[b, l2_index] * gamma_function[a, l3_index] * delta_function[c, l1_index] +
                                            3.0 * beta_function[a, l3_index] * gamma_function[c, l1_index] * delta_function[b, l2_index] +
                                            3.0 * beta_function[a, l3_index] * gamma_function[b, l2_index] * delta_function[c, l1_index]), x = r_array)
        
    if type == 3:

        value = cross_lens_power_spectra[source2, l2] * power_spectra[source1, source3, l3] * f_coefficient(source1, l1_index, l2_index, l3_index) + \
                cross_lens_power_spectra[source3, l3] * power_spectra[source1, source2, l2] * f_coefficient(source1, l1_index, l3_index, l2_index) + \
                cross_lens_power_spectra[source1, l1] * power_spectra[source2, source3, l3] * f_coefficient(source2, l2_index, l1_index, l3_index) + \
                cross_lens_power_spectra[source3, l3] * power_spectra[source1, source2, l1] * f_coefficient(source2, l2_index, l3_index, l1_index) + \
                cross_lens_power_spectra[source1, l1] * power_spectra[source2, source3, l2] * f_coefficient(source3, l3_index, l1_index, l2_index) + \
                cross_lens_power_spectra[source2, l2] * power_spectra[source1, source3, l1] * f_coefficient(source3, l3_index, l2_index, l1_index)

    return value


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_fisher(int l1_index):
    """
    Computer the Fisher Matrix element (both diagonal and off-diagonal) for type1 and type2 bispectra.

    Arguments:
    l1_index        (int)       value of first index
    type1           (string)    local / equilateral / orthogonal / ISW
    type1           (string)    local / equilateral / orthogonal / ISW

    Returns:
    fisher_element    (float)     term in the fisher sum with first l = l1
    """    

    cdef cnp.ndarray fisher_array = np.zeros((4, 7), dtype = np.double)
    cdef cnp.ndarray type_int_array = np.array([0, 1, 2, 3], dtype = np.intc)
    type_array = ['local', 'equilateral', 'orthogonal', 'ISW']
    cdef cnp.ndarray bispectra_order_array = np.array([[0, 0], [1, 1], [2, 2], [0, 3], [1, 3], [2, 3], [3,3]], dtype = np.intc)
    cdef cnp.ndarray fisher_element = np.zeros(len(bispectra_order_array), dtype = np.double)

    # Bispectrum Array
    cdef cnp.ndarray bispectrum_array = np.zeros((2,2,2,4))

    cdef int delta_l,a,b,c,d,e,f,l2_index,l3_index,i,k,l1,l2,l3
    cdef double wigner_factor, inv_cov_term

    # Sum over l and modes
    for l2_index in range(l_min_index, l1_index + 1):

        for l3_index in range(l_min_index, l2_index + 1):


            l1 = ell_values[l1_index]
            l2 = ell_values[l2_index]
            l3 = ell_values[l3_index]

            # Triangle Condition
            if l1 < l2 + l3 and l2 < l1 + l3 and l3 < l1 + l2 and (l1 + l2 + l3) % 2 == 0:

                # Calculating Delta

                if l1 == l2  and l2 == l3:

                    delta_l = 6

                elif l1 == l2 or l1 == l3 or l2 == l3:

                    delta_l = 2
                
                else:

                    delta_l = 1

                wigner_factor = (py3nj.wigner3j(int(l1 * 2), int(l2 * 2), int(l3 * 2), 0, 0, 0) ** 2)

                
                # Loop over T and/or E
                for a,b,c,d,e,f in X_options:

                    inv_cov_term = inverse_covariance_matrix[c, f, l1] * inverse_covariance_matrix[b, e, l2] * inverse_covariance_matrix[a, d, l3]

                    for i in range(len(type_array)):

                        if bispectrum_array[a,b,c,i] == 0:

                            bispectrum_array[a,b,c,i] = bispectrum(type_int_array[i], a, b, c, l1_index, l2_index, l3_index)

                        if bispectrum_array[d,e,f,i] == 0:

                            bispectrum_array[d,e,f,i] = bispectrum(type_int_array[i], d, e, f, l1_index, l2_index, l3_index)

                    fisher_element = np.zeros(len(bispectra_order_array), dtype = np.double)

                    for k in range(len(bispectra_order_array)):

                        fisher_element[k] = wigner_factor * ((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) * (1 / (4 * np.pi))) * \
                                            bispectrum_array[a,b,c,bispectra_order_array[k, 0]] * bispectrum_array[d,e,f,bispectra_order_array[k, 1]] * inv_cov_term * (1 / delta_l)

                        fisher_array[3,k] += fisher_element[k]

                        if a == b == c == d == e == f == 0:

                            fisher_array[1,k]+= fisher_element[k]
                        
                        if a == b == c == d == e == f == 1:

                            fisher_array[2,k] += fisher_element[k]

            bispectrum_array = np.zeros((2,2,2,4))

    fisher_array[0,:] = l1

    for i in range(len(bispectra_order_array)):

        address = 'Fisher_Elements_Noise' + '_' + type_array[bispectra_order_array[i, 0]] + '+' + type_array[bispectra_order_array[i, 1]] + '.csv'

        df = pd.DataFrame(fisher_array[:,i].reshape((1,4)))
        df.to_csv(address, mode = 'a', index = False, header = False)
