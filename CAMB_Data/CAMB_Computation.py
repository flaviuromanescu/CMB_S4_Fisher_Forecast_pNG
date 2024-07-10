# %% [markdown]
# Used as reference: [https://camb.readthedocs.io/en/latest/CAMBdemo.html]

# %%
import numpy as np

import camb
from camb import model, initialpower
import matplotlib
from matplotlib import pyplot as plt

import pandas as pd

# %%
# CAMB Parameters from Planck 2018

parameters = camb.CAMBparams()
parameters.set_cosmology(H0=67.37, ombh2=0.02233, omch2=0.1198, tau = 0.0543, TCMB=2.7255)
parameters.InitPower.set_params(ns = 0.9652, As = 2.09681e-9, pivot_scalar = 0.05)
parameters.set_for_lmax(5000)
parameters.set_accuracy(AccuracyBoost = 2, lAccuracyBoost = 2, lSampleBoost = 100)
parameters.WantScalars = True

# %%
# Get Transfer Functions

transfer_function_data = camb.get_transfer_functions(parameters).get_cmb_transfer_data()

T_transfer = transfer_function_data.delta_p_l_k[0, :, :]
E_transfer = transfer_function_data.delta_p_l_k[1, :, :]

ells = transfer_function_data.L
ells = ells.astype(np.int64)
prefactor = np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))

for i in range(len(transfer_function_data.q)):
    E_transfer[:,i] *= prefactor


# %%
# Solve for parameters
camb_results = camb.get_results(parameters)

camb_powers = camb_results.get_cmb_power_spectra(parameters, CMB_unit='K', raw_cl = True)

totCL=camb_powers['total']

lensing = camb_results.get_lens_potential_cls(5050, CMB_unit='K', raw_cl = True)

# %%

# TT Power Spectrum
TT_power_dataframe = pd.DataFrame(totCL[:,0])
TT_power_dataframe.to_csv('TT_Power_Spectrum.csv')

# EE Power Spectrum
EE_power_dataframe = pd.DataFrame(totCL[:,1])
EE_power_dataframe.to_csv('EE_Power_Spectrum.csv')

# TE Power Spectrum
TE_power_dataframe = pd.DataFrame(totCL[:,3])
TE_power_dataframe.to_csv('TE_Power_Spectrum.csv')

# TT Power Spectrum
PP_power_dataframe = pd.DataFrame(lensing[:,0])
PP_power_dataframe.to_csv('PP_Power_Spectrum.csv')

# EE Power Spectrum
PT_power_dataframe = pd.DataFrame(lensing[:,1])
PT_power_dataframe.to_csv('PT_Power_Spectrum.csv')

# TE Power Spectrum
PE_power_dataframe = pd.DataFrame(lensing[:,2])
PE_power_dataframe.to_csv('PE_Power_Spectrum.csv')

transfer_functions = {}

transfer_functions['Ells'] = ells
transfer_functions['k'] = transfer_function_data.q
transfer_functions['T'] = T_transfer
transfer_functions['E'] = E_transfer

np.save('Transfer_Functions', transfer_functions)

# T Transfer Function
T_transfer_function_dataframe = pd.DataFrame(T_transfer.T)
T_transfer_function_dataframe.to_csv('T_Transfer_Function.csv')

# E Transfer Function
E_transfer_function_dataframe = pd.DataFrame(E_transfer.T)
E_transfer_function_dataframe.to_csv('E_Transfer_Function.csv')

# ell values
ell_values_dataframe = pd.DataFrame(ells)
ell_values_dataframe.to_csv('Transfer_Function_Ell_Values.csv')

# k values
k_values_dataframe = pd.DataFrame(transfer_function_data.q)
k_values_dataframe.to_csv('Transfer_Function_k_Values.csv')


