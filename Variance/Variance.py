import numpy as np
import pandas as pd

# Generating arrays for storing fisher data
fisher_array = np.zeros((7,4999, 4))
fisher_array_sorted = np.zeros((7,4999, 4))

fisher_array_noise = np.zeros((7,4999, 4))
fisher_array_sorted_noise = np.zeros((7,4999, 4))

# Bispectra will need to be scaled by a factor of the CMB temperature due to the fact that the transfer functions were unitless
TCMB = 2.7255
TCMB_sq = TCMB**2

# Declare home PATH (ex: M:\Folder1\Folder2 (Windows) or /Folder1/Folder2 (Linux))
# If working on Windows, change path separators to '\' instead of '/'
PATH = r'insert_path_here'


# Importing fisher data
fisher_array[0] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_local+local.csv', header = None))
fisher_array[1] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_equilateral+equilateral.csv', header = None))
fisher_array[2] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_orthogonal+orthogonal.csv', header = None))
fisher_array[3] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_local+ISW.csv', header = None))
fisher_array[4] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_equilateral+ISW.csv', header = None))
fisher_array[5] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_orthogonal+ISW.csv', header = None))
fisher_array[6] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_ISW+ISW.csv', header = None))

fisher_array_noise[0] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_local+local.csv', header = None))
fisher_array_noise[1] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_equilateral+equilateral.csv', header = None))
fisher_array_noise[2] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_orthogonal+orthogonal.csv', header = None))
fisher_array_noise[3] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_local+ISW.csv', header = None))
fisher_array_noise[4] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_equilateral+ISW.csv', header = None))
fisher_array_noise[5] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_orthogonal+ISW.csv', header = None))
fisher_array_noise[6] = np.array(pd.read_csv(PATH + r'Compute_Fisher/Fisher_Elements_Noise_ISW+ISW.csv', header = None))

# Scaling by the temperature of the CMB
fisher_array[0,:,1:] *= (TCMB_sq)**3
fisher_array[1,:,1:] *= (TCMB_sq)**3
fisher_array[2,:,1:] *= (TCMB_sq)**3
fisher_array[3,:,1:] *= (TCMB_sq)**(3/2)
fisher_array[4,:,1:] *= (TCMB_sq)**(3/2)
fisher_array[5,:,1:] *= (TCMB_sq)**(3/2)

fisher_array_noise[0,:,1:] *= (TCMB_sq)**3
fisher_array_noise[1,:,1:] *= (TCMB_sq)**3
fisher_array_noise[2,:,1:] *= (TCMB_sq)**3
fisher_array_noise[3,:,1:] *= (TCMB_sq)**(3/2)
fisher_array_noise[4,:,1:] *= (TCMB_sq)**(3/2)
fisher_array_noise[5,:,1:] *= (TCMB_sq)**(3/2)


# Sorting the fisher arrays
for i in range(7):

    temp_array = fisher_array[i]
    fisher_array_sorted[i] = temp_array[temp_array[:,0].argsort(kind = 'mergesort')]

    temp_array = fisher_array_noise[i]
    fisher_array_sorted_noise[i] = temp_array[temp_array[:,0].argsort(kind = 'mergesort')]
    

# Calculate full Fisher element
max_index = 4999

# Creating array for storing fisher and variance values
Fisher = np.zeros((7,3))
Variance = np.zeros(3)
Fisher_noise = np.zeros((7,3))
Variance_noise = np.zeros(3)

# Looping over the data
for i in range(7):

    for j in range(max_index):

        for k in range(1,4):

            Fisher[i,k-1] += fisher_array_sorted[i,j,k]
            Fisher_noise[i,k-1] += fisher_array_sorted_noise[i,j,k]

#Calculating the variance in the absence of ISW-Lensing
for q in range(3):

    Variance[q] = 1 / np.sqrt(Fisher[q,2])
    Variance_noise[q] = 1 / np.sqrt(Fisher_noise[q,2])

print('The variance without noise and without ISW-Lensing for local, equilateral, and orthogonal shapes is: ', Variance)
print('The variance with noise and without ISW-Lensing for local, equilateral, and orthogonal shapes is: ', Variance_noise)


# Marginalization
# Creating Fisher Matrices
F_local_ISW = np.array([[Fisher[0,2], Fisher[3,2]], [Fisher[3,2], Fisher[6,2]]])
F_equilateral_ISW = np.array([[Fisher[1,2], Fisher[4,2]], [Fisher[4,2], Fisher[6,2]]])
F_orthogonal_ISW = np.array([[Fisher[2,2], Fisher[5,2]], [Fisher[5,2], Fisher[6,2]]])

F_local_ISW_noise = np.array([[Fisher_noise[0,2], Fisher_noise[3,2]], [Fisher_noise[3,2], Fisher_noise[6,2]]])
F_equilateral_ISW_noise = np.array([[Fisher_noise[1,2], Fisher_noise[4,2]], [Fisher_noise[4,2], Fisher_noise[6,2]]])
F_orthogonal_ISW_noise = np.array([[Fisher_noise[2,2], Fisher_noise[5,2]], [Fisher_noise[5,2], Fisher_noise[6,2]]])

# Creating arrays for variance values
Variance_lensed = np.zeros(3)
Variance_lensed_noise = np.zeros(3)

# Inverting Fisher Matrices
Variance_lensed[0] = np.sqrt(np.linalg.inv(F_local_ISW)[0, 0])
Variance_lensed[1] = np.sqrt(np.linalg.inv(F_equilateral_ISW)[0, 0])
Variance_lensed[2] = np.sqrt(np.linalg.inv(F_orthogonal_ISW)[0, 0])

Variance_lensed_noise[0] = np.sqrt(np.linalg.inv(F_local_ISW_noise)[0, 0])
Variance_lensed_noise[1] = np.sqrt(np.linalg.inv(F_equilateral_ISW_noise)[0, 0])
Variance_lensed_noise[2] = np.sqrt(np.linalg.inv(F_orthogonal_ISW_noise)[0, 0])

print('The variance without noise and with ISW-Lensing for local, equilateral, and orthogonal shapes is: ', Variance_lensed)
print('The variance with noise and with ISW-Lensing for local, equilateral, and orthogonal shapes is: ', Variance_lensed_noise)

# Calculating the bias
print('Without noise, the ISW-Lensing bias is ', F_local_ISW[0,1] / F_local_ISW[0,0], ' for the local shape, ', F_equilateral_ISW[0,1] / F_equilateral_ISW[0,0], ' for the equilateral shape, and ', F_orthogonal_ISW[0,1] / F_orthogonal_ISW[0,0], ' for the orthogonal shape')
print('With noise, the ISW-Lensing bias is ', F_local_ISW_noise[0,1] / F_local_ISW_noise[0,0], ' for the local shape, ', F_equilateral_ISW_noise[0,1] / F_equilateral_ISW_noise[0,0], ' for the equilateral shape, and ', F_orthogonal_ISW_noise[0,1] / F_orthogonal_ISW_noise[0,0], ' for the orthogonal shape')
