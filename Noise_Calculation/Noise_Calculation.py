import numpy as np
import pandas as p

# Declare home PATH (ex: M:\Folder1\Folder2 (Windows) or /Folder1/Folder2 (Linux))
# If working on Windows, change path separators to '\' instead of '/'
PATH = r'insert_path_here'

# Import noise data
Planck_Noise = np.array(pd.read_csv(PATH + r'/Noise_Calculation/Planck_Noise.csv', header = None))
CMBS4_Initial_Noise = np.array(pd.read_csv(PATH + r'/Noise_Calculation/CMB_S4_Initial_Noise.csv', header = None))

# Generate array for the final noise
CMBS4_Final_Noise = np.zeros(Planck_Noise.shape)

# Transforming from muK to K
Planck_Noise[:,1] *= 10**(-12)
Planck_Noise[:,2] *= 10**(-12)
CMBS4_Initial_Noise[:,1] *= 10**(-12)
CMBS4_Initial_Noise[:,2] *= 10**(-12)

# Computing noise
for i in range(len(Planck_Noise[:,0])):

    # Pure Planck Noise
    CMBS4_Final_Noise[i,0] = int(Planck_Noise[i,0])
    
    # T Noise
    CMBS4_Final_Noise[i,1] = (Planck_Noise[i,1]**(-1) + CMBS4_Initial_Noise[i,1]**(-1)) ** (-1)

    # E Noise
    CMBS4_Final_Noise[i,2] = (Planck_Noise[i,2]**(-1) + CMBS4_Initial_Noise[i,2]**(-1)) ** (-1)

# Saving to csv
df = pd.DataFrame(CMBS4_Final_Noise)
df.to_csv('CMB_S4_Final_Noise.csv', header = None, index = None)
