# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
Planck_Noise = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Noise_Calculation\Planck_Noise.csv', header = None))
CMBS4_Initial_Noise = np.array(pd.read_csv(r'M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Noise_Calculation\CMB_S4_Initial_Noise.csv', header = None))

# %%
CMBS4_Final_Noise = np.zeros(Planck_Noise.shape)

Planck_Noise[:,1] *= 10**(-12)
Planck_Noise[:,2] *= 10**(-12)
CMBS4_Initial_Noise[:,1] *= 10**(-12)
CMBS4_Initial_Noise[:,2] *= 10**(-12)

for i in range(len(Planck_Noise[:,0])):

    CMBS4_Final_Noise[i,0] = int(Planck_Noise[i,0])
    CMBS4_Final_Noise[i,1] = (Planck_Noise[i,1]**(-1) + CMBS4_Initial_Noise[i,1]**(-1)) ** (-1)
    CMBS4_Final_Noise[i,2] = (Planck_Noise[i,2]**(-1) + CMBS4_Initial_Noise[i,2]**(-1)) ** (-1)

df = pd.DataFrame(CMBS4_Final_Noise)
df.to_csv('CMB_S4_Final_Noise.csv', header = None, index = None)


# %%
fig = plt.figure(figsize=(10,6))

plt.plot(CMBS4_Final_Noise[:,0], CMBS4_Final_Noise[:,1], color = 'blue', alpha = 0.6)
plt.plot(CMBS4_Final_Noise[:,0], Planck_Noise[:,1], color = 'red', alpha = 0.6)
plt.plot(CMBS4_Final_Noise[20:,0], CMBS4_Initial_Noise[20:,1], color = 'green', alpha = 0.6)
plt.grid()
plt.show()


