import numpy as np 

s = np.load("data.npy", allow_pickle=True).item()

print(s["L_obs"])