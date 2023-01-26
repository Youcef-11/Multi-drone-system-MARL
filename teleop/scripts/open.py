#!/usr/bin/env python
import numpy as np 



if __name__ == '__main__':
    
    s = np.load("bebop2_train/data/data_simu.npy", allow_pickle=True).item()
    print(len(s["L_obs"]))