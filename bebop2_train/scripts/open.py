#!/usr/bin/env python
import numpy as np 



if __name__ == '__main__':
    
    s = np.load("../data/data_simu.npy", allow_pickle=True).item()
    print(s["L_obs"])