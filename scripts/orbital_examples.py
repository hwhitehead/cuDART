import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append("..")
from pysrc import *

def build_orbital_arrays(data_dir, D = 256, half_len = 20):

    for n in range(1,5):
        for l in range(n):
            for m in range(-l, l+1):
                orbital = Orbital(n = n, l = l, m = m, data_dir = data_dir)
                orbital.save_mesh(D = D, half_len = half_len)

if __name__ == "__main__":

    data_dir = "/mnt/kocsis1/cuDART_wdir/orbitals/orbital_data"
    build_orbital_arrays(data_dir, D = 256, half_len = 20)

    # save_dir = "/mnt/kocsis1/cuDART_wdir/orbitals/orbital_plots/n3l2m0"
    # orbital = Orbital(n = 3, l = 2, m = 0, data_dir = data_dir)
    # orbital.render(save_dir, num_img=200, N=256, half_len=20, D=256, vmin=-4.5, vmax=-2.5)

    # data = np.load("/scratch/github/cuDART_wdir/orbitals/n3l2m2D256L20.npy")
    # print(np.max(data))
    # print(np.min(data))