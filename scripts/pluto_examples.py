# external imports
import sys, os, gc
import numpy as np

# local import
sys.path.append("..")
from pysrc import *

def run_pluto_convert_example():

    config_file = "/mnt/kocsis1/cuDART_wdir/jet_analyst/config.ini"
    load_dir = "/mnt/kocsis1/cuDART_wdir/jet_data/"
    emm_dir = "/mnt/kocsis1/cuDART_wdir/emm_data/"

    preader = PlutoParticleReader(load_dir, config_file)
    
    apply_blur = True
    apply_mirror = True
    apply_boost = True
    blur_kwargs = {"sigma" : 2, "window" : 2}
    frequencies = ["1000MHz"]
    preader.emm_to_npy(snapshot_num = 490,
                        save_dir = emm_dir,
                        frequencies = frequencies,
                        sparse_step = 10, 
                        num_pfiles = 7, 
                        apply_blur = apply_blur,
                        apply_mirror = apply_mirror,
                        apply_boost = apply_boost,
                        blur_kwargs = blur_kwargs)

    load_str = os.path.join(emm_dir, "emm_" + frequencies[0] + ".npy")
    boosted_data = np.load(load_str)
    max_vx1 = np.max(boosted_data[...,1])
    max_vx2 = np.max(boosted_data[...,2])
    max_vx3 = np.max(boosted_data[...,3])

if __name__ == "__main__":

    run_pluto_convert_example()