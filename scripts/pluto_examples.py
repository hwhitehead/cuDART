# external imports
import sys, os, gc
import numpy as np

# local import
sys.path.append("..")
from pysrc import *

def run_pluto_convert_example():

    config_file = "/mnt/kocsis1/cuDART_wdir/jet_analyst/config.ini"
    load_dir = "/mnt/kocsis1/cuDART_wdir/jet_data/"
    save_dir = "/mnt/kocsis1/cuDART_wdir/emm_data/"
    
    preader = PlutoParticleReader(load_dir, config_file)
    
    apply_blur = True
    blur_kwargs = {"sigma" : 2, "window" : 2}
    frequencies = ["1000MHz"]
    preader.emm_to_npy(snapshot_num = 490,
                        frequencies = frequencies,
                        sparse_step = 2, 
                        num_pfiles = 7, 
                        apply_blur = apply_blur,
                        blur_kwargs = blur_kwargs)

    for frequency in frequencies:
        save_str = os.path.join(save_dir, "emm_" + frequency + ".npy")
        np.save(save_str, preader.emm_npy_data[frequency])

if __name__ == "__main__":

    run_pluto_convert_example()