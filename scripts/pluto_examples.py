# external imports
import sys, os, gc
import numpy as np

# local import
sys.path.append("..")
from pysrc import *

def run_pluto_convert_example():

    config_file = "/mnt/kocsis1/cuDART_wdir/jet_analyst/config.ini"
    load_dir = "/mnt/kocsis1/cuDART_wdir/jet_data/"
    save_str = "/mnt/kocsis1/cuDART_wdir/emm_data.npy"
    
    preader = PlutoReader(load_dir, 490, config_file)
    
    apply_blur = True
    blur_kwargs = {"sigma" : 2, "window" : 2}
    emm_data = preader.emm_to_npy(sparse_step = 4, 
                                    num_pfiles = 7, 
                                    apply_blur = apply_blur,
                                    blur_kwargs = blur_kwargs)
    np.save(save_str, emm_data)

if __name__ == "__main__":

    run_pluto_convert_example()