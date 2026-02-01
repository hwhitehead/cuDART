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
    emm_data = preader.emm_to_npy(sparse_step=10, num_pfiles=1)
    print(np.shape(emm_data))
    np.save(save_str, emm_data)

if __name__ == "__main__":

    run_pluto_convert_example()