import sys
import numpy as np
sys.path.append("..")
from pysrc import *

def sum_tracer(path_to_tracer_npy, path_to_raw_images):

    # construct a Camera 
    # unlabelled datasets are assumed to span a domain [-0.5, 0.5] in xyz

    camera = Camera()
    camera.num_pixels_X = 1024  # image dimensions in X, Y
    camera.num_pixels_Y = 1024
    camera.length_X = 1.0       # physical screen dimensions in X, Y
    camera.length_Y = 1.0
    camera.set_sph_pos(r = 1.0, theta = 0.499 * np.pi, phi = 0.0001, target_origin = True)
    cameras = [camera]          # package cameras (to support multicam)

    # build container for data and cameras
    scene = Scene(path_to_tracer_npy, path_to_saved_png, cameras)

    # render and save images
    scene.render(verbose=True, relativistic = False)

if __name__ = "__main__":

    # change these paths
    path_to_tracer_npy = "/mnt/kocsis1/cuDART_wdir/emm_data/emm_1000MHz.npy"
    path_to_raw_images = "/mnt/kocsis1/cuDART_wdir/emm_img/raw" # .npy auto appended with numeric suffix
    sum_tracer(path_to_tracer_npy, path_to_raw_images)