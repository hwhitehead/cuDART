# generic imports
import numpy as np
import matplotlib.pyplot as plt
import subprocess, os, sys, copy

# local import
sys.path.append("..")
from pysrc import *

def demo_scene_gen(num_img = 200, verbose=True, remove_data=True):

    # define targets
    npy_load_str = os.path.join(host_dir, "inputs/sn_alt.npy")
    npy_save_str = os.path.join(host_dir, "outputs/auto/sn")
    png_save_str = os.path.join(host_dir, "outputs/auto/sn")

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = (-38.0/180) * np.pi
    template_camera.length_X = 0.66
    template_camera.length_Y = 0.66

    # build camera array, inherit from template
    theta_ar = np.linspace(0,np.pi,num_img)
    cameras = []
    for theta in theta_ar:
        camera = copy.deepcopy(template_camera)
        camera.theta = theta
        cameras.append(camera)

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras)

    # render and save images
    scene.render(verbose=verbose, max_mem=0.5)
    scene.plot(png_save_str, verbose=verbose, remove_data=True)

if __name__ == "__main__":

    demo_scene_gen()