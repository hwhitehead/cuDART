import sys, os, gc
import numpy as np

sys.path.append("..")
from pysrc import *

def build_athena_example(homogenize=False, verbose=False):

    h_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    data_dir = "/mnt/kocsis1/cuDART_wdir/athena/homo_mesh"
    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"

    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 2
    bounds = [[-l * rh, l * rh], [-l * rh, l * rh], [-l * rh, l * rh]]

    ath_data = AthenaData(h_str)
    mesh = ath_data.build_mesh(data_dir, homogenize=homogenize, bounds=bounds, tracer_type="T", level=4, nzfill=5, verbose=verbose)

    return

def render_athena_example():

    load_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    data_dir = "/mnt/kocsis1/cuDART_wdir/athena/homo_mesh"
    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/athena/output/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/athena/output/img"

    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 3

    template_camera = Camera()
    template_camera.num_pixels_X = 512
    template_camera.num_pixels_Y = 512
    template_camera.tilt = 0.0
    template_camera.length_X = 4 * rh
    template_camera.length_Y = 4 * rh

    num_img = 100
    phi = epsilon
    theta_ar = np.linspace(epsilon,np.pi - epsilon,num_img, endpoint=False)
    cameras = []
    for theta in theta_ar:
        camera = copy.deepcopy(template_camera)
        camera.set_sph_pos(r = 5 * l * rh, theta = theta, phi = phi, target_origin = True)
        cameras.append(camera)

    scene = Scene(data_dir, npy_save_str, cameras)
    scene.render(verbose=True)
    scene.plot(png_save_str, remove_raw_images = False, vmin=None, vmax=None)

if __name__ == "__main__":

    build_athena_example(homogenize=True, verbose=True)

    render_athena_example()