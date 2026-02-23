import sys, os, gc
import numpy as np

sys.path.append("..")
from pysrc import *

def build_inhomo_athena_example():

    h_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    data_dir = "/mnt/kocsis1/cuDART_wdir/athena/mesh"
    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"

    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 3
    bounds = [[-l * rh, l * rh], [-l * rh, l * rh], [-l * rh, l * rh]]

    ath_data = AthenaData(h_str)
    mesh = ath_data.build_mesh(data_dir, homogenize=False, bounds=bounds, tracer_type="prs", level=2)

    return

def render_athena_example():

    load_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    data_dir = "/mnt/kocsis1/cuDART_wdir/athena/mesh"
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
    template_camera.length_X = 2 * rh
    template_camera.length_Y = 2 * rh
    template_camera.set_sph_pos(r=5 * l * rh, phi = 0.0001, theta = 0.4999 * np.pi)
    cameras = [template_camera]

    scene = Scene(data_dir, npy_save_str, cameras)
    scene.render()
    scene.plot(png_save_str, remove_raw_images = False)

if __name__ == "__main__":

    build_inhomo_athena_example()

    render_athena_example()