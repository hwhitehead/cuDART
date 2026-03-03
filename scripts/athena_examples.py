import sys, os, gc
import numpy as np
from matplotlib import cm

sys.path.append("..")
from pysrc import *

def build_athena_example(homogenize=False, verbose=False, level=4):

    h_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    data_dir = "/mnt/kocsis1/cuDART_wdir/athena/inhomo_rho_mesh"
    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"

    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 2
    bounds = [[-l * rh, l * rh], [-l * rh, l * rh], [-l * rh, l * rh]]

    ath_data = AthenaData(h_str)
    mesh = ath_data.build_mesh(data_dir, homogenize=homogenize, bounds=bounds, tracer_type="rho", level=level, nzfill=5, verbose=verbose)

    return

def render_athena_example():

    load_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    data_dir = "/mnt/kocsis1/cuDART_wdir/athena/inhomo_rho_mesh"
    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/athena/inhomo_rho_output/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/athena/inhomo_rho_output/img"

    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 3

    template_camera = Camera()
    template_camera.num_pixels_X = 512
    template_camera.num_pixels_Y = 512
    template_camera.tilt = 0.0
    template_camera.length_X = 4 * rh
    template_camera.length_Y = 4 * rh

    num_img = 300
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

def remap(input, inp_min, inp_max):

    input[input < inp_min] = inp_min
    input[input > inp_max] = inp_max
    input = (input - inp_min) / (inp_max - inp_min)

    return input

def composite_plot(rho_str, vz_str, save_str):

    rho_data = np.load(rho_str).T
    vz_data = np.load(vz_str).T
    
    rho_grey = remap(np.log10(rho_data), -3, -1.5)
    vz_grey = remap(np.log10(vz_data), -4.4, -2)
    # vz_grey[vz_grey > 0.5] = 0.5
    # vz_grey[vz_grey < 0.2] = 0.0

    rho_cmap = plt.get_cmap("afmhot")
    vz_cmap = plt.get_cmap("Blues")
    rho_RGBA = rho_cmap(rho_grey)
    vz_RGBA = vz_cmap(vz_grey)
    rho_RGBA[...,3] = 0.6
    vz_RGBA[...,3] = 0.5 * vz_grey ** 2
    # vz_RGBA[...,3][vz_RGBA[...,3] < 0.05] = 0.0

    set_plot_defaults(use_tex=True)
    fig = plt.figure(figsize=(10.0/3, 10.0/3))
    ax = fig.add_subplot()
    ax.set_facecolor("k")
    ax.imshow(rho_RGBA)
    ax.imshow(vz_RGBA)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig(save_str, dpi=300, bbox_inches="tight")
    plt.close("all")

def loop_composites(num_img=300):

    rho_dir = "/scratch/thesis/jets/cudart_renders/rho_hr"
    vz_dir = "/scratch/thesis/jets/cudart_renders/vz_hr"
    save_dir = "/scratch/thesis/jets/cudart_renders/comp"

    for n in range(100,num_img):
        rho_str = os.path.join(rho_dir, "raw" + str(n).zfill(5) + ".npy")
        vz_str = os.path.join(vz_dir, "raw" + str(n).zfill(5) + ".npy")
        save_str = os.path.join(save_dir, str(n).zfill(5) + ".png")
        composite_plot(rho_str, vz_str, save_str)

if __name__ == "__main__":

    build_athena_example(homogenize=False)
    render_athena_example()
    #loop_composites()