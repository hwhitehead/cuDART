import sys, os, gc
import numpy as np
from matplotlib import cm

sys.path.append("..")
from pysrc import *

def build_athena_example(header, homogenize=True, verbose=False, level=5, tracer_type="rho"):

    h_str = "/mnt/kocsis1/cuDART_wdir/athena/raw_data/nshear.out1.00060.athdf"
    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"

    data_dir = os.path.join("/mnt/kocsis1/cuDART_wdir/athena", header + "_mesh")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 2
    bounds = [[-l * rh, l * rh], [-l * rh, l * rh], [-l * rh, l * rh]]

    ath_data = AthenaData(h_str)
    mesh = ath_data.build_mesh(data_dir, homogenize=homogenize, bounds=bounds, tracer_type=tracer_type, homo_level=level, nzfill=5, verbose=verbose)

    return

def render_athena_example(header, prof_file = None):

    bh_npy_str = "/mnt/kocsis1/cuDART_wdir/athena/bh_data.npy"

    data_dir = os.path.join("/mnt/kocsis1/cuDART_wdir/athena", header + "_mesh")
    out_dir = os.path.join("/mnt/kocsis1/cuDART_wdir/athena", header + "_output")
    npy_save_str = os.path.join(out_dir, "raw")
    png_save_str = os.path.join(out_dir, "img")

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    bh = BlackHole(0, bh_npy_str)
    rh = np.cbrt(bh.m[0] / (3 * bh.Omega0 ** 2))
    l = 2

    template_camera = Camera()
    template_camera.num_pixels_X = 1024
    template_camera.num_pixels_Y = 1024
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
    scene.render(verbose=True, save_profile = prof_file)
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
    vz_grey = remap(np.log10(vz_data), -4.5, -2)

    rho_cmap = plt.get_cmap("afmhot")
    vz_cmap = plt.get_cmap("Blues")
    rho_RGBA = rho_cmap(rho_grey)
    vz_RGBA = vz_cmap(vz_grey)
    rho_RGBA[...,3] = 0.6
    vz_RGBA[...,3] = 0.5 * vz_grey ** 2

    set_plot_defaults(use_tex=True)
    width_ratios = np.array([0.05, 1, 0.05])
    height_ratios = np.array([1])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    fig = plt.figure(figsize=(10.0 / 3, h_over_w * 10.0 / 3))
    gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios)
    caxl = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0,1])
    caxr = fig.add_subplot(gs[0,2])

    ax.set_facecolor("k")
    ax.set_aspect("equal")
    ax.imshow(rho_RGBA)
    ax.imshow(vz_RGBA)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    rho_sm = plt.cm.ScalarMappable(cmap="afmhot", norm=plt.Normalize(vmin=-3, vmax=-1.5))
    vz_sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=-4.5, vmax=-2))
    fig.colorbar(rho_sm, cax=caxl, orientation="vertical")
    fig.colorbar(vz_sm, cax=caxr, orientation="vertical")
    caxl.yaxis.tick_left()
    caxl.yaxis.set_label_position("left")
    caxl.set_ylabel(r"$\int \rho ds$ [arb.]")
    caxr.set_ylabel(r"$\int \left|v_z\right| ds$ [arb.]")

    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(save_str, dpi=300, bbox_inches="tight")
    plt.close("all")

def loop_composites(num_img=300):

    rho_dir = "/scratch/thesis/jets/cudart_renders/inhomo_rho_output"
    vz_dir = "/scratch/thesis/jets/cudart_renders/inhomo_vz_output"
    save_dir = "/scratch/thesis/jets/cudart_renders/comp_label"

    for n in range(0,num_img, 1):
        rho_str = os.path.join(rho_dir, "raw" + str(n).zfill(5) + ".npy")
        vz_str = os.path.join(vz_dir, "raw" + str(n).zfill(5) + ".npy")
        save_str = os.path.join(save_dir, str(n).zfill(5) + ".png")
        composite_plot(rho_str, vz_str, save_str)

if __name__ == "__main__":

    # build_athena_example(header="homo_rho", homogenize=True)
    #build_athena_example(header="inhomo_vz", homogenize=False, tracer_type="vel_z")
    # render_athena_example(header="homo_rho", prof_file="/mnt/kocsis1/cuDART_wdir/athena/homo_prof.txt")
    #render_athena_example(header="inhomo_vz", prof_file="/mnt/kocsis1/cuDART_wdir/athena/inhomo_vz_prof.txt")
    loop_composites()



