# external imports
import sys, os, gc
import numpy as np
import matplotlib.image as mpimg
import pandas as pd

# local import
sys.path.append("..")
from pysrc import *

epsilon = 1e-2 # small number to avoid casts with exact cooordinate alignment

def extract_pluto_data_example():

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
                        sparse_step = 2, 
                        num_pfiles = 7, 
                        apply_blur = apply_blur,
                        apply_mirror = apply_mirror,
                        apply_boost = apply_boost,
                        blur_kwargs = blur_kwargs)

def render_pluto_helix(relativistic = False):

    print("cuDART: starting jet render example...")

    # define targets
    npy_load_str = "/mnt/kocsis1/cuDART_wdir/emm_data/emm_1000MHz.npy"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/emm_img/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/emm_img/img"

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 1024
    template_camera.num_pixels_Y = 1024
    template_camera.tilt = (0.0 / 180) * np.pi
    template_camera.length_X = 0.1
    template_camera.length_Y = 0.1

    # build camera array, inherit from template
    num_img = 300
    num_checkpoints = 32
    z_vals = np.linspace(-0.45, 0.45, num_checkpoints)
    thetas = [0.5 * np.pi * x for x in range(num_checkpoints)]
    radius = 0.5
    x_vals = radius * np.cos(thetas)
    y_vals = radius * np.sin(thetas)
    checkpoints = np.zeros(shape=(num_checkpoints, 3))
    checkpoints[:, 0] = x_vals
    checkpoints[:, 1] = y_vals
    checkpoints[:, 2] = z_vals
    target = np.array([0,0,0])
    gcam = GuidedCamera(checkpoints = checkpoints, targets = target)

    camera_times = np.linspace(0, 1, num_img)
    cameras = gcam.generate_cameras(template_camera = template_camera, num_img = num_img, camera_times = camera_times, mode = "chord")
    for i in range(num_img):
        this_origin = cameras[i].origin
        this_target = np.array([0,0,this_origin[2]])
        cameras[i].set_target(this_target)
    
    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras)
    print("built scene")

    # render and save images
    scene.render(verbose = True, relativistic = relativistic)
    print("finished rendering raw images")

    scene.plot(png_save_str, cmap = "afmhot", verbose = True, remove_raw_images = True, vmin=18, vmax=21)
    print("finished rendering rasterised images")

    print("unlablled render example finished.")

def render_pluto_data_example(relativistic=False, remove_raw_images = True, profile = False):

    print("cuDART: starting jet render example...")

    # define targets
    npy_load_str = "/mnt/kocsis1/cuDART_wdir/emm_data/emm_1000MHz.npy"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/emm_img/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/emm_img/img"

    # build mesh
    # if os.path.isdir(npy_load_str):
    #     xls = [[-0.25,-0.25,-0.5], [-0.5,-0.25,-0.25], [-0.25,-0.5,-0.25]]
    #     xrs = [[0.25,0.25,0.5], [0.5,0.25,0.25], [0.25,0.5,0.25]]
    #     mesh = Mesh(npy_load_str)
    #     for i, sub_str in enumerate(["", "_x", "_y"]):
    #         npy_str = os.path.join(npy_load_str, "emm_1000MHz" + sub_str + ".npy")
    #         mb_data = np.load(npy_str)
    #         mesh.add_meshblock(mb_data, xls[i], xrs[i])
    #     mesh.write_header()

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = (60.0 / 180) * np.pi
    template_camera.length_X = 0.25 # defval 0.66
    template_camera.length_Y = 0.25

    # build camera array, inherit from template
    num_img = 100
    phi = epsilon
    theta_ar = np.linspace(epsilon,np.pi - epsilon,num_img, endpoint=False)
    cameras = []
    for theta in theta_ar:
        camera = copy.deepcopy(template_camera)
        camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)
        cameras.append(camera)
    print("initialised cameras")

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras)
    print("built scene")

    # render and save images
    scene.render(verbose = True, relativistic = relativistic, profile = profile)
    print("finished rendering raw images")

    scene.plot(png_save_str, cmap = "afmhot", verbose = True, remove_raw_images = remove_raw_images, vmin=18, vmax=21)
    print("finished rendering rasterised images")

    print("unlablled render example finished.")

def save_alt(axes="x"):

    load_str = "/mnt/kocsis1/cuDART_wdir/emm_data/emm_1000MHz.npy"
    input_data = np.load(load_str)
    input_shape = np.shape(input_data)
    
    if axes == "x":
        output_shape = np.array([input_shape[2], input_shape[1], input_shape[0], 4])
        output_data = np.zeros(shape=output_shape, dtype=np.float32)
        output_data[:, :, :, 0] = np.einsum("ijk->kji", input_data[:, :, :, 0])
        output_data[:, :, :, 1] = np.einsum("ijk->kji", input_data[:, :, :, 3])
        output_data[:, :, :, 2] = np.einsum("ijk->kji", input_data[:, :, :, 2])
        output_data[:, :, :, 3] = np.einsum("ijk->kji", input_data[:, :, :, 1])
        save_str = load_str[:-4] + "_x.npy"
        np.save(save_str, output_data.astype(np.float32))
    else:
        output_shape = np.array([input_shape[0], input_shape[2], input_shape[1], 4])
        output_data = np.zeros(shape=output_shape, dtype=np.float32)
        output_data[:, :, :, 0] = np.einsum("ijk->ikj", input_data[:, :, :, 0])
        output_data[:, :, :, 1] = np.einsum("ijk->ikj", input_data[:, :, :, 1])
        output_data[:, :, :, 2] = np.einsum("ijk->ikj", input_data[:, :, :, 3])
        output_data[:, :, :, 3] = np.einsum("ijk->ikj", input_data[:, :, :, 2])
        save_str = load_str[:-4] + "_y.npy"
        np.save(save_str, output_data.astype(np.float32))

def save_lcs():

    boosted_dir = "/mnt/kocsis1/cuDART_wdir/emm_img/boosted_raws"
    unboosted_dir = "/mnt/kocsis1/cuDART_wdir/emm_img/unboosted_raws"
    num_img = 200

    for loc in [boosted_dir, unboosted_dir]:
        lum_ar = np.zeros(shape=(num_img))
        save_str = os.path.join(loc, "lum.npy")
        for n in range(num_img):
            raw_str = os.path.join(loc, "raw" + str(n).zfill(3) + ".npy")
            raw_img = np.load(raw_str)
            lum_ar[n] = np.sum(raw_img)
        np.save(save_str, lum_ar)

def comp_plot():

    vmin = 18
    vmax = 21
    cmap = "afmhot"

    boosted_dir = "/mnt/kocsis1/cuDART_wdir/emm_img/boosted_raws"
    unboosted_dir = "/mnt/kocsis1/cuDART_wdir/emm_img/unboosted_raws"
    save_dir = "/mnt/kocsis1/cuDART_wdir/emm_img"
    num_img = 200
    pix_dims = np.array([2048, 2048])
    X = np.linspace(0,1,pix_dims[0]+1)
    Y = np.linspace(0,1,pix_dims[1]+1)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    tax_span = np.linspace(0, 1, num_img)

    img_str = "/mnt/kocsis1/cuDART_wdir/agn.jpeg"
    example_img = mpimg.imread(img_str)
    inset_aspect = np.shape(example_img)[0] / np.shape(example_img)[1]
    inset_size = 0.33
    inset_border = 0.05

    set_plot_defaults()
    L = 20.0 / 3
    height_ratios = np.array([0.3, 1])
    width_ratios = np.array([0.05, 1,1])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    fig = plt.figure(figsize=(L, L * h_over_w))
    gs = fig.add_gridspec(2, 3, height_ratios = height_ratios,width_ratios = width_ratios)
    axl = fig.add_subplot(gs[1,1])
    axr = fig.add_subplot(gs[1,2])
    axs = [axl, axr]
    tax = fig.add_subplot(gs[0,:])
    cax = fig.add_subplot(gs[1,0])
    inset_ax = axr.inset_axes([1 - inset_size - inset_border, 
                                1 - inset_size * inset_aspect - inset_border, 
                                inset_size, 
                                inset_size * inset_aspect])
    plt.subplots_adjust(hspace=0.1, wspace=0)

    inset_ax.imshow(example_img)
    inset_ax.xaxis.set_ticks([])
    inset_ax.xaxis.set_ticks([])
    for spine in ["bottom", "top", "right", "left"]:
        inset_ax.spines[spine].set_color("w")

    # plot colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, orientation="vertical")
    cax.set_ylabel(r"$\log_{10}\left(I_{\nu}\right)$")
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position("left")

    tax.set_xlim([0,1])
    tax.xaxis.tick_top()
    tax.xaxis.set_label_position("top")
    tax.set_xlabel(r"$\theta$ / $\pi$")
    tax.set_ylabel(r"$\left(L_\nu - \bar{L}_\nu \right)/ \bar{L}_\nu$")
    axl.plot([],[],color='w', label="Unboosted", linestyle="dashed")
    axr.plot([],[],color='w', label="Boosted", linestyle="solid")
    axl.legend(loc="upper left", frameon=False, labelcolor="linecolor")
    axr.legend(loc="upper left", frameon=False, labelcolor="linecolor")
    for ax in [axl, axr]:
        ax.set_facecolor("k")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    

    line_styles = ["dashed", "solid"]
    for i, loc in enumerate([unboosted_dir, boosted_dir]):
        lum_data = np.load(os.path.join(loc, "lum.npy"))
        if i == 0:
            lum_mean = np.mean(lum_data)
        tax.plot(tax_span, (lum_data - lum_mean) / lum_mean, color='k', linestyle=line_styles[i])
    
    for n in range(num_img):

        tstamp = tax.axvline(x=tax_span[n], color='k', alpha=0.2)

        unboosted_raw_str = os.path.join(unboosted_dir, "raw" + str(n).zfill(3) + ".npy")
        unboosted_img = np.load(unboosted_raw_str)
        pcl = axl.pcolormesh(XX, YY, np.log10(unboosted_img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")

        boosted_raw_str = os.path.join(boosted_dir, "raw" + str(n).zfill(3) + ".npy")
        boosted_img = np.load(boosted_raw_str)
        pcr = axr.pcolormesh(XX, YY, np.log10(boosted_img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")

        save_str = os.path.join(save_dir, "img" + str(n).zfill(3) + ".png")
        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        tstamp.remove()
        pcl.remove()
        pcr.remove()

    plt.close("all")

def run_profiler():

    image_dims = [64, 128, 256, 512]
    data_dims = [64, 128, 256, 512]

    data_dir = "/mnt/kocsis1/cuDART_wdir/prof_data"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/emm_img/raw"

    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = 0.0
    template_camera.length_X = 1.0
    template_camera.length_Y = 1.0
    template_camera.set_sph_pos(r = 2.0, theta = (1 - epsilon) * np.pi, phi = epsilon, target_origin = True)

    for i, image_dim in enumerate(image_dims):
        for j, data_dim in enumerate(data_dims):
            for label, relativistic in zip(["unboosted_", "boosted_"], [False, True]):
                npy_load_str = os.path.join(data_dir, label + str(data_dim) + ".npy")

                camera = copy.deepcopy(template_camera)
                camera.num_pixels_X = image_dim
                camera.num_pixels_Y = image_dim
                cameras = [camera] * 100

                scene = Scene(npy_load_str, npy_save_str, cameras)
                save_profile = os.path.join(data_dir, "profiles/profile_N{0}D{1}b{2}.txt".format(image_dim, data_dim, relativistic))
                scene.render(verbose = False, relativistic = relativistic, save_profile = save_profile)

                print("finished N = " + str(image_dim) + ", D = " + str(data_dim) + " relativistic = " + str(relativistic))
                print("\n\n\n\n\n")

def plot_old_profiler_results():

    set_plot_defaults()
    fig = plt.figure(figsize=(10.0 / 3, 10.0 / 3))
    ax = fig.add_subplot()
    axr = ax.twinx()

    image_dims = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    log10_image_dims = np.log10(image_dims)
    unboosted_ms = np.array([0.890, 1.24, 4.53, 8.31, 13.5, 28.7, 111, 277])
    boosted_ms = np.array([2.26, 2.28, 6.14, 9.02, 35.8, 137, 408, 1460])

    unboosted_mp_ps = 1e-6 * image_dims ** 2 / (1e-3 * unboosted_ms)
    boosted_mp_ps = 1e-6 * image_dims ** 2 / (1e-3 * boosted_ms)

    ax.set_xlabel(r"$\log_{10}(N)$")
    ax.set_ylabel(r"$\log_{10}(\tau \left[\mathrm{ms}\right])$")
    ax.plot(log10_image_dims, np.log10(unboosted_ms), color='k')
    ax.plot(log10_image_dims, np.log10(boosted_ms), color='r')

    axr.set_ylabel(r"$\log_{10}(\frac{N^2}{\tau} \left[\mathrm{MPs}^{-1}\right])$")
    axr.plot(log10_image_dims, np.log10(unboosted_mp_ps), color='k', linestyle="dashed")
    axr.plot(log10_image_dims, np.log10(boosted_mp_ps), color='r', linestyle="dashed")

    fig.savefig("profile.png", dpi=300, bbox_inches="tight")

def plot_profiler_results():

    data_dims = np.array([64, 128, 245, 512])
    image_dims = np.array([64, 128, 245, 512])
    log_data_dims = np.log10(data_dims)
    log_image_dims = np.log10(image_dims)
    n64_ub = [0.195, 0.327, 0.579, 1.08]
    n64_b = [0.53, 1.01, 1.91, 3.46]
    n128_ub = [0.122, 0.311, 0.597, 1.13]
    n128_b = [0.536, 1.01, 1.90, 3.47]
    n256_ub = [0.148, 0.305, 6.44, 13.9]
    n256_b = [0.541, 1.02, 9.49, 21.5]
    n512_ub = [0.431, 0.868, 2.02, 52.5]
    n512_b = [1.96, 3.81, 7.41, 78.3]

    all_timings = np.zeros(shape=(4,4,2)) # N, D, ub/b
    all_timings[0, :, 0] = n64_ub
    all_timings[0, :, 1] = n64_b
    all_timings[1, :, 0] = n128_ub
    all_timings[1, :, 1] = n128_b
    all_timings[2, :, 0] = n256_ub
    all_timings[2, :, 1] = n256_b
    all_timings[3, :, 0] = n512_ub
    all_timings[3, :, 1] = n512_b

    ub_timings = [n64_ub, n128_ub, n256_ub, n512_ub]
    b_timings = [n64_b, n128_b, n256_b, n512_b]

    set_plot_defaults()
    height_ratios = np.array([1.0, 1.0])
    width_ratios = np.array([2.0])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    L = 20.0 / 3
    fig = plt.figure(figsize=(L, L * h_over_w))
    gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])

    ax0.set_xlabel(r"$\log_{10}(D)$")
    ax0.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
    for N_index in range(4):
        ax0.plot(log_data_dims, np.log10(all_timings[N_index, :, 0]), linestyle="solid")
        ax0.plot(log_data_dims, np.log10(all_timings[N_index, :, 1]), linestyle="dashed")

    ax1.set_xlabel(r"$\log_{10}(N)$")
    ax1.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
    for D_index in range(4):
        ax1.plot(log_image_dims, np.log10(all_timings[:, D_index, 0]), linestyle="solid")
        ax1.plot(log_image_dims, np.log10(all_timings[:, D_index, 1]), linestyle="dashed")

    fig.savefig("/scratch/github/cuDART_wdir/profile.png", dpi=300, bbox_inches="tight")

def extract_profiler_times():

    profile_dir = "/scratch/github/cuDART_wdir/profiles"
    
    data_dims = np.array([64, 128, 256, 512])
    image_dims = np.array([64, 128, 256, 512])
    dd, ii = np.meshgrid(data_dims, image_dims, indexing="ij")
    log_data_dims = np.log10(data_dims)
    log_image_dims = np.log10(image_dims)

    avg_times = np.zeros(shape=(4, 4, 2))

    for i, image_dim in enumerate(image_dims):
        for j, data_dim in enumerate(data_dims):
            for label, relativistic in zip(["unboosted_", "boosted_"], [False, True]):
                load_str = os.path.join(profile_dir, "profile_N" + str(image_dim) + "D" + str(data_dim) + "b" + str(relativistic) + ".txt")
                render_time = time_from_prof(load_str)
                if relativistic:
                    avg_times[i,j,1] = render_time
                else:
                    avg_times[i,j,0] = render_time

    ub_mp_ps = 1e-6 * ii ** 2 / avg_times[:,:,0]
    b_mp_ps = 1e-6 * ii ** 2 / avg_times[:,:,1]

    avg_times *= 1e3

    set_plot_defaults()
    height_ratios = np.array([1.0, 1.0])
    width_ratios = np.array([2.0])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    L = 20.0 / 3
    fig = plt.figure(figsize=(L, L * h_over_w))
    gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])

    ax0.set_title(r"Render Image $N^2$ from domain $D^3$ with $N$, $D \in \{64,128,256,512\}$")

    ax1.set_xlabel(r"$\log_{10}(D)$")
    ax1.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
    colors = ["r", "g", "b", "k"]
    labels = ["N = {0}".format(x) for x in image_dims]
    for i, label in enumerate(labels):
        ax1.plot([],[],label=label, color=colors[i])
    ax1.legend(loc="upper left", frameon=False)
    for N_index in range(4):
        ax1.plot(log_data_dims, np.log10(avg_times[N_index, :, 0]), linestyle="solid", color=colors[N_index])
        ax1.plot(log_data_dims, np.log10(avg_times[N_index, :, 1]), linestyle="dashed", color=colors[N_index])

    ax0.set_xlabel(r"$\log_{10}(N)$")
    ax0.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
    colors = ["r", "g", "b", "k"]
    labels = ["D = {0}".format(x) for x in data_dims]
    for i, label in enumerate(labels):
        ax0.plot([],[],label=label, color=colors[i])
    ax0.legend(loc="upper left", frameon=False)
    for D_index in range(4):
        ax0.plot(log_image_dims, np.log10(avg_times[:, D_index, 0]), linestyle="solid", color=colors[D_index])
        ax0.plot(log_image_dims, np.log10(avg_times[:, D_index, 1]), linestyle="dashed", color=colors[D_index])

    # ax0.set_xlabel(r"$C \equiv \log_{10}(N^2D)$")
    # ax0.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
    # ax0.scatter([],[], marker="o", edgecolors="k", facecolors="none", label="Unboosted")
    # ax0.scatter([],[], marker="o", edgecolors="r", facecolors="none", label="Boosted")
    # ax0.legend(loc="upper left", frameon=False)
    # for D_index in range(4):
    #     ax0.scatter(np.log10(image_dims ** 2), np.log10(avg_times[D_index, :, 0]), marker="o", edgecolors="k", facecolors="none")
    #     ax0.scatter(np.log10(image_dims ** 2), np.log10(avg_times[D_index, :, 1]), marker="o", edgecolors="r", facecolors="none")
    #     ax0.scatter(np.log10(image_dims ** 2), np.log10(avg_times[D_index, :, 0]), marker="o", edgecolors="k", facecolors="none")
        # ax0.scatter(np.log10(image_dims ** 2), np.log10(avg_times[D_index, :, 1]), marker="o", edgecolors="r", facecolors="none")

    # ax1.set_xlabel(r"$C \equiv \log_{10}(N^2D)$")
    # ax1.set_ylabel(r"$\log_{10}(\mathrm{MP}/\mathrm{s})$")
    # for N_index in range(4):
    #     ax1.scatter(np.log10(image_dims ** 2), np.log10(ub_mp_ps[N_index, :]), marker="o", edgecolors="k", facecolors="none")
    #     ax1.scatter(np.log10(image_dims ** 2), np.log10(b_mp_ps[N_index, :]), marker="o", edgecolors="r", facecolors="none")

    fig.savefig("/scratch/github/cuDART_wdir/profile.png", dpi=300, bbox_inches="tight")




if __name__ == "__main__":

    data_dir = "/mnt/kocsis1/cuDART_wdir/profiling/data"
    prof_dir = "/mnt/kocsis1/cuDART_wdir/profiling/profiles"
    save_str = "/mnt/kocsis1/cuDART_wdir/profiling/timings.npy"
    D_span = [64,128,256]
    N_span = [64,128,256]
    profiler = Profiler(data_dir = data_dir, prof_dir = prof_dir)
    # profiler.build(D_span = D_span, save_boosted = True)
    profiler.run(N_span = N_span, D_span = D_span, num_iter=10)
    profiler.save_timings(save_str = save_str, N_span = N_span, D_span = D_span)

    timing_data = np.load(save_str)
    print(timing_data)
    