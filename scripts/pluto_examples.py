# external imports
import sys, os, gc
import numpy as np
import matplotlib.image as mpimg

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

def render_pluto_data_example(relativistic=False, remove_raw_images = True):

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
    template_camera.length_X = 0.66
    template_camera.length_Y = 0.66

    # build camera array, inherit from template
    num_img = 200
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
    scene.render(verbose = True, relativistic = relativistic)
    print("finished rendering raw images")

    scene.plot_wlightcurve(png_save_str, cmap = "afmhot", verbose = True, remove_raw_images = remove_raw_images, vmin=18, vmax=21)
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
    print(np.shape(example_img))

    set_plot_defaults()
    L = 20.0 / 3
    height_ratios = np.array([0.3, 1])
    width_ratios = np.array([1,1,0.05])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    fig = plt.figure(figsize=(L, L * h_over_w))
    gs = fig.add_gridspec(2,3,height_ratios=height_ratios,width_ratios=width_ratios)
    axl = fig.add_subplot(gs[1,0])
    axr = fig.add_subplot(gs[1,1])
    axs = [axl, axr]
    tax = fig.add_subplot(gs[0,:])
    cax = fig.add_subplot(gs[1,2])
    # inset = axl.inset_axes([0.75, 0.05, 0.5, ])
    plt.subplots_adjust(hspace=0, wspace=0)

    # plot colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, orientation="vertical")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r"$\log_{10}\left(I_{\nu}\right)$")

    tax.set_xlim([0,1])
    tax.xaxis.tick_top()
    tax.xaxis.set_label_position("top")
    tax.set_xlabel(r"$\theta$ / $\pi$")
    axl.plot([],[],color='w', label="Unboosted", linestyle="dashed")
    axr.plot([],[],color='w', label="Unboosted", linestyle="solid")
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
        title_str = r"$\left(L_\nu - \bar{L}_\nu \right)/ \bar{L}_\nu$"
        tax.set_title(title_str)
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

if __name__ == "__main__":

    comp_plot()
    