# external imports
import sys, os, gc
import numpy as np
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd

# local import
pysrc = os.path.join(os.path.dirname(__file__), "..", "pysrc")
sys.path.append(pysrc)
from cudart import *

epsilon = 1e-2 # small number to avoid casts with exact cooordinate alignment

kpc_to_m = 1e3 * 3.086e+16
Myr_to_s = 1e6 * 365 * 24 * 60 * 60
c_light = 3e8

def extract_pluto_data_example():

    from pluto_reader import PlutoParticleReader, VTKLoader, Frequencies, Units

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

def render_pluto_data_example(relativistic=False, remove_raw_images = True, save_profile = None, save_lc = None, append = False):

    print("cuDART: starting jet render example...")

    # define targets
    # npy_load_str = "/mnt/kocsis1/cuDART_wdir/emm_data/emm_1000MHz.npy"
    # npy_save_str = "/mnt/kocsis1/cuDART_wdir/append_test/raw"
    # png_save_str = "/mnt/kocsis1/cuDART_wdir/append_test/img"
    # camera_file_name = "/mnt/kocsis1/cuDART_wdir/append_test/cameras.txt"
    npy_load_str = "/mnt/kocsis1/cuDART_wdir/lookback_data/snapshot00009.npy"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/lookback_data/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/lookback_data/img"
    camera_file_name = "/mnt/kocsis1/cuDART_wdir/lookback_data/cameras.txt"
    # npy_load_str = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/emm_1000MHz.npy"
    # npy_save_str = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/emm_img/raw"
    # png_save_str = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/emm_img/img"
    # camera_file_name = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/cameras.txt"

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = (60.0 / 180) * np.pi
    template_camera.length_X = 0.66 # defval 0.66
    template_camera.length_Y = 0.66

    # build camera array, inherit from template
    num_img = 10
    phi = epsilon
    theta_ar = np.linspace(epsilon,np.pi - epsilon,num_img, endpoint=False)
    cameras = []
    for theta in theta_ar:
        camera = copy.deepcopy(template_camera)
        camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)
        cameras.append(camera)
    print("initialised cameras")

    # generate scene
    
    scene = Scene(npy_load_str, npy_save_str, cameras, camera_file_name=camera_file_name)
    print("built scene")

    # render and save images
    scene.render(verbose = True, relativistic = relativistic, save_profile = save_profile, append = append)
    print("finished rendering raw images")

    if save_lc is not None:
        scene.calc_lightcurve(save_lc)
        print("saved lightcurve")

    scene.plot(png_save_str, cmap = "afmhot", verbose = True, remove_raw_images = remove_raw_images, vmin=18, vmax=21)
    print("finished rendering rasterised images")

    print("unlabelled render example finished.")

def cardinal_rotate(axes="x"):

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

def comp_plot():

    vmin = 18
    vmax = 22
    cmap = "afmhot"

    boosted_dir = "/mnt/kocsis1/cuDART_wdir/emm_img/index3.6"
    unboosted_dir = "/mnt/kocsis1/cuDART_wdir/emm_img/unboosted"
    save_dir = "/mnt/kocsis1/cuDART_wdir/emm_img"
    num_img = 100
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

        unboosted_raw_str = os.path.join(unboosted_dir, "raw" + str(n).zfill(5) + ".npy")
        unboosted_img = np.load(unboosted_raw_str)
        pcl = axl.pcolormesh(XX, YY, np.log10(unboosted_img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")

        boosted_raw_str = os.path.join(boosted_dir, "raw" + str(n).zfill(5) + ".npy")
        boosted_img = np.load(boosted_raw_str)
        pcr = axr.pcolormesh(XX, YY, np.log10(boosted_img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")

        save_str = os.path.join(save_dir, "img" + str(n).zfill(5) + ".png")
        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        tstamp.remove()
        pcl.remove()
        pcr.remove()

    plt.close("all")

def build_lookback_data(num_snapshots = 10):

    save_dir = "/mnt/kocsis1/cuDART_wdir/lookback_data/"
    max_emm = 1.0
    xspan = np.linspace(0,1,100)
    yspan = np.linspace(0,1,100)
    zspan = np.linspace(-1,1,200)
    xx, yy, zz = np.meshgrid(xspan, yspan, zspan, indexing="ij")
    r_sqr = (xx-0.5) ** 2 + (yy-0.5) ** 2
    r_jet = 0.1
    in_jet_column = (r_sqr < r_jet ** 2)

    v_adv = 1.0 / num_snapshots
    v_jet = 0.99 # in units of c
    for n in range(0, num_snapshots):
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        save_data = np.zeros(shape=(100,100,200))
        L_jet = v_adv * n
        in_lead = in_jet_column & (zz > 0) & (zz < L_jet) 
        in_tail = in_jet_column & (zz < 0) & (zz > -L_jet)
        save_data[in_lead] = max_emm
        save_data[in_tail] = max_emm
        save_data = save_data.astype(np.float32)
        np.save(save_str, save_data)

def build_blob_data(num_snapshots = 10, save_dir="/mnt/kocsis1/cuDART_wdir/lookback_data/", gamma_bulk=2):

    header_str = os.path.join(save_dir, "header.txt")
    max_emm = 1.0
    long_dim = 500
    short_dim = 250
    xspan = np.linspace(-0.25,0.25,short_dim) # in code units, longest side is length unity
    yspan = np.linspace(-0.25,0.25,short_dim)
    zspan = np.linspace(-0.5,0.5,long_dim)
    ispan = np.array([0,1,2,3])
    xx, yy, zz, ii = np.meshgrid(xspan, yspan, zspan, ispan, indexing="ij")
    xy_sqr = xx ** 2 + yy ** 2
    snapshot_size = np.size(xx)

    v_in_c = np.sqrt(1 - 1.0 / gamma_bulk ** 2)
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)

    r_blob_in_kpc = 2.5
    L_in_kpc = 120 # full domain length
    r_blob_in_code = r_blob_in_kpc / L_in_kpc
    T_in_Myr = 0.5 * L_in_kpc / v_in_kpc_per_Myr # duration to reach domain edge
    t_span = np.linspace(0, T_in_Myr, num_snapshots) # evenly space over duration
    with open(header_str, "w") as f:
        f.write("{0} {1} {2} {3}".format(num_snapshots, snapshot_size, t_span[1], L_in_kpc))
    
    for n, t_in_Myr in enumerate(t_span):
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        save_data = np.zeros_like(xx)
        lead_center_in_kpc = t_in_Myr * v_in_kpc_per_Myr
        lead_center = lead_center_in_kpc / L_in_kpc # cast to code units
        tail_center = -lead_center
        lead_ZZ = zz - lead_center
        tail_ZZ = zz - tail_center

        rr_lead_sqr = ((zz - lead_center) ** 2 + xy_sqr) / r_blob_in_code ** 2
        rr_tail_sqr = ((zz - tail_center) ** 2 + xy_sqr) / r_blob_in_code ** 2

        in_lead = (rr_lead_sqr < 1)
        in_tail = (rr_tail_sqr < 1)
        emm_lead = max_emm * (1.0 - rr_lead_sqr)
        emm_tail = max_emm * (1.0 - rr_tail_sqr)

        lead_emm_mask = (in_lead) & (ii == 0)
        tail_emm_mask = (in_tail) & (ii == 0)
        lead_vel_mask = (in_lead) & (ii == 3)
        tail_vel_mask = (in_tail) & (ii == 3)
        save_data[lead_emm_mask] = emm_lead[lead_emm_mask]
        save_data[tail_emm_mask] = emm_tail[tail_emm_mask]
        save_data[lead_vel_mask] = v_in_c
        save_data[tail_vel_mask] = -v_in_c
        save_data = save_data.astype(np.float32)
        np.save(save_str, save_data)

def build_boosted_lookback_data(num_snapshots = 10, save_dir="/mnt/kocsis1/cuDART_wdir/lookback_data/"):

    max_emm = 1.0
    xspan = np.linspace(0,1,100)
    yspan = np.linspace(0,1,100)
    zspan = np.linspace(-1,1,200)
    ispan = np.array([0,1,2,3])
    xx, yy, zz, ii = np.meshgrid(xspan, yspan, zspan, ispan, indexing="ij")
    r_sqr = (xx-0.5) ** 2 + (yy-0.5) ** 2
    r_jet = 0.025
    in_jet_column = (r_sqr < r_jet ** 2)

    v_adv = 4 # in units kpc per Myr
    v_adv *= 70
    L_in_kpc = 120 # domain extent (full)
    T_in_Myr = L_in_kpc / v_adv
    v_jet = 0.99 # in units of c
    t_span = np.linspace(0, T_in_Myr, num_snapshots) # evenly space over duration
    for n in range(0, num_snapshots):
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        save_data = np.zeros(shape=(100,100,200,4))
        t_in_Myr = t_span[n]
        L_jet_in_kpc = v_adv * t_in_Myr
        L_jet_code = L_jet_in_kpc / L_in_kpc
        in_lead = in_jet_column & (zz > 0) & (zz < L_jet_code) 
        in_tail = in_jet_column & (zz < 0) & (zz > -L_jet_code)
        emm_mask = (in_lead | in_tail) & (ii == 0) 
        lead_vel_mask = (in_lead) & (ii == 3)
        tail_vel_mask = (in_tail) & (ii == 3)
        save_data[emm_mask] = max_emm
        save_data[lead_vel_mask] = v_jet
        save_data[tail_vel_mask] = -v_jet
        save_data = save_data.astype(np.float32)
        np.save(save_str, save_data)

def render_lookback_example(relativistic=False, remove_raw_images = True, save_profile = None, save_lc = None, append = False):

    print("cuDART: starting jet render example...")

    # define targets
    load_str = "/mnt/kocsis1/cuDART_wdir/lookback_data"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/lookback_data/flat/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/lookback_data/flat/img"
    camera_file_name = "/mnt/kocsis1/cuDART_wdir/lookback_data/cameras.txt"
    # npy_load_str = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/emm_1000MHz.npy"
    # npy_save_str = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/emm_img/raw"
    # png_save_str = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/emm_img/img"
    # camera_file_name = "/data/phys-dynamic-disc/wadh6663/cuDART_wdir/cameras.txt"

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = (60.0 / 180) * np.pi
    template_camera.t_obs = 0.5 # in units of Myr
    phi = epsilon
    theta = np.pi / 2 + epsilon
    template_camera.length_X = 1.0 * np.sin(theta) # size window to fit jet alignment
    template_camera.length_Y = 1.0 * np.sin(theta)
    template_camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)
    
    num_img = 100
    cameras = []
    L_in_kpc = 120 
    gamma_bulk = 2
    v_in_c = np.sqrt(1 - 1.0 / gamma_bulk ** 2)
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)
    T_in_Myr = 0.5 * L_in_kpc / v_in_kpc_per_Myr # duration to reach domain edge
    dist_to_camera_in_kpc = 2 * L_in_kpc
    t_delay_in_Myr = dist_to_camera_in_kpc * kpc_to_m / (c_light * Myr_to_s)
    t_delay_in_Myr *= 0.95
    for t in np.linspace(t_delay_in_Myr, t_delay_in_Myr + T_in_Myr * 2, num_img):
        camera = copy.deepcopy(template_camera)
        camera.t_obs = t
        cameras.append(camera)
    print("initialised cameras")

    # # build camera array, inherit from template
    # num_img = 10
    # phi = epsilon
    # theta_ar = np.linspace(epsilon,np.pi - epsilon,num_img, endpoint=False)
    # cameras = []
    # for theta in theta_ar:
    #     camera = copy.deepcopy(template_camera)
    #     camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)
    #     cameras.append(camera)
    

    # generate scene
    
    scene = Scene(load_str, npy_save_str, cameras, camera_file_name=camera_file_name)
    print("built scene")

    # render and save images
    scene.render(verbose = True, relativistic = relativistic, save_profile = save_profile, append = append, lookback=True)
    print("finished rendering raw images")

    if save_lc is not None:
        scene.calc_lightcurve(save_lc)
        print("saved lightcurve")

    #scene.plot(png_save_str, cmap = "afmhot", verbose = True, remove_raw_images = remove_raw_images, vmin=-6, vmax=0, show_grid=True)
    print("finished rendering rasterised images")

    print("unlabelled render example finished.")

def label_lookback(num_img=100, sparse_step=1):

    load_dir = "/mnt/kocsis1/cuDART_wdir/lookback_data/flat"
    save_dir = "/mnt/kocsis1/cuDART_wdir/lookback_data/flat/label"

    Gamma = 2
    beta = np.sqrt(1 - 1.0 / Gamma ** 2)
    theta = np.pi / 2
    F = (1 + beta * np.cos(theta)) / (1 - beta * np.cos(theta))
    L_domain = 120 # in kpc
    L_img = L_domain * np.sin(theta)

    v_in_kpc_per_Myr = beta * c_light / (kpc_to_m / Myr_to_s)
    T_in_Myr = 0.5 * L_domain / v_in_kpc_per_Myr # duration to reach domain edge
    t_span = np.linspace(0, T_in_Myr * 2, num_img)

    label_str = r"$\Gamma = 2$" + "\n"
    label_str += r"$\beta = \sqrt{3}/2$" + "\n"
    label_str +=  r"$\theta = \pi / 2$" + "\n"
    label_str += r"$\beta_\mathrm{T}^+ = \beta$" + "\n"
    label_str += r"$\beta_\mathrm{T}^+ / \beta_\mathrm{T}^- = 1$"

    set_plot_defaults()
    width_ratios = np.array([1,0.05])
    height_ratios = np.array([1])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    L = 10.0 / 3
    fig = plt.figure(figsize=(L, h_over_w * L))
    gs = fig.add_gridspec(1,2,width_ratios=width_ratios,height_ratios=height_ratios)
    ax = fig.add_subplot(gs[:,0])
    cax = fig.add_subplot(gs[:,1])
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_facecolor("k")
    ax.text(0.45,0.45,label_str, color='w', va="top", ha="right")
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.set_title("Regular Motion")

    X = np.linspace(-0.5,0.5,2048)
    Y = np.linspace(-0.5,0.5,2048)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    vmin = -6
    vmax = 0
    cmap = "afmhot"

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, orientation="vertical")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r"$\log_{10}\left(I_{\nu}/I_{\nu,0}\right)$")

    ax.axhline(y=0, color='w', alpha=0.2, zorder=20)
    ax.axvline(x=0, color='w', alpha=0.2, zorder=20)

    bar_length = 25 / L_img
    sb = AnchoredSizeBar(ax.transData, bar_length, "25kpc", "lower left", pad=1, zorder=10,
                        size_vertical = 0.05 * bar_length, frameon=False, color='w', label_top=True)
    ax.add_artist(sb)

    for n in range(0, num_img, sparse_step):
        load_str = os.path.join(load_dir, "raw" + str(n).zfill(5) + ".npy")
        save_str = os.path.join(save_dir, "img" + str(n).zfill(5) + ".png")
        img = np.load(load_str)
        pc = ax.pcolormesh(XX, YY, np.log10(img), cmap=cmap, vmin=vmin, vmax=vmax)
        time_label = "$\Delta t$ = {0:.3f}Myr".format(t_span[n])
        label = ax.text(-0.45,0.45,time_label,color='w', va="top", ha="left", zorder=20)
        
        fig.savefig(save_str, dpi=600, bbox_inches="tight")
        label.remove()
        pc.remove()

    plt.close("all")

def plot_superluminal(num_img=100, sparse_step=1):

    load_dir1 = "/mnt/kocsis1/cuDART_wdir/lookback_data/flat"
    load_dir2 = "/mnt/kocsis1/cuDART_wdir/lookback_data"
    load_dirs = [load_dir1, load_dir2]
    save_dir = "/mnt/kocsis1/cuDART_wdir/lookback_data/comp"

    Gamma = 2
    beta = np.sqrt(1 - 1.0 / Gamma ** 2)
    thetas = [np.pi / 2, np.pi / 4]
    L_domain = 120 # in kpc
    L_imgs = [L_domain * np.sin(theta) for theta in thetas]
    titles = [r"Regular Motion", r"Superluminal Motion"]

    v_in_kpc_per_Myr = beta * c_light / (kpc_to_m / Myr_to_s)
    T_in_Myr = 0.5 * L_domain / v_in_kpc_per_Myr # duration to reach domain edge
    t_span = np.linspace(0, T_in_Myr * 2, num_img)

    label_str0 = r"$\Gamma = 2$" + "\n"
    label_str0 += r"$\beta \simeq 0.9$" + "\n"

    label_str1 = r"$\theta = \pi/2$" + "\n"
    label_str1 += r"$\beta_\mathrm{T}^+ \simeq 0.9$" + "\n"
    label_str1 += r"$\beta_\mathrm{T}^+ / \beta_\mathrm{T}^- = 1$"

    label_str2 = r"$\theta = \pi/4$" + "\n"
    label_str2 += r"$\beta_\mathrm{T}^+ \simeq 1.6$" + "\n"
    label_str2 += r"$\beta_\mathrm{T}^+ / \beta_\mathrm{T}^- \simeq 4.2$"

    label_strs = [label_str1, label_str2]

    set_plot_defaults()
    width_ratios = np.array([1,0.025,1,0.025,0.05])
    height_ratios = np.array([1])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    L = 20.0 / 3
    fig = plt.figure(figsize=(L, h_over_w * L))
    gs = fig.add_gridspec(np.size(height_ratios),np.size(width_ratios),width_ratios=width_ratios,height_ratios=height_ratios)
    axl = fig.add_subplot(gs[0,0])
    spacerl = fig.add_subplot(gs[0,1])
    spacerl.axis("off")
    axr = fig.add_subplot(gs[0,2])
    spacerr = fig.add_subplot(gs[0,3])
    spacerr.axis("off")
    cax = fig.add_subplot(gs[0,4])
    axes = [axl, axr]
    plt.subplots_adjust(hspace=0, wspace=0)

    X = np.linspace(-0.5,0.5,2048)
    Y = np.linspace(-0.5,0.5,2048)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    vmin = -6
    vmax = 0
    cmap = "afmhot"

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, orientation="vertical")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r"$\log_{10}\left(I_{\nu}/I_{\nu,0}\right)$")

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_facecolor("k")
        ax.set_xlim([-0.5,0.5])
        ax.set_ylim([-0.5,0.5])
        ax.text(-0.45,0.45,label_str0,va="top", ha="left",color='w')
        # ax.axhline(y=0, color='w', alpha=0.2, zorder=20)
        # ax.axvline(x=0, color='w', alpha=0.2, zorder=20)

        bar_length = 25 / L_imgs[i]
        sb = AnchoredSizeBar(ax.transData, bar_length, "25kpc", "lower left", pad=1, zorder=10,
                            size_vertical = 0.05 * bar_length, frameon=False, color='w', label_top=True)
        ax.add_artist(sb)

    for n in range(0, num_img, sparse_step):
        
        pcs = []
        labels = []
        for i, ax in enumerate(axes):
            load_str = os.path.join(load_dirs[i], "raw" + str(n).zfill(5) + ".npy")
            save_str = os.path.join(save_dir, "img" + str(n).zfill(5) + ".png")
            img = np.load(load_str)
            pcs.append(ax.pcolormesh(XX, YY, np.log10(img), cmap=cmap, vmin=vmin, vmax=vmax))
            labels.append(ax.text(0.45,0.45,label_strs[i], color='w', va="top", ha="right"))
            # time_label = "$\Delta t$ = {0:.3f}Myr".format(t_span[n])
            # label = ax.text(-0.45,0.45,time_label,color='w', va="top", ha="left", zorder=20)
        
        fig.savefig(save_str, dpi=600, bbox_inches="tight")
        for i in range(2):
            pcs[i].remove()
            labels[i].remove()

    plt.close("all")

def build_morphology_suite(num_snapshots=50,gamma_span=[2,4,8]):

    master_dir = "/mnt/kocsis2/hww27/cuDART_wdir/blob_data"
    for gamma in gamma_span:
        save_dir = os.path.join(master_dir, "gamma{0}".format(gamma))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        build_blob_data(num_snapshots=num_snapshots,save_dir=save_dir,gamma_bulk=gamma)

def plot_morphology():

    master_dir = "/mnt/kocsis2/hww27/cuDART_wdir/blob_data"
    save_str = os.path.join(master_dir,"morphology.png")

    spec_snapshot = 2
    gamma_span = [1.15, 2, 4, 8]
    theta_span = [np.pi / 2, np.pi / 4, np.pi / 8]
    theta_labels = [r"$\theta = \pi / 2$", r"$\theta = \pi / 4$", r"$\theta = \pi / 8$"]
    num_gamma = np.size(gamma_span)
    num_theta = np.size(theta_span)
    L_in_kpc = 120 
    r_blob_in_kpc = 2.5
    r_blob_in_code = r_blob_in_kpc / L_in_kpc
    long_res = 256
    long_scale = 1.0 # fill domain vertically
    short_scale = 4.0 * r_blob_in_code
    img_aspect = short_scale / long_scale
    short_res = int(long_res * img_aspect)
    X = np.linspace(-0.5 * long_scale, 0.5 * long_scale, long_res)
    Y = np.linspace(-0.5 * short_scale, 0.5 * short_scale, short_res)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    vmin = -6
    vmax = 0
    cmap = "afmhot"
    xtrim_fac = 0.8 # mult xspan of subplots by this factor

    set_plot_defaults()
    L_fig = 20.0 / 3
    width_ratios = np.array([xtrim_fac] * num_theta + [0.05])
    height_ratios = np.array([img_aspect] * num_gamma)
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    fig = plt.figure(figsize=(L_fig, L_fig * h_over_w))
    gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)

    axes = []
    for i in range(num_gamma):
        row = []
        for j in range(num_theta):
            row.append(fig.add_subplot(gs[i,j]))
        axes.append(row)
    cax = fig.add_subplot(gs[:,-1])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, orientation="vertical")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r"$\log_{10}\left(I_{\nu}/I_{\nu,0}\right)$")

    for i, gamma in enumerate(gamma_span):
        load_dir = os.path.join(master_dir, "gamma{0}".format(gamma))
        beta = np.sqrt(1 - 1.0 / gamma ** 2)
        for j, theta in enumerate(theta_span):
            length_ratio = np.sqrt(1 - 2 * beta * np.cos(theta) + beta ** 2) / (1 - beta * np.cos(theta))
            label = "$\mathcal{L}$" + " = {0:.2f}".format(length_ratio)
            axes[i][j].set_xlim([-xtrim_fac * 0.5, xtrim_fac * 0.5])
            axes[i][j].set_aspect("equal")
            axes[i][j].set_facecolor("k")
            load_str = os.path.join(load_dir, "raw" + str(j).zfill(5) + ".npy")
            if os.path.exists(load_str):
                img = np.load(load_str)
                axes[i][j].pcolormesh(XX, YY, np.log10(img), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i][j].plot([],[],alpha=0,label=label)
            axes[i][j].legend(loc="upper left", frameon=False,labelcolor='w')
            # axes[i][j].axvline(x=-0.5 * np.sin(theta),color='w')
            # axes[i][j].axvline(x=0.5 * np.sin(theta),color='w')

            if i == 0:
                axes[i][j].xaxis.set_label_position("top")
                axes[i][j].set_xlabel(theta_labels[j])
                axes[i][j].xaxis.set_ticks([])
            else:
                axes[i][j].xaxis.set_visible(False)
            
            if j == 0:
                axes[i][j].yaxis.set_ticks([])
                axes[i][j].set_ylabel("$\Gamma$ = {0}".format(gamma))
            else:
                axes[i][j].yaxis.set_visible(False)

    plt.subplots_adjust(hspace=0,wspace=0)
    fig.savefig(save_str, dpi=600, bbox_inches="tight")
    plt.close("all")

def render_morphology(gamma_span=[1.15,2,4,8],theta_span=[np.pi / 2, np.pi / 4, np.pi / 8]):

    master_dir = "/mnt/kocsis2/hww27/cuDART_wdir/blob_data"
    L_in_kpc = 120 
    r_blob_in_kpc = 2.5
    r_blob_in_code = r_blob_in_kpc / L_in_kpc

    # build template camera
    long_res = 256
    long_scale = 1.0 # fill domain vertically (ignore projection effects)
    short_scale = 4.0 * r_blob_in_code
    img_aspect = short_scale / long_scale
    template_camera = Camera()
    template_camera.tilt = np.pi / 2
    template_camera.t_obs = 0.5 # in units of Myr
    phi = epsilon
    theta = np.pi / 2 + epsilon
    template_camera.length_X = long_scale
    template_camera.length_Y = short_scale
    template_camera.num_pixels_X = long_res
    template_camera.num_pixels_Y = int(long_res * img_aspect)
    template_camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)

    for gamma_bulk in gamma_span:
        load_dir = os.path.join(master_dir, "gamma{0}".format(gamma_bulk))
        npy_save_str = os.path.join(load_dir, "raw")

        # build cameras for this gamma value
        v_in_c = np.sqrt(1.0 - 1.0 / gamma_bulk ** 2)
        v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)
        dist_to_camera_in_kpc = 2.0 * L_in_kpc
        t_delay_in_Myr = dist_to_camera_in_kpc * kpc_to_m / (c_light * Myr_to_s)
        T_in_Myr = 0.5 * L_in_kpc / v_in_kpc_per_Myr # duration to reach domain edge
        t_delay_in_Myr = dist_to_camera_in_kpc * kpc_to_m / (c_light * Myr_to_s)
        print("T = {0}Myr".format(T_in_Myr))
        print("t_delay = {0}Myr".format(t_delay_in_Myr))
        # cycle over orientations
        cameras = []
        for theta in theta_span:
            # find proper time
            L_projected = L_in_kpc * np.sin(theta) # projected size of domain 
            x_obs_in_m = 0.25 * L_projected * kpc_to_m # in SI, target displaced half from center
            D_in_m = 2.0 * L_in_kpc * kpc_to_m
            d_in_m = x_obs_in_m * (1 - v_in_c * np.cos(theta)) / (v_in_c * np.sin(theta)) + D_in_m
            t_obs_in_s = d_in_m / c_light
            t_obs = t_obs_in_s / Myr_to_s # cast to Myr
            print("gamma = {0}, theta = {1}, t_obs = {2}Myr".format(gamma_bulk,theta,t_obs))
            # t_obs = t_delay_in_Myr + 0.1 * T_in_Myr

            # find proper camera position (shift in X)
            camera = copy.deepcopy(template_camera)
            camera.t_obs = t_obs
            camera.theta = theta
            camera.set_sph_pos(r=2.0,theta=theta,phi=phi,target_origin=True)
            # unit_normal = camera.normal / np.linalg.norm(camera.normal)
            # unit_Y = camera.bias - np.dot(camera.bias,unit_normal)
            # unit_Y /= np.linalg.norm(unit_Y)
            # unit_X = np.cross(unit_normal, unit_Y)
            # delta_X = -(target_x_obs / kpc_to_m) / L_in_kpc # in code units
            # camera.origin += unit_X * delta_X

            cameras.append(camera)

        print("initialised cameras for gamma = {0}".format(gamma_bulk))

        scene = Scene(load_dir, npy_save_str, cameras)
        print("finished rendering for gamma = {0}".format(gamma_bulk))

        scene.render(verbose = True, relativistic = True, lookback=True)
        print("finished rendering raw images for gamma = {0}".format(gamma_bulk))

if __name__ == "__main__":

    #build_blob_data(num_snapshots=50)
    #render_pluto_data_example(relativistic=False, remove_raw_images = False, append=False)
    #render_lookback_example(relativistic=True, remove_raw_images = False)
    # plot_superluminal(num_img=100, sparse_step=1)
    #build_morphology_suite(gamma_span=[1.15])
    #render_morphology()
    plot_morphology()