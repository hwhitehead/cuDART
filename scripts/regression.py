"""
This file contains a collection of functions test basic cuDART functionality, supporting build mock data set construction, and rendering with or without the lookback routine
"""

# external imports
import sys, os, shutil
import numpy as np
import argparse, re
import time
import matplotlib.patches as patches

# local import
pysrc = os.path.join(os.path.dirname(__file__), "..", "pysrc")
sys.path.append(pysrc)
from cudart import *

def build_unlabelled_regression_suite(save_dir, sim_args, verbose = True):

    # construct template data for regression suite, without labels
    # the template data features twin emitting regions travelling at a fixed velocity in opposite directions
    # the emitting regions are spheres, ellipsoids or jets dependeng on sim_args["build_mode"]
    # if sim_args["target_theta"] not None, ensure snapshot sampling occurs at unaliased frequency

    if (verbose): 
        print("starting regression suite data construction...")
        print("saving data at {0}".format(save_dir))

    if not os.path.isdir(save_dir):
        try: 
            os.mkdir(save_dir)
        except:
            raise Exception("unable to build dir at {0}".format(save_dir))

    # define simulation parameters
    v_in_c = np.sqrt(1 - 1.0 / sim_args["Gamma"] ** 2)                                  # calculate ejecta velocity
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)                         # cast to astro units
    r_in_code = sim_args["r_in_kpc"] / sim_args["L_in_kpc"]                             # cast to code units (where L_domain = 1.0)          
    T_in_Myr = 0.5 * (sim_args["L_in_kpc"] + sim_args["r_in_kpc"]) / v_in_kpc_per_Myr   # calc duration to reach domain edge
    
    # build empty domain 
    max_emm = 1.0
    Lz = 1.0                                                                    # set domain length in z to unity in code units
    Ly = Lz * sim_args["domain_dims"][1] / sim_args["domain_dims"][2]           # auto scale x, y directions
    Lx = Lz * sim_args["domain_dims"][0] / sim_args["domain_dims"][2] 
    xspan = np.linspace(-0.5 * Lx, 0.5 * Lx, sim_args["domain_dims"][0])        # build domain centered on (0,0,0)
    yspan = np.linspace(-0.5 * Ly, 0.5 * Ly, sim_args["domain_dims"][1])
    zspan = np.linspace(-0.5 * Lz, 0.5 * Lz, sim_args["domain_dims"][2])
    ispan = np.array([0,1,2,3])                                                 # dummy indices for spatial, velocity axes
    xx, yy, zz, ii = np.meshgrid(xspan, yspan, zspan, ispan, indexing="ij")     # construct mesh as (x,y,z,i)
    xy_sqr = xx ** 2 + yy ** 2
    snapshot_size = np.size(xx)
    if (verbose): print("built empty mesh.")

    # determine cadence, if critical timing routine flagged
    if sim_args["target_theta"] is not None:
        crit_fac = (1 - v_in_c * np.cos(sim_args["target_theta"])) / np.sin(sim_args["target_theta"])
        dt_crit_in_s = crit_fac * sim_args["r_in_kpc"] * kpc_to_m / (v_in_c * c_light)
        dt_crit = dt_crit_in_s / Myr_to_s
        num_snapshots_crit = int(T_in_Myr / dt_crit)
        num_snapshots = num_snapshots_crit if (num_snapshots_crit > sim_args["num_snapshots"]) else sim_args["num_snapshots"]
    else:
        num_snapshots = sim_args["num_snapshots"]    
    t_span = np.linspace(0, T_in_Myr, num_snapshots)                # evenly snapshot times over duration

    # build header data
    header_str = os.path.join(save_dir, "header.txt")
    
    with open(header_str, "w") as f:
        f.write("{0} {1} {2} {3}".format(num_snapshots, snapshot_size, t_span[1], sim_args["L_in_kpc"]))
    if (verbose): print("built header.")

    if sim_args["build_mode"] == "sphere_rest":
        z_scale = 1.0 / sim_args["Gamma"]
    else:
        z_scale = 1.0

    # build snapshots
    emm_adv = 1.0
    emm_rec = 1.0
    for n, t_in_Myr in enumerate(t_span):
        # unlabelled data is a single .npy file, without a header
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        
        # build data array for this snapshot in time
        save_data = np.zeros_like(xx)
        offset_in_kpc = t_in_Myr * v_in_kpc_per_Myr             # calculate current offset in kpc
        offset = offset_in_kpc / sim_args["L_in_kpc"]           # cast to code units
        
        if sim_args["build_mode"] == "jet":                     # test occupancy in cylinder with moving front
            in_radius = (xy_sqr < r_in_code ** 2)
            in_adv = in_radius & (zz > 0) & (zz < offset)
            in_rec = in_radius & (zz < 0) & (zz > -offset)
        else:                                                   # test occupancy in blob with moving center
            dz_adv = (zz - offset) / z_scale
            dz_rec = (zz + offset) / z_scale
            rr_adv_sqr = (dz_adv ** 2 + xy_sqr)
            rr_rec_sqr = (dz_rec ** 2 + xy_sqr)
            in_adv = (rr_adv_sqr < r_in_code ** 2)
            in_rec = (rr_rec_sqr < r_in_code ** 2)

        lead_emm_mask = (in_adv) & (ii == 0)
        tail_emm_mask = (in_rec) & (ii == 0)
        lead_vel_mask = (in_adv) & (ii == 3)
        tail_vel_mask = (in_rec) & (ii == 3)
        save_data[lead_emm_mask] = emm_adv                      # match emission in lead/tail
        save_data[tail_emm_mask] = emm_rec 
        save_data[lead_vel_mask] = v_in_c                       # invert velocity in lead/tail
        save_data[tail_vel_mask] = -v_in_c
        save_data = save_data.astype(np.float32)                # ENSURE cast to float32!!!
        np.save(save_str, save_data)                            # save snapshot data
        if (verbose): print("built dataset for snapshot {0}/{1}".format(n,num_snapshots))

    if (verbose): print("finished dataset construction.")

# TODO: ensure labelled ctor is tested before v1.0
def build_labelled_regression_suite(save_dir, sim_args, verbose=True, sphere_in_rest = False):

    # construct template data for regression suite, with labels
    # the template data features twin emitting regions travelling at a fixed velocity in opposite directions
    # the emitting regions are spheres in the observer frame
    # the emission in the spheres falls off quadratically with radius, out to a fixed, finite radius
    # if given specific theta, ensure sampling occurs at unaliased frequency

    if (verbose): 
        print("starting regression suite data construction...")
        print("saving data at {0}".format(save_dir))

    if not os.path.isdir(save_dir):
        try: 
            os.mkdir(save_dir)
        except:
            raise Exception("unable to build dir at {0}".format(save_dir))

    # define simulation parameters
    v_in_c = np.sqrt(1 - 1.0 / sim_args["Gamma"] ** 2)                          # calculate ejecta velocity
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)                 # cast to astro units
    r_blob_in_code = sim_args["r_in_kpc"] / sim_args["L_domain"]                  # cast to code units (where L_domain = 1.0)
    T_in_Myr = 0.5 * sim_args["L_domain"] / v_in_kpc_per_Myr                    # calc duration for blob to reach domain edge
    
    # build empty domain 
    max_emm = 1.0
    Lz = 1.0                                                                    # set domain length in z to unity in code units
    Ly = Lz * sim_args["domain_dims"][1] / sim_args["domain_dims"][2]           # auto scale x, y directions
    Lx = Lz * sim_args["domain_dims"][0] / sim_args["domain_dims"][2] 
    xspan = np.linspace(-0.5 * Lx, 0.5 * Lx, sim_args["domain_dims"][0])        # build domain centered on (0,0,0)
    yspan = np.linspace(-0.5 * Ly, 0.5 * Ly, sim_args["domain_dims"][1])
    zspan = np.linspace(-0.5 * Lz, 0.5 * Lz, sim_args["domain_dims"][2])
    ispan = np.array([0,1,2,3])                                                 # dummy indices for spatial, velocity axes
    xx, yy, zz, ii = np.meshgrid(xspan, yspan, zspan, ispan, indexing="ij")     # construct mesh as (x,y,z,i)
    xy_sqr = xx ** 2 + yy ** 2
    snapshot_size = np.size(xx)
    if (verbose): print("built empty mesh.")

    # determine cadence
    if sim_args["target_theta"] is not None:
        crit_fac = (1 - v_in_c * np.cos(sim_args["target_theta"])) / np.sin(sim_args["target_theta"])
        dt_crit_in_s = crit_fac * sim_args["r_in_kpc"] * kpc_to_m / (v_in_c * c_light)
        dt_crit = dt_crit_in_s / Myr_to_s
        num_snapshots_crit = int(T_in_Myr / dt_crit)
        num_snapshots = num_snapshots_crit if (num_snapshots_crit > sim_args["num_snapshots"]) else sim_args["num_snapshots"]
    else:
        num_snapshots = sim_args["num_snapshots"]    
    t_span = np.linspace(0, T_in_Myr, num_snapshots)                # evenly snapshot times over duration

    # build header data
    header_str = os.path.join(save_dir, "header.txt")
    
    with open(header_str, "w") as f:
        f.write("{0} {1} {2} {3}".format(num_snapshots, snapshot_size, t_span[1], sim_args["L_domain"]))
    if (verbose): print("built header.")

    # build snapshots
    for n, t_in_Myr in enumerate(t_span):
        # labelled data lives in its own folder, including a header text file
        save_dir = os.path.join(save_dir, "snapshot" + str(n).zfill(5))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        # build data array for this snapshot in time
        save_data = np.zeros_like(xx)
        offset_in_kpc = t_in_Myr * v_in_kpc_per_Myr             # calculate current offset in kpc
        offset = offset_in_kpc / sim_args["L_in_kpc"]           # cast to code units
        
        if sim_args["build_mode"] == "jet":                     # test occupancy in cylinder with moving front
            in_radius = (xy_sqr < r_in_code ** 2)
            in_adv = in_radius & (zz > 0) & (zz < offset)
            in_rec = in_radius & (zz < 0) & (zz > -offset)
        else:                                                   # test occupancy in blob with moving center
            dz_adv = (zz - offset) / z_scale
            dz_rec = (zz + offset) / z_scale
            rr_adv_sqr = (dz_adv ** 2 + xy_sqr)
            rr_rec_sqr = (dz_rec ** 2 + xy_sqr)
            in_adv = (rr_adv_sqr < r_in_code ** 2)
            in_rec = (rr_rec_sqr < r_in_code ** 2)

        lead_emm_mask = (in_adv) & (ii == 0)
        tail_emm_mask = (in_rec) & (ii == 0)
        lead_vel_mask = (in_adv) & (ii == 3)
        tail_vel_mask = (in_rec) & (ii == 3)
        save_data[lead_emm_mask] = emm_adv                      # match emission in lead/tail
        save_data[tail_emm_mask] = emm_rec 
        save_data[lead_vel_mask] = v_in_c                       # invert velocity in lead/tail
        save_data[tail_vel_mask] = -v_in_c
        save_data = save_data.astype(np.float32)  
        
        # for demonstration, partition data into two MeshBlocks split along z = 0
        mask_a = (zz >= 0)                                          
        mask_b = (zz < 0)

        # build Mesh to contain MeshBlock data
        mesh = Mesh(save_dir)

        # partitioned data needs to have spatial labels
        mb_data_a = save_data[mask_a]                                           # select lower half of data 
        xl_a = np.array([-0.5 * Lx, -0.5 * Ly, -0.5 * Lz])                      # lower corner of data_a
        xr_a = np.array([0.5 * Lx, 0.5 * Ly, 0.0])                              # upper corner of data_a
        mesh.add_meshblock(mb_data_a, xl_a, xr_a)                               # add MeshBlock to Mesh

        mb_data_b = save_data[mask_b]                                           # select upper half of data
        xl_b = np.array([-0.5 * Lx, -0.5 * Ly, 0.0])                            # lower corner of data_b
        xr_b = np.array([0.5 * Lx, 0.5 * Ly, 0.5 * Lz])                         # upper corner of data_b
        mesh.add_meshblock(mb_data_b, xl_b, xr_b)                               # add MeshBlock to Mesh

        mesh.write_header()                                                     # save header for Mesh directory
        if (verbose): print("built labelled dataset for snapshot {0}/{1}".format(n,num_snapshots))

    if (verbose): print("finished labelled dataset construction.")

def render_single_snapshot(load_dir, save_dir, camera_args, snapshot_index = None, num_snapshots = None, verbose = True):

    # run a rendering test on a single snapshot of data
    # data should be loaded from suite built using build_regression_suite (-b or -bl flags)
    # render uses a set number of cameras evenly spanning the azimuthal axis 
    # optional camera_args["save_fig"] to render raws as figures (.npy -> .png)

    if (verbose): 
        print("starting no-lookback render test...")
        print("reading data from {0}".format(load_dir))
        print("saving data at {0}".format(save_dir))

    # calculate snapshot_index if not specified
    if snapshot_index is None:
        if num_snapshots is None:
            # calculate number of snapshots in load_dir 
            _, dirs, files = next(os.walk(load_dir))
            num_dirs = len(dirs)
            if num_dirs > 0: # assume data is labelled
                num_snapshots = num_dirs
            else:
                file_array = np.array(files)
                npy_files = [file for file in files if file.startswith("snapshot")]
                num_snapshots = len(npy_files)
                if num_snapshots is None:
                    raise Exception("unable to locate .npy files at {0}".format(load_dir))
        snapshot_index = np.floor(0.5 * (num_snapshots-1))

    # check input, output directory existence
    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(path))

    # check for specific snapshot input file
    load_str = os.path.join(load_dir, "snapshot" + str(int(snapshot_index)).zfill(str_zfill) + ".npy")
    if not os.path.exists(load_str):
        raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(load_str))

    # prepare array of cameras (cycle evenly over theta in [0,pi])
    if (verbose): print(r"building {0} cameras evenly spanning theta in [0,pi]".format(camera_args["num_img"]))
    theta_ar = np.linspace(epsilon, np.pi - epsilon, camera_args["num_img"])
    cameras = []
    for i, theta in enumerate(theta_ar):
        camera = copy.deepcopy(camera_args["template"])
        camera.theta = theta
        if (camera_args["resize_img"]):
            camera.length_X = camera_args["template"].length_X * np.sin(theta)
            camera.length_Y = camera_args["template"].length_Y * np.sin(theta)
        camera.set_sph_pos(r = 2.0, target_origin = True)
        cameras.append(camera)    
        if (verbose): print("built camera {0}/{1} at theta = {2:.2f}deg...".format(i+1, camera_args["num_img"], theta*180.0/np.pi))
    if (verbose): print("finished camera initialisation.")

    # generate scene
    scene = Scene(load_str = load_str, save_dir = save_dir, cameras = cameras, camera_file_name = camera_args["camera_file_name"])
    if (verbose): print("built scene.")

    # render and save images
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = False, verbose_cpp = verbose,
                save_profile = False)
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(fig_save_dir = save_dir, cmap = "afmhot", verbose = verbose, remove_raw_npy = False, vmin= -6, vmax = 0)
        if (verbose): print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

def render_without_lookback(load_dir, save_dir, camera_args, verbose = True):

    # run a rendering test without lookback, on all snapshots of data in load_dir
    # data should be loaded from suite built using build_regression_suite (-b or -bl flags)
    # render uses a set number of cameras evenly spanning the azimuthal axis 
    # optional camera_args["save_fig"] to render raws as figures (.npy -> .png)

    if (verbose): 
        print("starting no-lookback render test...")
        print("reading data from {0}".format(load_dir))
        print("saving data at {0}".format(save_dir))

    # check input, output directory existence
    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(path))

    # determine total number of snapshots in load_dir
    _, dirs, files = next(os.walk(load_dir))
    num_snapshots = len([file for file in files if file.startswith("snapshot")])
    if (verbose): print(r"Identified {0} snapshots in {1}".format(num_snapshots, load_dir))

    # generate a single camera, reuse over renders
    if (verbose): print(r"building single camera")
    if (camera_args["resize_img"]): # apply resize by orientaiton if flagged
            camera_args["template"].length_X *= np.sin(camera_args["template"].theta)
            camera_args["template"].length_Y *= np.sin(camera_args["template"].theta)
    camera_args["template"].set_sph_pos(r = 2.0, target_origin = True)
    cameras = [camera_args["template"]]
    if (verbose): print("finished camera initialisation.")

    # generate scratch space for raw
    scratch_dir = os.path.join(save_dir, "scratch")

    # iterate over snapshots
    for n in range(num_snapshots):

        # generate scene
        load_str = os.path.join(load_dir, "snapshot" + str(n).zfill(5) + ".npy")
        scene = Scene(load_str = load_str, save_dir = scratch_dir, cameras = cameras, camera_file_name = camera_args["camera_file_name"])
        if (verbose): print("built scene for snapshot {0}.".format(n))

        # render raw images
        scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = False, verbose_cpp = verbose,
                    save_profile = False)
        if (verbose): print("finished rendering raw images.")

        # copy raw to main save direction
        scratch_file = os.path.join(scratch_dir, "raw00000.npy")
        raw_file = os.path.join(save_dir, "raw" + str(n).zfill(5) + ".npy")
        shutil.move(scratch_file, raw_file)
        
    # render figures if flagged
    if (camera_args["save_fig"]):
        # define persistent figure to reuse for each image
        fig = plt.figure(figsize=(10.0/3,10.0/3))
        ax = fig.add_subplot()
        ax.set_facecolor("k")
        plt.subplots_adjust(hspace=0, wspace=0)
        X = np.linspace(0,1,camera_args["template"].num_pixels_X+1)
        Y = np.linspace(0,1,camera_args["template"].num_pixels_Y+1)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        # iterate over raw images)
        for i in range(num_snapshots):
            # build load, save paths
            load_str = os.path.join(save_dir, "raw{0}.npy".format(str(i).zfill(str_zfill)))
            save_str = os.path.join(save_dir, "img{0}.png".format(str(i).zfill(str_zfill)))

            # load image data
            img = np.load(load_str)
            img = np.log10(img)
            pc = ax.pcolormesh(XX, YY, img, vmin=-6, vmax=0, cmap="afmhot", shading="flat")
            
            # save figure, and cleanup
            fig.savefig(save_str, dpi=300, bbox_inches="tight")
            pc.remove()
            if (verbose): print("saved png at {0}".format(save_str))

        plt.close("all")

    if (verbose): print("finished no-lookback render, see {0} for output".format(save_dir))

def render_with_lookback(load_dir, save_dir, sim_args, camera_args, verbose = True):

    # run a rendering test using finite speed of light with multiple snapshots
    # data should be loaded from suite built using build_regression_suite (-b flag)
    # render uses a set number of cameras evenly spaced in observer time
    # optional call included to render raws as figures (.npy -> .png)
    # sim_args is passed here to calculate timings, generally it is not required

    if (verbose): 
        print("starting lookback render test...")
        print("reading data from {0}".format(load_dir))
        print("saving data at {0}".format(save_dir))

    # check input, output directory existence
    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(path))

    # check for ALL snapshot input files
    for n in range(0, sim_args["num_snapshots"]):
        snapshot_str = os.path.join(load_dir, "snapshot" + str(n).zfill(5) + ".npy")
        if not os.path.exists(snapshot_str):
            raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(snapshot_str))

    # collect data from args
    v_in_c = np.sqrt(1.0 - 1.0 / sim_args["Gamma"] ** 2)                                        # calculate velocity in units of c
    theta = camera_args["template"].theta                                                       # collect orientation from template

    # calculate start time (just before light from origin reaches camera)
    D_in_m = 2.0 * sim_args["L_in_kpc"] * kpc_to_m                                              # origin-camera seperation 
    t_min_in_s = D_in_m / c_light                                                               # light flight time from origin to camera
    t_min = t_min_in_s / Myr_to_s                                                               # cast to astro/code units                 
    #t_min *= 0.95                                                                              # start render just before flight time 

    # alternate timings to match no-lookback form
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)                                 # cast to astro units    
    t_max = t_min + 0.5 * (sim_args["L_in_kpc"] + sim_args["r_in_kpc"]) / v_in_kpc_per_Myr      # calc duration to reach domain edge

    # # calculate stop time (when receding ejectum reaches maximal extent)
    # x_max_in_m = 0.5 * sim_args["L_in_kpc"] * np.sin(theta) * kpc_to_m                          # max obs blob displacement for given theta
    # d_in_m = x_max_in_m * (1 + v_in_c * np.cos(theta)) / (v_in_c * np.sin(theta)) + D_in_m      # invert superluminal motion eq to calc flight time
    # t_max_in_s = d_in_m / c_light                                                               # observer time when RECEDING blob reaches domain edge
    # t_max = t_max_in_s / Myr_to_s                                                               # cast to astro/code units    

    # generate array of cameras, evenly seperated in observer time
    if (verbose): print(r"building {0} cameras evenly spanning t_obs in [{1},{2}]Myr".format(camera_args["num_img"],t_min, t_max))
    t_obs_ar = np.linspace(t_min, t_max, camera_args["num_img"])
    if (camera_args["resize_img"]): # apply resize by orientaiton if flagged
            camera_args["template"].length_X *= np.sin(theta)
            camera_args["template"].length_Y *= np.sin(theta)
    camera_args["template"].set_sph_pos(r = 2.0, target_origin = True)
    cameras = []
    for i, t_obs in enumerate(t_obs_ar):
        camera = copy.deepcopy(camera_args["template"])
        camera.t_obs = t_obs
        cameras.append(camera) 
        if (verbose): print("built camera {0}/{1} at t_obs = {2:.3f}Myr...".format(i + 1, camera_args["num_img"], t_obs))
    if (verbose): print("finished camera initialisation.")

    # generate scene
    scene = Scene(load_str = load_dir, save_dir = save_dir, cameras = cameras, camera_file_name = camera_args["camera_file_name"])
    if (verbose): print("built scene.")

    # render and save images
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = True, verbose_cpp = verbose,
                save_profile = False)
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(fig_save_dir = save_dir, cmap = "afmhot", verbose = verbose, remove_raw_npy = False, vmin = -6, vmax = 0)
        print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

def compare_lookback(load_dir, save_dir, sim_args, camera_args, verbose = True, show_masks = False):

    if (verbose): 
        print("starting lookback render test...")
        print("reading data from {0}".format(load_dir))
        print("saving data at {0}".format(save_dir))

    # check input, output directory existence
    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(path))

    # check for ALL snapshot input files
    for n in range(0, sim_args["num_snapshots"]):
        snapshot_str = os.path.join(load_dir, "snapshot" + str(n).zfill(5) + ".npy")
        if not os.path.exists(snapshot_str):
            raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(snapshot_str))

    # collect data from args
    v_in_c = np.sqrt(1.0 - 1.0 / sim_args["Gamma"] ** 2)                                        # calculate velocity in units of c
    theta = camera_args["template"].theta                      
    r_blob_in_code = sim_args["r_in_kpc"] / sim_args["L_in_kpc"]

    # first render at midpoint time for emitter
    nolookback_load_str = os.path.join(load_dir, "snapshot" + str(int(0.5 * sim_args["num_snapshots"])).zfill(5) + ".npy")

    # second render at midpoint displacement for observer
    x_obs_mid_m = 0.25 * sim_args["L_in_kpc"] * np.sin(theta) * kpc_to_m                        # cast to astro units    
    D_in_m = 2.0 * sim_args["L_in_kpc"] * kpc_to_m                                              # origin-camera seperation    
    d_mid_m = x_obs_mid_m * (1 - v_in_c * np.cos(theta)) / (v_in_c * np.sin(theta)) + D_in_m
    t_obs_in_s = d_mid_m / c_light
    t_obs = t_obs_in_s / Myr_to_s

    # identify ejecta positions
    D_av = c_light * t_obs * Myr_to_s - D_in_m
    x_adv_m = v_in_c * np.sin(theta) * D_av / (1 - v_in_c * np.cos(theta))
    x_adv = x_adv_m / (sim_args["L_in_kpc"] * kpc_to_m)
    x_rec_m = v_in_c * np.sin(theta) * D_av / (1 + v_in_c * np.cos(theta))
    x_rec = x_rec_m / (sim_args["L_in_kpc"] * kpc_to_m)

    x_adv_pos = (0.5 - x_adv) # shift to image space
    x_rec_pos = (0.5 + x_rec)
    x_positions = [[0.5 - 0.25 * np.sin(theta), 0.5 + 0.25 * np.sin(theta)], [x_adv_pos, x_rec_pos]]
    
    # generate single camera
    camera_args["template"].set_sph_pos(r = 2.0, phi = epsilon, theta = theta, target_origin = True)
    camera_args["template"].t_obs = t_obs # only used with the lookback render
    cameras = [camera_args["template"]]

    # build scene and call render, with and without lookback
    labels = ["nolookback", "lookback"]
    save_dirs = [os.path.join(save_dir, label) for label in labels]
    for local_save_dir in save_dirs:
        if not os.path.exists(local_save_dir):
            os.mkdir(local_save_dir)
    load_strs = [nolookback_load_str, load_dir]
    lookbacks = [False, True]
    for i, label in enumerate(labels):
        scene = Scene(load_str = load_strs[i], save_dir = save_dirs[i], cameras = cameras, camera_file_name = camera_args["camera_file_name"])
        scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = lookbacks[i], verbose_cpp = verbose)
    if (verbose): print("finished raw image generation")

    # plot composite
    set_plot_defaults(use_tex=False)
    height_ratios = np.array([1])
    width_ratios = np.array([1,1,0.05])
    h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
    L_fig = 20.0 / 3
    fig = plt.figure(figsize=(L_fig, L_fig * h_over_w))
    gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)
    axl = fig.add_subplot(gs[:, 0])
    axr = fig.add_subplot(gs[:, 1])
    cax = fig.add_subplot(gs[:, 2])
    axes = [axl, axr]

    X = np.linspace(0,camera_args["template"].length_X,camera_args["template"].num_pixels_X)
    Y = np.linspace(0,camera_args["template"].length_Y,camera_args["template"].num_pixels_Y)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    dA = (X[1] - X[0]) * (Y[1] - Y[0])

    subplot_labels = ["Rendered without Lookback", "Rendered with Lookback"]
    math_label = r"$\frac{F_\mathrm{adv}}{F_\mathrm{rec}}$"
    png_str = os.path.join(save_dir, "penrose-terrell.png")
    r_mask = r_blob_in_code * 1.25
    for i, save_dir in enumerate(save_dirs):
        raw_str = os.path.join(save_dir, "raw00001.npy")
        img = np.load(raw_str)
        

        # build masks
        r_adv_sqr = (XX - x_positions[i][0]) ** 2 + (YY - 0.5) ** 2
        r_rec_sqr = (XX - x_positions[i][1]) ** 2 + (YY - 0.5) ** 2
        
        in_adv = (r_adv_sqr < r_mask ** 2)
        in_rec = (r_rec_sqr < r_mask ** 2)

        if show_masks:
            r_adv_sqr_offset = (XX - x_positions[i][0]) ** 2 + (YY - 0.25) ** 2
            r_rec_sqr_offset = (XX - x_positions[i][1]) ** 2 + (YY - 0.25) ** 2
            in_adv_offset = (r_adv_sqr_offset < r_mask ** 2)
            in_rec_offset = (r_rec_sqr_offset < r_mask ** 2)
            CC = np.zeros_like(XX)
            CC[:] = np.nan
            CC[in_adv_offset] = 0.8
            CC[in_rec_offset] = 0.5
            pc = axes[i].pcolormesh(XX, YY, CC, vmin = 0, vmax = 1, cmap = "afmhot", zorder=10)

        L_adv = np.sum(img[in_adv]) * dA
        L_rec = np.sum(img[in_rec]) * dA
        L_ratio = L_adv / L_rec

        pc = axes[i].pcolormesh(XX, YY, np.log10(img), vmin = -6, vmax = 0, cmap = "afmhot", zorder=-5)
        axes[i].set_xlim([0,1])
        axes[i].set_ylim([0,1])
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)
        axes[i].text(0.5,0.95,s=subplot_labels[i], color='w', va="top", ha="center")
        axes[i].text(0.5,0.05,s=math_label + " = {0:.3f}".format(L_ratio), color='w', va="bottom", ha="center")
        axes[i].set_facecolor("k")

    true_ratio = np.power((1 + v_in_c * np.cos(theta)) / (1 - v_in_c * np.cos(theta)), 3.0 + 0.6)
    fig.suptitle(r"$\theta = \frac{\pi}{4}$, $\Gamma = 2$ -> $\frac{F_\mathrm{adv}}{F_\mathrm{rec}} = \left(\frac{1+\beta \cos(\theta)}{1-\beta \cos(\theta)}\right)^{3-\alpha}$" + " = {0:.3f}".format(true_ratio))

    sm = plt.cm.ScalarMappable(cmap="afmhot", norm=plt.Normalize(vmin=-6, vmax=0))
    fig.colorbar(sm, cax=cax, orientation="vertical")
    cax.set_ylabel(r"$\log_{10}(I_\nu / I_{\nu,0})$")

    plt.subplots_adjust(hspace = 0, wspace= 0)
    fig.savefig(png_str, dpi=300, bbox_inches="tight")
    plt.close("all")

    if (verbose): print("finished penrose-terrell test, see {0} for output".format(png_str))

def report_profiling(save_dir, verbose = True):

    profiler = Profiler(save_dir)
    profiler.report(verbose)

if __name__ == "__main__":

    """
    sim_args 
    Dictionary for simulation parameters
    
    Gamma           bulk Lorentz factor
    L_in_kpc        length of longest simulation domain edge in kpc
    r_in_kpc        length scale of emitting region in kpc
    domain_dims     simulation dimensions as [nx, ny, nz]
    num_snapshots   total number of saved snapshots
    target_theta    if not None, builds snapshots at critical rate
    build_mode      emitter geometry (sphere, sphere_rest or jet)
    """

    sim_args = {"Gamma": 2.0,
                "L_in_kpc": 120.0,
                "r_in_kpc": 2.5,
                "domain_dims": [250,250,500],
                "num_snapshots": 100,
                "target_theta": None,
                "build_mode": "sphere"}

    # construct template camera
    template_camera = Camera()                          # see pysrc/cudart.py for class documentation
    template_camera.tilt = (90.0 / 180) * np.pi         # tilt from bias vector (aligned with z axis)
    template_camera.t_obs = 0.5                         # overwritten to even spacing in t_obs for lookback            
    template_camera.phi = epsilon                       # small value, system axisymmetric in phi
    template_camera.theta = 0.25 * np.pi + epsilon      # overwritten to even spacing in theta for no-lookback
    template_camera.length_X = 1.0                      # longest simulation side length is 1.0 in code units
    template_camera.length_Y = 1.0                      # square domain
    template_camera.num_pixels_X = 2048                 # ensure square pixels
    template_camera.num_pixels_Y = 2048

    """
    camera_ags
    Dictionary for render parameters

    num_img             number of images to render
    resize_img          adjust image dimensions to viewing orientation
    relativistic        boolean to include relativistic boosting
    template            template camera to copy properties from
    camera_file_name    if not None, generate peristent text file with camera properties
    save_fig            boolean to generate .png figures from raw .npy images
    snapshot_index      index of simulation data to load (only for rendering without lookback, defaults to midpoint)
    """

    camera_args = {"num_img": 100,
                    "resize_img": True,
                    "relativistic": False,
                    "template": template_camera,
                    "camera_file_name": None,
                    "save_fig": True,
                    "snapshot_index": None}

    # handle command line arguments for regression tests
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "-build",
                        action="store_true",
                        default=False,
                        help="build regression suite data")
    parser.add_argument("-bl", "-build_labelled",
                        action="store_true",
                        default=False,
                        help="build labelled regression suite data")
    parser.add_argument("-r", "-render",
                        action="store_true",
                        default=False,
                        help="run in no-lookback test")
    parser.add_argument("-rl", "-render_lookback",
                        action="store_true",
                        default=False,
                        help="run in lookback mode")
    parser.add_argument("-rc", "-render_comp",
                        action="store_true",
                        default=False,
                        help="run penrose-terrell test")
    parser.add_argument("-v", "-verbose",
                        action="store_true",
                        default=False,
                        help="run in verbose mode")
    parser.add_argument("--save_dir",
                        default=None,
                        help="path to raw/figure outputs")
    parser.add_argument("--data_dir",
                        default=None,
                        help="path to input datasets")
    parser.add_argument("-p", "-profile",
                        action="store_true",
                        default=False,
                        help="generate profiling report from output")
    parser.add_argument("--build_mode",
                        default=None,
                        help="build mode for simulation data (sphere, sphere_rest, jet)")
    args = vars(parser.parse_args())

    # check valid build_mode
    if args["build_mode"] is not None:
        build_mode = args["build_mode"].lower()
        if build_mode not in ["sphere", "sphere_rest", "jet"]:
            raise Exception("build_mode option must be one of [sphere, sphere_rest, jet]")
        else:
            sim_args["build_mode"] = build_mode

    # except multiple build-type flags
    if (args["b"] + args["bl"] > 1):
        raise Exception("build routines share write space (data_dir), please select only one at a time") 

    # except multiple run-type flags
    if (args["r"] + args["rl"] + args["rc"] > 1):
        raise Exception("render routines share write space (save_dir), please select only one at a time")
    
    # construct regression data suite with or without labels
    if (args["b"]):
        if (args["data_dir"] is None):
            raise Exception("unable to build unlabelled regression suite data without save location (use --data_dir)")
        build_unlabelled_regression_suite(args["data_dir"], sim_args, args["v"])
    elif (args["bl"]):
        if (args["data_dir"] is None):
            raise Exception("unable to build labelled regression suite data without save location (use --data_dir)")
        build_labelled_regression_suite(args["data_dir"], sim_args, args["v"])

    # run render routine with or without lookback, or by comparison
    if (args["r"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run no-lookback test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run no-lookback test without load location (use --data_dir)")
        render_without_lookback(load_dir = args["data_dir"], 
                            save_dir = args["save_dir"], 
                            camera_args = camera_args, 
                            verbose = args["v"])
    elif (args["rl"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run lookback test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run lookback test without load location (use --data_dir)")
        render_with_lookback(load_dir = args["data_dir"], 
                            save_dir = args["save_dir"], 
                            sim_args = sim_args, 
                            camera_args = camera_args, 
                            verbose = args["v"])
    elif (args["rc"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run penrose-terrell test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run penrose-terrell test without load location (use --data_dir)")
        compare_lookback(load_dir = args["data_dir"], 
                                save_dir = args["save_dir"], 
                                sim_args = sim_args, 
                                camera_args = camera_args, 
                                verbose = args["v"])

    # report profiling for earlier run
    if (args["p"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run profiler summary without save location (use --save_dir)")
        report_profiling(save_dir = args["save_dir"], verbose = args["v"])