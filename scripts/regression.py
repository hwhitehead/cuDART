"""
This file contains a collection of functions to build mock data sets, and render images with or without the lookback routine
"""

# external imports
import sys, os
import numpy as np
import argparse, re

# local import
pysrc = os.path.join(os.path.dirname(__file__), "..", "pysrc")
sys.path.append(pysrc)
from cudart import *

def build_unlabelled_regression_suite(save_dir, sim_args, verbose = True, sphere_in_rest = False):

    # construct template data for regression suite, without labels
    # the template data features twin emitting regions travelling at a fixed velocity in opposite directions
    # the emitting regions are spheres in the observer frame
    # the emission in the spheres falls off quadratically with radius, out to a fixed, finite radius
    # if given specific theta, ensure sampling occurs at unaliased frequency

    if (verbose): 
        print("starting regression suite data construction...")
        print("saving data at {0}".format(save_dir))

    if not os.path.isdir(save_dir):
        raise Exception("{0} does not exist".format(save_dir))

    # define simulation parameters
    v_in_c = np.sqrt(1 - 1.0 / sim_args["Gamma"] ** 2)                          # calculate ejecta velocity
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)                 # cast to astro units
    r_blob_in_code = sim_args["r_blob"] / sim_args["L_domain"]                  # cast to code units (where L_domain = 1.0)
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

    # determine cadence, if critical timing routine flagged
    if sim_args["target_theta"] is not None:
        crit_fac = (1 - v_in_c * np.cos(sim_args["target_theta"])) / np.sin(sim_args["target_theta"])
        dt_crit_in_s = crit_fac * sim_args["r_blob"] * kpc_to_m / (v_in_c * c_light)
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

    if sphere_in_rest:
        z_scale = 1.0 / sim_args["Gamma"]
    else:
        z_scale = 1.0

    # build snapshots
    for n, t_in_Myr in enumerate(t_span):
        # unlabelled data is a single .npy file, without a header
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        
        # build data array for this snapshot in time
        save_data = np.zeros_like(xx)
        lead_center_in_kpc = t_in_Myr * v_in_kpc_per_Myr                        # calculate ejecta position in kpc
        lead_center = lead_center_in_kpc / sim_args["L_domain"]                 # cast to code units
        tail_center = -lead_center                                              # tailing ejecta symmetric in x-y
        lead_ZZ = zz - lead_center
        tail_ZZ = zz - tail_center

        dz_lead = (zz - lead_center) / z_scale
        dz_tail = (zz - tail_center) / z_scale

        rr_lead_sqr = (dz_lead ** 2 + xy_sqr) / r_blob_in_code ** 2             # sph radius from leading/tailing ejecta
        rr_tail_sqr = (dz_tail ** 2 + xy_sqr) / r_blob_in_code ** 2

        in_lead = (rr_lead_sqr < 1)
        in_tail = (rr_tail_sqr < 1)
        emm_lead = max_emm * (1.0 - rr_lead_sqr)                                # emission falls off quadratically from center
        emm_tail = max_emm * (1.0 - rr_tail_sqr)

        lead_emm_mask = (in_lead) & (ii == 0)
        tail_emm_mask = (in_tail) & (ii == 0)
        lead_vel_mask = (in_lead) & (ii == 3)
        tail_vel_mask = (in_tail) & (ii == 3)
        save_data[lead_emm_mask] = emm_lead[lead_emm_mask]                      # match emission in lead/tail
        save_data[tail_emm_mask] = emm_tail[tail_emm_mask]  
        save_data[lead_vel_mask] = v_in_c                                       # invert velocity in lead/tail
        save_data[tail_vel_mask] = -v_in_c
        save_data = save_data.astype(np.float32)                                # ENSURE cast to float32!!!
        np.save(save_str, save_data)                                            # save snapshot data
        if (verbose): print("built dataset for snapshot {0}/{1}".format(n,num_snapshots))

    if (verbose): print("finished dataset construction.")

def build_labelled_regression_suite(save_dir, sim_args, verbose=True, sphere_in_rest = False):

    # TODO: test this deployment

    # construct template data for regression suite, with labels
    # the template data features twin emitting regions travelling at a fixed velocity in opposite directions
    # the emitting regions are spheres in the observer frame
    # the emission in the spheres falls off quadratically with radius, out to a fixed, finite radius
    # if given specific theta, ensure sampling occurs at unaliased frequency

    if (verbose): 
        print("starting regression suite data construction...")
        print("saving data at {0}".format(save_dir))

    if not os.path.isdir(save_dir):
        raise Exception("{0} does not exist".format(save_dir))

    # define simulation parameters
    v_in_c = np.sqrt(1 - 1.0 / sim_args["Gamma"] ** 2)                          # calculate ejecta velocity
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)                 # cast to astro units
    r_blob_in_code = sim_args["r_blob"] / sim_args["L_domain"]                  # cast to code units (where L_domain = 1.0)
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
        dt_crit_in_s = crit_fac * sim_args["r_blob"] * kpc_to_m / (v_in_c * c_light)
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
        lead_center_in_kpc = t_in_Myr * v_in_kpc_per_Myr                        # calculate ejecta position in kpc
        lead_center = lead_center_in_kpc / sim_args["L_domain"]                 # cast to code units
        tail_center = -lead_center                                              # tailing ejecta symmetric in x-y
        lead_ZZ = zz - lead_center
        tail_ZZ = zz - tail_center

        rr_lead_sqr = ((zz - lead_center) ** 2 + xy_sqr) / r_blob_in_code ** 2  # sph radius from leading/tailing ejecta
        rr_tail_sqr = ((zz - tail_center) ** 2 + xy_sqr) / r_blob_in_code ** 2

        in_lead = (rr_lead_sqr < 1)
        in_tail = (rr_tail_sqr < 1)
        emm_lead = max_emm * (1.0 - rr_lead_sqr)                                # emission falls off quadratically from center
        emm_tail = max_emm * (1.0 - rr_tail_sqr)

        lead_emm_mask = (in_lead) & (ii == 0)
        tail_emm_mask = (in_tail) & (ii == 0)
        lead_vel_mask = (in_lead) & (ii == 3)
        tail_vel_mask = (in_tail) & (ii == 3)
        save_data[lead_emm_mask] = emm_lead[lead_emm_mask]                      # match emission in lead/tail
        save_data[tail_emm_mask] = emm_tail[tail_emm_mask]  
        save_data[lead_vel_mask] = v_in_c                                       # invert velocity in lead/tail
        save_data[tail_vel_mask] = -v_in_c
        save_data = save_data.astype(np.float32)                                # ENSURE cast to float32!!!
        
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

def run_nolookback_test(load_dir, save_dir, camera_args, snapshot_index = None, num_snapshots = None, verbose = True):

    # run a rendering test on a single snapshot of data
    # data should be loaded from suite built using build_regression_suite (-b flag)
    # render uses a set number of cameras evenly spanning the azimuthal axis 
    # optional call included to render raws as figures (.npy -> .png)

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
            raise Exception("{0} does not exist".format(save_dir))

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
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = False, verbose_cpp = verbose)
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(fig_save_dir = save_dir, cmap = "afmhot", verbose = verbose, remove_raw_npy = False, vmin= -6, vmax = 0)
        if (verbose): print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

def run_lookback_test(load_dir, save_dir, sim_args, camera_args, verbose = True, flexload = False):

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
            raise Exception("{0} does not exist".format(save_dir))

    # check for ALL snapshot input files
    for n in range(0, sim_args["num_snapshots"]):
        snapshot_str = os.path.join(load_dir, "snapshot" + str(n).zfill(5) + ".npy")
        if not os.path.exists(snapshot_str):
            raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(snapshot_str))

    # collect data from args
    v_in_c = np.sqrt(1.0 - 1.0 / sim_args["Gamma"] ** 2)                                        # calculate velocity in units of c
    theta = camera_args["template"].theta                                                       # collect orientation from template

    # calculate start time (just before light from origin reaches camera)
    D_in_m = 2.0 * sim_args["L_domain"] * kpc_to_m                                              # origin-camera seperation 
    t_min_in_s = D_in_m / c_light                                                               # light flight time from origin to camera
    t_min = t_min_in_s / Myr_to_s                                                               # cast to astro/code units                 
    t_min *= 0.95                                                                               # start render just before flight time 

    # calculate stop time (when receding ejectum reaches maximal extent)
    x_max_in_m = 0.5 * sim_args["L_domain"] * np.sin(theta) * kpc_to_m                          # max obs blob displacement for given theta
    d_in_m = x_max_in_m * (1 + v_in_c * np.cos(theta)) / (v_in_c * np.sin(theta)) + D_in_m      # invert superluminal motion eq to calc flight time
    t_max_in_s = d_in_m / c_light                                                               # observer time when RECEDING blob reaches domain edge
    t_max = t_max_in_s / Myr_to_s                                                               # cast to astro/code units    

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
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = True, verbose_cpp = verbose, flexload = flexload)
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(fig_save_dir = save_dir, cmap = "afmhot", verbose = verbose, remove_raw_npy = False, vmin = -6, vmax = 0)
        print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

if __name__ == "__main__":

    # dict for simulation args (all lengths in kpc)
    sim_args = {"Gamma": 2.0,
                "L_domain": 120.0,
                "r_blob": 2.5,
                "domain_dims": [100,100,200],
                "num_snapshots": 100,
                "target_theta": None}

    # construct template camera
    template_camera = Camera()
    template_camera.tilt = (45.0 / 180) * np.pi     # default 45deg tilt from bias aligned with z axis 
    template_camera.t_obs = 0.5                     # overwritten to even spacing in t_obs for lookback            
    template_camera.phi = epsilon                   # small value, system axisymmetric in phi
    template_camera.theta = 0.25 * np.pi + epsilon  # overwritten to even spacing in theta for no-lookback
    template_camera.length_X = 1.0                  # longest simulation size 1.0 in code units
    template_camera.length_Y = 1.0                  # square domain
    template_camera.num_pixels_X = 2048             # ensure square pixels
    template_camera.num_pixels_Y = 2048

    # dict for camera args
    camera_args = {"num_img": 100,
                    "resize_img": False,
                    "relativistic": True,
                    "template": template_camera,
                    "camera_file_name": None,
                    "save_fig": True,
                    "snapshot_index": None}

    # handle command line arguments for regression tests
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        action="store_true",
                        default=False,
                        help="build regression suite data")
    parser.add_argument("-bl",
                        action="store_true",
                        default=False,
                        help="build labelled regression suite data")
    parser.add_argument("-bpt",
                        action="store_true",
                        default=False,
                        help="build as sphere in rest frame")
    parser.add_argument("-r", 
                        action="store_true",
                        default=False,
                        help="run in no-lookback test")
    parser.add_argument("-rl", 
                        action="store_true",
                        default=False,
                        help="run in lookback mode")
    parser.add_argument("-f",
                        action="store_true",
                        default=False,
                        help="run with flexload")
    parser.add_argument("-v", 
                        action="store_true",
                        default=False,
                        help="run in verbose mode")
    parser.add_argument("--save_dir",
                        default=None,
                        help="path to raw/figure outputs")
    parser.add_argument("--data_dir",
                        default=None,
                        help="path to input datasets")

    args = vars(parser.parse_args())
    
    # except multiple run-type flags
    if (args["r"] and args["rl"]):
        raise Exception("no-lookback and lookback share write space, please select only one")
    
    # except multiple build-type flags
    if (args["b"] and args["bl"]):
        raise Exception("labelled and unlabelled builders share write space, please select only one") 

    # construct regression data suite with or without labels
    if (args["b"]):
        if (args["data_dir"] is None):
            raise Exception("unable to build unlabelled regression suite data without save location (use --data_dir)")
        build_unlabelled_regression_suite(args["data_dir"], sim_args, args["v"], args["bpt"])
    elif (args["bl"]):
        if (args["data_dir"] is None):
            raise Exception("unable to build labelled regression suite data without save location (use --data_dir)")
        build_labelled_regression_suite(args["data_dir"], sim_args, args["v"], args["bpt"])

    # run render routine with or without lookback
    if (args["r"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run no-lookback test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run no-lookback test without load location (use --data_dir)")
        run_nolookback_test(load_dir = args["data_dir"], 
                            save_dir = args["save_dir"], 
                            camera_args = camera_args, 
                            verbose = args["v"])
    elif (args["rl"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run lookback test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run lookback test without load location (use --data_dir)")
        run_lookback_test(load_dir = args["data_dir"], 
                            save_dir = args["save_dir"], 
                            sim_args = sim_args, 
                            camera_args = camera_args, 
                            verbose = args["v"],
                            flexload=args["f"])