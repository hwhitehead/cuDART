# external imports
import sys, os
import numpy as np
import argparse, re

# local import
pysrc = os.path.join(os.path.dirname(__file__), "..", "pysrc")
sys.path.append(pysrc)
from cudart import *

# define units
epislon = 1e-2 # small angle to avoid casts along coordinate axes
kpc_to_m = 1e3 * 3.086e+16
Myr_to_s = 1e6 * 365 * 24 * 60 * 60
c_light = 3e8

def build_regression_suite(save_dir, sim_args, verbose=True):

    # construct template data for regression suite
    # the template data features twin emitting regions travelling at a fixed velocity in opposite directions
    # the emitting regions are spheres in the observer frame
    # the emission in the spheres falls off quadratically with radius, out to a fixed, finite radius

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

    # build header data
    header_str = os.path.join(save_dir, "header.txt")
    t_span = np.linspace(0, T_in_Myr, sim_args["num_snapshots"])                # evenly snapshot times over duration
    with open(header_str, "w") as f:
        f.write("{0} {1} {2} {3}".format(sim_args["num_snapshots"], snapshot_size, t_span[1], sim_args["L_domain"]))
    if (verbose): print("built header.")

    # build snapshots
    for n, t_in_Myr in enumerate(t_span):
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
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
        np.save(save_str, save_data)                                            # save snapshot data
        if (verbose): print("built dataset for snapshot {0}".format(n))
    
    if (verbose): print("finished dataset construction.")

def run_nolookback_test(load_dir, save_dir, sim_args, camera_args, verbose=True):

    # run a rendering test on a single snapshot of data
    # data should be loaded from suite built using build_regression_suite (-b flag)
    # render uses a set number of cameras evenly spanning the azimuthal axis 
    # optional call included to render raws as figures (.npy -> .png)

    if (verbose): 
        print("starting no-lookback render test...")
        print("reading data from {0}".format(load_dir))
        print("saving data at {0}".format(save_dir))

    # check user input
    if camera_args["snapshot_index"] is None: # select middle snapshot
        snapshot_index = int(np.floor(0.5 * (sim_args["num_snapshots"] - 1)))
        print("auto setting snapshot_index = {0}".format(snapshot_index))
    elif (camera_args["snapshot_index"] < 0): # check snapshot lower oob
        print("selected snapshot negative, using first snapshot.")
    else: # check snapshot upper oob
        last_index = sim_args["num_snapshots"] - 1
        if sim_args["snapshot_index"] > last_index:
            print("selected snapshot exceeds simulation bounds, using last ({0})".format(last_index))
            snapshot_index = last_index

    # check input, output directory existence
    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(save_dir))

    # format input and output paths
    npy_load_str = os.path.join(load_dir, "snapshot" + str(int(snapshot_index)).zfill(5) + ".npy")
    npy_save_str = os.path.join(save_dir, "raw")
    png_save_str = os.path.join(save_dir, "img")

    # check for specific snapshot input file
    if not os.path.exists(npy_load_str):
        raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(npy_load_str))

    # prepare array of cameras (cycle in theta)
    if (verbose): print(r"building {0} cameras evenly spanning theta in [0,pi]".format(camera_args["num_img"]))
    theta_ar = np.linspace(epsilon, np.pi - epsilon,camera_args["num_img"])
    cameras = []
    for i, theta in enumerate(theta_ar):
        camera = copy.deepcopy(camera_args["template"])
        camera.theta = theta
        if (camera_args["resize_img"]):
            camera.length_X = camera_args["template"].length_X * np.sin(theta)
            camera.length_Y = camera_args["template"].length_Y * np.sin(theta)
        camera.set_sph_pos(r = 2.0, target_origin = True)
        cameras.append(camera)    
        if (verbose): print("built camera {0} at theta = {1:.2f}deg...".format(i,theta*180.0/np.pi))
    if (verbose): print("finished camera initialisation.")

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras, camera_file_name=camera_args["camera_file_name"])
    if (verbose): print("built scene.")

    # render and save images
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = False)
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(png_save_str, cmap = "afmhot", verbose = verbose, remove_raw_images = False, vmin=-6, vmax=0)
        print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

def run_lookback_test(load_dir, save_dir, sim_args, camera_args, verbose=True):

    # run a rendering test using finite speed of light with multiple snapshots
    # data should be loaded from suite built using build_regression_suite (-b flag)
    # render uses a set number of cameras evenly spaced in observer time
    # optional call included to render raws as figures (.npy -> .png)

    if (verbose): 
        print("starting lookback render test...")
        print("reading data from {0}".format(load_dir))
        print("saving data at {0}".format(save_dir))

    # check input, output directory existence
    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(save_dir))

    # format input and output paths
    npy_load_str = load_dir # input is entire data directory
    npy_save_str = os.path.join(save_dir, "raw")
    png_save_str = os.path.join(save_dir, "img")

    # check for ALL snapshot input files
    for n in range(0, sim_args["num_snapshots"]):
        snapshot_str = os.path.join(npy_load_str, "snapshot" + str(n).zfill(5) + ".npy")
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
    if (verbose): print(r"building {0} cameras evenly spanning observer time in [{1},{2}]".format(camera_args["num_img"],t_min, t_max))
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
        if (verbose): print("built camera {0} at t_obs = {1:.3f}...".format(i,t_obs))
    if (verbose): print("finished camera initialisation.")

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras, camera_file_name=camera_args["camera_file_name"])
    if (verbose): print("built scene.")

    # render and save images
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"], lookback = True)
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(png_save_str, cmap = "afmhot", verbose = verbose, remove_raw_images = False, vmin=-6, vmax=0)
        print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

if __name__ == "__main__":

    # dict for simulation args (all lengths in kpc)
    sim_args = {"Gamma": 2.0,
                "L_domain": 120.0,
                "r_blob": 2.5,
                "domain_dims": [100,100,200],
                "num_snapshots": 100}

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
    parser.add_argument("-r", 
                        action="store_true",
                        default=False,
                        help="run in no-lookback test")
    parser.add_argument("-rl", 
                        action="store_true",
                        default=False,
                        help="run in lookback mode")
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
    
    if (args["r"] and args["rl"]):
        raise Exception("no-lookback and lookback share write space, please select only one")

    if (args["b"]):
        if (args["data_dir"] is None):
            raise Exception("unable to build regression suite data without save location (use --data_dir)")
        build_regression_suite(args["data_dir"], sim_args, args["v"])
    
    if (args["r"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run no-lookback test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run no-lookback test without load location (use --data_dir)")
        run_nolookback_test(args["data_dir"], args["save_dir"], sim_args, camera_args, args["v"])
    
    if (args["rl"]):
        if (args["save_dir"] is None):
            raise Exception("unable to run lookback test without save location (use --save_dir)")
        if (args["data_dir"] is None):
            raise Exception("unable to run lookback test without load location (use --data_dir)")
        run_lookback_test(args["data_dir"], args["save_dir"], sim_args, camera_args, args["v"])