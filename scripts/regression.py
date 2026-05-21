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
    if (verbose): print("starting regression suite data construction...")

    if not os.path.isdir(save_dir):
        raise Exception("{0} does not exist".format(save_dir))

    # define simulation parameters
    v_in_c = np.sqrt(1 - 1.0 / sim_args["Gamma"] ** 2)
    v_in_kpc_per_Myr = v_in_c * c_light / (kpc_to_m / Myr_to_s)
    r_blob_in_code = sim_args["r_blob"] / sim_args["L_domain"]
    T_in_Myr = 0.5 * sim_args["L_domain"] / v_in_kpc_per_Myr # duration to reach domain edge
    
    # build empty domain 
    max_emm = 1.0
    Lz = sim_args["L_domain"]
    Ly = (Lz * sim_args["domain_dims"][1]) / sim_args["domain_dims"][1] 
    Lx = (Lz * sim_args["domain_dims"][1]) / sim_args["domain_dims"][1] 
    xspan = np.linspace(-0.5 * Lx, 0.5 * Lz, sim_args["domain_dims"][0])
    yspan = np.linspace(-0.5 * Ly, 0.5 * Ly, sim_args["domain_dims"][1])
    zspan = np.linspace(-0.5 * Lz, 0.5 * Lz, sim_args["domain_dims"][2])
    ispan = np.array([0,1,2,3]) 
    xx, yy, zz, ii = np.meshgrid(xspan, yspan, zspan, ispan, indexing="ij")
    xy_sqr = xx ** 2 + yy ** 2
    snapshot_size = np.size(xx)
    if (verbose): print("built empty mesh.")

    # build header data
    header_str = os.path.join(save_dir, "header.txt")
    t_span = np.linspace(0, T_in_Myr, sim_args["num_snapshots"]) # evenly space over duration
    with open(header_str, "w") as f:
        f.write("{0} {1} {2} {3}".format(sim_args["num_snapshots"], snapshot_size, t_span[1], sim_args["L_domain"]))
    if (verbose): print("built header.")

    # build snapshots
    for n, t_in_Myr in enumerate(t_span):
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        save_data = np.zeros_like(xx)
        lead_center_in_kpc = t_in_Myr * v_in_kpc_per_Myr
        lead_center = lead_center_in_kpc / sim_args["L_domain"] # cast to code units
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
        if (verbose): print("built dataset for snapshot {0}".format(n))
    
    if (verbose): print("finished dataset construction.")

def run_nolookback_test(load_dir, save_dir, sim_args, camera_args, verbose=True):

    # run a rendering test on a single snapshot of data
    if (verbose): print("starting no-lookback render test...")

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

    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(save_dir))

    npy_load_str = os.path.join(load_dir, "snapshot" + str(int(snapshot_index)).zfill(5) + ".npy")
    npy_save_str = os.path.join(save_dir, "raw")
    png_save_str = os.path.join(save_dir, "img")

    if not os.path.exists(npy_load_str):
        raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(npy_load_str))

    # prepare array of cameras (cycle theta)
    theta_ar = np.linspace(epsilon, np.pi - epsilon,camera_args["num_img"])
    cameras = []
    for i, theta in enumerate(theta_ar):
        camera = copy.deepcopy(camera_args["template"])
        camera.theta = theta
        if (camera_args["resize_img"]):
            camera.length_X = camera_args["template"].length_X * np.sin(theta)
            camera.length_Y = camera_args["template"].length_Y * np.sin(theta)
        camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)
        cameras.append(camera)    
        if (verbose): print("built camera {0}...".format(i))
    if (verbose): print("finished camera initialisation.")

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras, camera_file_name=camera_args["camera_file_name"])
    if (verbose): print("built scene.")

    # render and save images
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"])
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(png_save_str, cmap = "afmhot", verbose = verbose, remove_raw_images = remove_raw_images, vmin=-6, vmax=0)
        print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

def run_lookback_test(load_dir, save_dir, sim_args, camera_args, verbose=True):

    # run a rendering test on multiple snapshot of data (TODO: populate this)
    if (verbose): print("starting lookback render test...")

    # check user input
    if camera_args["snapshot_index"] is None:
        snapshot_index = np.floor(0.5 * (sim_args["num_snapshots"] - 1))
        print("auto setting snapshot_index = {0}".format(snapshot_index))
    else:
        last_index = sim_args["num_snapshots"] - 1
        if sim_args["snapshot_index"] > last_index:
            print("selected snapshot exceeds simulation bounds, using last ({0})".format(last_index))
            snapshot_index = last_index

    for path in [load_dir, save_dir]:
        if not os.path.isdir(path):
            raise Exception("{0} does not exist".format(save_dir))

    npy_load_str = os.path.join(load_dir, "snapshot" + str(snapshot_index).zfill(5) + ".npy")
    npy_save_str = os.path.join(save_dir, "raw")
    png_save_str = os.path.join(save_dir, "img")

    if not os.path.exists(npy_load_str):
        raise Exception("no file found at {0}, did you forget to build dataset with -b before?".format(npy_load_str))

    # prepare array of cameras (cycle theta)
    theta_ar = np.linspace(epsilon, np.pi - epsilon,num_img)
    cameras = []
    for i, theta in enumerate(theta_ar):
        camera = copy.deepcopy(camera_args["template"])
        camera.theta = theta
        if (camera_args["resize_img"]):
            camera.length_X = camera_args["template"].length_X * np.sin(theta)
            camera.length_Y = camera_args["template"].length_Y * np.sin(theta)
        camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)    
        if (verbose): print("built camera {0}...".format(i))
    if (verbose): print("finished camera initialisation.")

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras, camera_file_name=camera_args["camera_file_name"])
    if (verbose): print("built scene.")

    # render and save images
    scene.render(verbose = verbose, relativistic = camera_args["relativistic"])
    if (verbose): print("finished rendering raw images.")

    if (camera_args["save_fig"]):
        scene.plot(png_save_str, cmap = "afmhot", verbose = verbose, remove_raw_images = remove_raw_images, vmin=-6, vmax=0)
        print("finished rendering figures.")

    if (verbose): print("finished no-lookback test, see {0} for output".format(save_dir))

if __name__ == "__main__":

    # dict for simulation args (all lengths in kpc)
    sim_args = {"Gamma": 2.0,
                "L_domain": 120.0,
                "r_blob": 2.5,
                "domain_dims": [100,100,200],
                "num_snapshots": 10}

    # template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = (60.0 / 180) * np.pi
    template_camera.t_obs = 0.5 # in units of Myr
    phi = epsilon
    theta = np.pi / 2 + epsilon
    template_camera.length_X = 1.0 * np.sin(theta) # size window to fit initial orientation
    template_camera.length_Y = 1.0 * np.sin(theta) # longest simulation length 1.0 in code units

    # dict for camera args
    camera_args = {"num_img": 10,
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