"""
This file contains a collection of functions to build mock data sets, and render images with or without the lookback routine
"""

# external imports
import sys, os
import numpy as np
import argparse, re
import time
import matplotlib.patches as patches

# local import
pysrc = os.path.join(os.path.dirname(__file__), "..", "pysrc")
sys.path.append(pysrc)
from cudart import *

def build_helical_snapshots(save_dir, sim_args, verbose = True):

    """
    Generate a dataset featuring twin ejecta moving in opposite directions
    Each ejectum moves on a helical path, with an outer pitch angle phi
    The cylindrical radius of the helix is related to the ejecta radius by a factor r_fac
    """

    # setup sim properties
    r_fac = 2                                               # radius of helix is this factor of the ejectum radius
    R_in_kpc = sim_args["r_in_kpc"] * r_fac
    phi = np.pi / 4                                         # helix pitch angle at outer helical edge (R + r)
    beta_total = np.sqrt(1 - 1.0 / sim_args["Gamma"] ** 2)  # lorentz factor of fastest moving component (outer edge)
    beta_xy = beta_total * np.cos(phi)                      # azimuthal velocity projection
    beta_z = beta_total * np.sin(phi)                       # velocity projection along rotation axis

    R_in_m = R_in_kpc * kpc_to_m
    r_in_m = sim_args["r_in_kpc"] * kpc_to_m
    omega = beta_xy * c_light / (R_in_m + r_in_m)           # solid body angular frequency (rad/s)

    # calculate sim duration
    vz_in_kpc_per_Myr = beta_z * c_light / (kpc_to_m / Myr_to_s)                         # cast to astro units
    T_in_Myr = 0.5 * (sim_args["L_in_kpc"] + sim_args["r_in_kpc"]) / vz_in_kpc_per_Myr   # calc duration to reach domain edge
    
    # cast to code units
    r_in_code = sim_args["r_in_kpc"] / sim_args["L_in_kpc"]                     # spatial length unit L_domain         
    R_in_code = R_in_kpc / sim_args["L_in_kpc"]                                 # spatial length unit L_domain
    vz_in_code = vz_in_kpc_per_Myr / sim_args["L_in_kpc"]                       # velocity scale L_domain/Myr               
    omega_in_code = omega * Myr_to_s                                            # temporal time unit Myr
    emm_adv = 1.0
    emm_rec = 1.0

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
    R_cyl = np.sqrt(xy_sqr)
    snapshot_size = np.size(xx)
    if (verbose): print("built empty mesh.")

     # determine cadence, if critical timing routine flagged
    if sim_args["target_theta"] is not None:
        crit_fac = (1 - beta_total * np.cos(sim_args["target_theta"])) / np.sin(sim_args["target_theta"])
        dt_crit_in_s = crit_fac * sim_args["r_in_kpc"] * kpc_to_m / (beta_total * c_light)
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

    # build snapshots
    for n, t_in_Myr in enumerate(t_span):
        # unlabelled data is a single .npy file, without a header
        save_str = os.path.join(save_dir, "snapshot" + str(n).zfill(5) + ".npy")
        
        # build data array for this snapshot in time
        save_data = np.zeros_like(xx)
        
        # identify position of blob centers
        adv_phase = omega_in_code * t_in_Myr                            # prograde rotation at omega
        rec_phase = np.pi + adv_phase                                   # offset rec phase by pi
        
        x_adv_c = R_in_code * np.cos(adv_phase)
        y_adv_c = R_in_code * np.sin(adv_phase)
        z_adv_c = vz_in_code * t_in_Myr            
        
        x_rec_c = R_in_code * np.cos(rec_phase)  
        y_rec_c = R_in_code * np.sin(rec_phase)
        z_rec_c = -vz_in_code * t_in_Myr                                # rec travels in opposite direction

        dr_adv_sqr = (xx - x_adv_c) ** 2 + (yy - y_adv_c) ** 2 + (zz - z_adv_c) ** 2
        dr_rec_sqr = (xx - x_rec_c) ** 2 + (yy - y_rec_c) ** 2 + (zz - z_rec_c) ** 2

        in_adv = (dr_adv_sqr < r_in_code ** 2)
        in_rec = (dr_rec_sqr < r_in_code ** 2)

        # build masks
        adv_emm_mask = (in_adv) & (ii == 0)
        adv_vx_mask = (in_adv) & (ii == 1)
        adv_vy_mask = (in_adv) & (ii == 2)
        adv_vz_mask = (in_adv) & (ii == 3)
        rec_emm_mask = (in_rec) & (ii == 0)
        rec_vx_mask = (in_rec) & (ii == 1)
        rec_vy_mask = (in_rec) & (ii == 2)
        rec_vz_mask = (in_rec) & (ii == 3)
        
        save_data[adv_emm_mask] = emm_adv
        save_data[rec_emm_mask] = emm_rec

        save_data[adv_vx_mask] = -omega_in_code * R_cyl[adv_vx_mask] * np.sin(adv_phase)
        save_data[rec_vx_mask] = -omega_in_code * R_cyl[rec_vx_mask]* np.sin(rec_phase)

        save_data[adv_vy_mask] = omega_in_code * R_cyl[adv_vy_mask] * np.cos(adv_phase)
        save_data[rec_vy_mask] = omega_in_code * R_cyl[rec_vy_mask] * np.cos(rec_phase)

        save_data[adv_vz_mask] = beta_z
        save_data[rec_vz_mask] = -beta_z

        save_data = save_data.astype(np.float32)                                # ENSURE cast to float32!!!
        np.save(save_str, save_data)                                            # save snapshot data
        if (verbose): print("built dataset for snapshot {0}/{1}".format(n,num_snapshots))

    if (verbose): print("finished dataset construction.")



if __name__ == "__main__":

    sim_args = {"Gamma": 2.0,
                "L_in_kpc": 120.0,
                "r_in_kpc": 2.5,
                "domain_dims": [250,250,500],
                "num_snapshots": 100,
                "target_theta": None}

    save_dir = "/mnt/kocsis2/hww27/cuDART_wdir/regression/helical_data"
    build_helical_snapshots(save_dir, sim_args)