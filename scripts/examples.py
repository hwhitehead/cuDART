# generic imports
import numpy as np
import matplotlib.pyplot as plt
import subprocess, os, sys, copy

# local import
sys.path.append("..")
from pysrc import *

def build_mesh():

    npy_load_str = os.path.join(host_dir, "inputs/sn_alt.npy")

    data_dir = os.path.join(host_dir, "inputs/mesh_demo")

    mesh = Mesh(data_dir)
    mb_data = np.load(npy_load_str)
    xl = np.array([-0.5,-0.5,-0.5])
    xr = np.array([0.5,0.5,0.5])
    xl_zoom = np.array([-0.25,-0.25,-0.25])
    xr_zoom = np.array([0.25,0.25,0.25])
    mesh.add_meshblock(mb_data, xl, xr)
    mesh.add_meshblock(mb_data, xl_zoom, zr_zoom)
    mesh.write_header()

def render_from_mesh(verbose=True, remove_data=True):

    data_dir = os.path.join(host_dir, "inputs/mesh_demo")
    npy_save_str = os.path.join(host_dir, "inputs/mesh_demo")
    png_save_str = os.path.join(host_dir, "outputs/mesh_demo/img")

    # build camera
    camera = Camera()
    camera.num_pixels_X = 2048
    camera.num_pixels_Y = 2048
    camera.tilt = (-38.0/180) * np.pi
    camera.length_X = 0.66
    camera.length_Y = 0.66
    cameras = [camera]

    scene = Scene(data_dir, npy_save_str, cameras)

    scene.render(verbose=verbose)
    scene.plot(png_save_str, verbose=verbose, remove_data=remove_data)

def demo_scene_gen(num_img = 5, verbose=True, remove_data=True):

    # define targets
    npy_load_str = os.path.join(host_dir, "inputs/sn_alt.npy")
    npy_save_str = os.path.join(host_dir, "outputs/mesh/sn")
    png_save_str = os.path.join(host_dir, "outputs/mesh/sn")

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 2048
    template_camera.num_pixels_Y = 2048
    template_camera.tilt = (-38.0/180) * np.pi
    template_camera.length_X = 0.66
    template_camera.length_Y = 0.66

    # build camera array, inherit from template
    ep = 1e-4 # avoid purely aligned casts
    theta_ar = np.linspace(ep,np.pi - ep,num_img, endpoint=False)
    cameras = []
    for theta in theta_ar:
        camera = copy.deepcopy(template_camera)
        camera.theta = theta
        cameras.append(camera)

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras)

    # render and save images
    scene.render(verbose=verbose)
    scene.plot(png_save_str, verbose=verbose, remove_data=True)

if __name__ == "__main__":

    build_mesh()
    render_from_mesh()