# generic imports
import numpy as np
import matplotlib.pyplot as plt
import subprocess, os, sys, copy

# local import
sys.path.append("..")
from pysrc import *

epsilon = 1e-2 # small number to avoid casts with exact cooordinate alignment

def build_labelled_example():

    npy_load_str = os.path.join(host_dir, "inputs/sn_alt.npy")

    data_dir = os.path.join(host_dir, "inputs/mesh_demo")

    mesh = Mesh(data_dir)
    mb_data = np.load(npy_load_str)
    xl = np.array([-0.5,-0.5,-0.5])
    xr = np.array([0.5,0.5,0.5])
    xl_zoom = np.array([-0.25,-0.25,-0.25])
    xr_zoom = np.array([0.25,0.25,0.25])
    mesh.add_meshblock(mb_data, xl, xr)
    mesh.add_meshblock(mb_data, xl_zoom, xr_zoom)
    mesh.write_header()

def render_labelled_example():

    print("cuDART: starting labelled render example...")

    data_dir = os.path.join(host_dir, "inputs/mesh_demo")
    npy_save_str = os.path.join(host_dir, "outputs/labelled/raw")
    png_save_str = os.path.join(host_dir, "outputs/labelled/img")

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 512
    template_camera.num_pixels_Y = 512
    template_camera.tilt = 0
    template_camera.length_X = 0.66
    template_camera.length_Y = 0.66

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

    scene = Scene(data_dir, npy_save_str, cameras)
    print("built scene")

    scene.render(verbose = True)
    print("finished rendering raw images")

    scene.plot(png_save_str, verbose = True, remove_raw_images = True)
    print("unlablled render example finished.")

def render_unlabelled_example():

    print("cuDART: starting unlabelled render example...")

    # define targets
    npy_load_str = os.path.join(host_dir, "inputs/sn_alt.npy")
    npy_save_str = os.path.join(host_dir, "outputs/unlabelled/raw")
    png_save_str = os.path.join(host_dir, "outputs/unlabelled/img")

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 512
    template_camera.num_pixels_Y = 512
    template_camera.tilt = 0
    template_camera.length_X = 0.66
    template_camera.length_Y = 0.66

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
    scene.render(verbose = True)
    print("finished rendering raw images")

    scene.plot(png_save_str, verbose = True, remove_raw_images = True)
    print("finished rendering rasterised images")

    print("unlablled render example finished.")

if __name__ == "__main__":

    render_unlabelled_example()