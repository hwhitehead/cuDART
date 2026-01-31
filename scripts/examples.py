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
    print("lablled render example finished.")

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

def guided_camera_example():

    checkpoints = np.array([[1, 1, 0], [2, 2, 0], [3, 2, 0], [3, 4, 0], [1, 3, 0], [-1, 2, 0], [-1, 0, 0], [2, 0, 0]])
    target = np.array([[0,1,0]])
    gcam = GuidedCamera(checkpoints = checkpoints, targets=target)

    num_imgs = 10

    camera_times = np.linspace(0, 1, num_imgs)
    cameras = gcam.generate_cameras(num_img = num_imgs, camera_times = camera_times, mode="chord")

    set_plot_defaults()
    fig = plt.figure()
    gs = fig.add_gridspec(1,2,width_ratios=np.array([1,0.05]))
    ax = fig.add_subplot(gs[:,0])
    cax = fig.add_subplot(gs[:,1])

    # plot checkpoints
    ax.scatter(gcam.checkpoints[:, 0], gcam.checkpoints[:, 1], color='k',marker='x', label="Checkpoints")  # x-y plot

    # plot origin spline
    t_span = np.linspace(0,1,100)
    origin_spline_guide = gcam.origin_spline.eval_spline(t_span)
    ax.plot(origin_spline_guide[:,0], origin_spline_guide[:,1], color='k', linestyle = "dotted", zorder=-5, label="Guide")

    normal_len = 0.2
    ax.scatter([],[],color='r', label="Cameras")
    ax.scatter(target[0, 0], target[0, 1], color='b', label="Target")
    arrow_kwargs = {"head_width": 0.05, "width": 0.02, "zorder": -10}
    for camera in cameras:
        # plot camera positions
        ax.scatter(camera.origin[0], camera.origin[1], color='r')
        # add normals
        ax.arrow(camera.origin[0], camera.origin[1], camera.normal[0] * normal_len, camera.normal[1] * normal_len, color='b', **arrow_kwargs)

    ax.legend(loc="upper left", frameon=False)
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig("guided_cam.png", dpi=300, bbox_inches="tight")
    plt.close("all")

def render_jet(relativistic=false):

    print("cuDART: starting jet render example...")

    # define targets
    npy_load_str = "/mnt/kocsis1/cuDART_wdir/snapshot0490_mirror_relativistic.npy"
    npy_save_str = "/mnt/kocsis1/cuDART_wdir/raw"
    png_save_str = "/mnt/kocsis1/cuDART_wdir/img"

    # build template camera
    template_camera = Camera()
    template_camera.num_pixels_X = 1024
    template_camera.num_pixels_Y = 1024
    template_camera.tilt = (60.0 / 180) * np.pi
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
    # theta = 0.5 * np.pi + epsilon    
    # phi_ar = np.linspace(epsilon,2 * np.pi - epsilon,num_img, endpoint=False)
    # cameras = []
    # for phi in phi_ar:
    #     camera = copy.deepcopy(template_camera)
    #     camera.set_sph_pos(r = 2.0, theta = theta, phi = phi, target_origin = True)
    #     cameras.append(camera)
    print("initialised cameras")

    # generate scene
    scene = Scene(npy_load_str, npy_save_str, cameras)
    print("built scene")

    # render and save images
    scene.render(verbose = True, relativistic = relativistic)
    print("finished rendering raw images")

    scene.plot(png_save_str, cmap = "afmhot", verbose = True, remove_raw_images = True, vmin=17, vmax=20)
    print("finished rendering rasterised images")

    print("unlablled render example finished.")

if __name__ == "__main__":

    render_jet(True)