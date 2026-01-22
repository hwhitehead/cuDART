import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess
import glob

host_dir = "/mnt/users/hww27/cuDART"
str_zfill = 3

class Camera:
    """
    The Camera class is a basic struct to wrap the image viewing orientation and dimensions
    """
    def __init__(self, origin = np.array([1.0,0.0,0.0]), normal = np.array([-1.0,0.0,0.0]), bias = np.array([0.0,0.0,1.0]),
                    num_pixels_X = 512, num_pixels_Y = 512, length_X = 1.0, length_Y = 1.0, tilt = 0.0):
        self.origin = origin
        self.normal = normal
        self.bias = bias
        self.num_pixels_X = num_pixels_X
        self.num_pixels_Y = num_pixels_Y
        self.length_X = length_X
        self.length_Y = length_Y
        self.bias = bias
        self.tilt = tilt
        self.num_pixels = num_pixels_X * num_pixels_Y

    def header_str(self):
        return "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n".format(*self.origin, *self.normal, *self.bias, self.tilt, self.length_X, self.length_Y)
    
    def set_sph_pos(self, r, theta, phi, target_origin=False):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        self.origin = r * np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
        if (target_origin): self.set_target(np.array([0.0, 0.0, 0.0])) # target coordinate origin

    def set_cyl_pos(self, R, phi, z, target_origin=False):
        self.origin = np.array([R * np.cos(phi), R * np.sin(phi), z])
        if (target_origin): self.set_target(np.array([0.0, 0.0, 0.0])) # target coordinate origin

    def set_target(self, target):
        self.normal = target - self.origin

class Scene:
    """ 
    this class provides a simple way for the user to call cuDART and process the results
    """
    def __init__(self, load_str, save_str, cameras = None, camera_file_name = None): 

        # parse load/save strings        
        self.load_str = load_str
        self.save_str = save_str.removesuffix(".png")

        if cameras is None:
            self.cameras = [Camera()] # initialise single default camera
        elif isinstance(cameras, list) or isinstance(cameras, np.ndarray):
            self.cameras = cameras # pass list type input directly
        else:
            self.cameras = [cameras] # package singel type input
        self.camera_file_name = camera_file_name

        if self.camera_file_name is None:
            self.temp_camera_file = "temp_camera_file.txt"
        else:
            self.temp_camera_file = self.camera_file_name

        # check camera dimensions are const across all cameras
        self.num_pixels_X = cameras[0].num_pixels_X
        self.num_pixels_Y = cameras[0].num_pixels_Y
        for camera in self.cameras:
            if camera.num_pixels_X != self.num_pixels_X or camera.num_pixels_Y != self.num_pixels_Y:
                raise Exception("all Camera objects must have coherant image dimensions.")

    def build_camera_file(self):

        with open(self.temp_camera_file, "w") as f:
            # add header with const image dimensions (zero packed for istringstream read)
            f.write("{0} {1} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(self.cameras[0].num_pixels_X, self.cameras[0].num_pixels_Y))
            # add line for each camera to run render with
            for camera in self.cameras:
                f.write(camera.header_str())

    def make_clean(self):
        subprocess.run(["make","clean"], check = True)       

    def make(self):
        subprocess.run(["make"], check = True)

    def render(self, profile = False, verbose = False, check_make = True, force_make = False, plot = False, max_mem = None):

        # prepare camera space
        self.build_camera_file()
        if (verbose):
            print("generated camera file at " + self.temp_camera_file)

        # check savespace exists
        save_dir = os.path.dirname(self.save_str)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # check executable exists, or build
        path_to_executable = os.path.join(host_dir, "bin/cudart")
        if not os.path.isfile(path_to_executable):
            if (verbose):
                print("unable to located executable, forcing remake.")
            self.make()
        elif check_make:
            if verbose:
                print("checking for updates since last make")
            self.make()
        elif force_make:
            if verbose:
                print("forcing remake of executable")
            self.make_clean()
            self.make

        # call executable        
        command = [path_to_executable, "-i", self.load_str, "-s", self.save_str,"-c",self.temp_camera_file]
        if profile: 
            command = ["nvprof"] + command
        if verbose:
            command = command + ["-v"]
        if max_mem is not None:
            command = command + ["-m", str(max_mem)]
        print("calling render executable")
        subprocess.run(command, check = True)
        print("executable finished.")

        # destroy temp camera file if called
        if not self.temp_camera_file == self.camera_file_name:
            if os.path.exists(self.temp_camera_file):
                #os.remove(self.temp_camera_file)
                if verbose:
                    print("removed temporary camera file")

    def plot(self, save_location, cmap="Greys", vmin=-13, vmax=-10, remove_raw_images=False, verbose=False):
        
        # TODO: add labelling options, axes?
        
        save_location = save_location.removesuffix(".png") # strip as needed

        # check savespace exists
        save_dir = os.path.dirname(save_location)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # define persistent figure
        fig = plt.figure(figsize=(10.0/3,10.0/3))
        ax = fig.add_subplot()
        plt.subplots_adjust(hspace=0, wspace=0)
        X = np.linspace(0,1,self.num_pixels_X+1)
        Y = np.linspace(0,1,self.num_pixels_Y+1)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        ax.axis("off")

        num_images = len(self.cameras)
        for i in range(num_images):

            load_str = self.save_str + str(i).zfill(str_zfill) + ".npy"
            save_str = save_location + str(i).zfill(str_zfill) + ".png"

            img = np.load(load_str)

            pc = ax.pcolormesh(XX, YY, np.log10(img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
            fig.savefig(save_str, dpi=300, bbox_inches="tight")
            pc.remove()
            if (verbose):
                print("saved png at " + save_str)

            if (remove_raw_images):
                os.remove(load_str)
                if (verbose):
                    print("removed data file at " + load_str)

        plt.close("all")

class Mesh:

    def __init__(self, data_dir, nzfill=3):

        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        self.num_mb = 0
        self.mb_headers = []
        self.nzfill = nzfill
    
    def add_meshblock(self, mb_data, xl, xr):

        # TODO: add check against VRAM to ensure no meshblock exceeds limit, then auto refine

        mb_data = mb_data.astype(np.float32)
        npy_str = os.path.join(self.data_dir, "meshblock" + str(self.num_mb).zfill(self.nzfill) + ".npy")
        np.save(npy_str, mb_data)
        mb_shape = np.shape(mb_data)
        mb_size = np.size(mb_data)
        mb_header = [mb_size,*mb_shape,*xl,*xr]
        self.mb_headers.append(mb_header)
        self.num_mb += 1

    def write_header(self):

        header_str = os.path.join(self.data_dir, "header.txt")
        with open(header_str, "w") as f:
            for mb_header in self.mb_headers:
                f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(*mb_header))



