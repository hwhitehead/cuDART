import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess

host_dir = "/mnt/users/hww27/cuDART"
str_zfill = 3

class Camera:
    """
    this class is a basic struct to contain camera properties
    """

    def __init__(self, R=2.0, theta=0.500001 * np.pi, phi=0.0001, 
                    num_pixels_X=512, num_pixels_Y=512, length_X=1.0, length_Y=1.0,
                    bias = np.array([0.0,0.0,1.0]), tilt=0.0):
        # copy data
        self.R = R
        self.theta = theta
        self.phi = phi
        self.num_pixels_X = num_pixels_X
        self.num_pixels_Y = num_pixels_Y
        self.length_X = length_X
        self.length_Y = length_Y
        self.bias = bias
        self.tilt = tilt

    def unpack(self):
        return self.R, self.theta, self.phi, self.tilt, self.length_X, self.length_Y # TODO: implement BIAS

class Scene:
    """ 
    this class provides a simple way for the user to call cuDART and process the results
    """
    def __init__(self, load_str, save_str, cameras=None, camera_file_name=None): 

        # parse load/save strings        
        if not load_str.endswith(".npy"): # enforce suffix
            load_str += ".npy"
        self.load_str = load_str
        self.save_str = save_str.removesuffix(".png")

        if cameras is None:
            self.cameras = [Camera()]
        elif isinstance(cameras, list) or isinstance(cameras, np.ndarray):
            self.cameras = cameras
        else:
            self.cameras = [cameras]
        self.camera_file_name = camera_file_name

        if self.camera_file_name is None:
            self.temp_camera_file = "temp_camera_file.txt"
        else:
            self.temp_camera_file = self.camera_file_name

        # check camera dimensions
        self.num_pixels_X = cameras[0].num_pixels_X
        self.num_pixels_Y = cameras[0].num_pixels_Y
        for camera in self.cameras:
            if camera.num_pixels_X != self.num_pixels_X or camera.num_pixels_Y != self.num_pixels_Y:
                raise Exception("all Camera objects must have coherant image dimensions.")

    def build_camera_file(self):

        with open(self.temp_camera_file, "w") as f:
            f.write("{0} {1} 0.0 0.0 0.0 0.0\n".format(self.cameras[0].num_pixels_X, self.cameras[0].num_pixels_Y)) # header
            for camera in self.cameras:
                f.write("{0} {1} {2} {3} {4} {5}\n".format(*camera.unpack()))

    def make(self):

        subprocess.run(["make","clean"])
        subprocess.run(["make"])

    def render(self, profile=False, verbose=False, force_make=False, plot=False):

        # prepare camera space
        self.build_camera_file()
        if (verbose):
            print("generated camera file.")

        # check executable exists, or build
        path_to_executable = os.path.join(host_dir, "bin/cudart")
        if not os.path.isfile(path_to_executable):
            if (verbose):
                print("unable to located executable, forcing remake.")
            self.make()

        # call executable        
        command = [path_to_executable, "-i", self.load_str, "-s", self.save_str,"-c",self.temp_camera_file]
        if profile: 
            command = ["nvprof"] + command
        if verbose:
            command = command + ["-v"]
        print("calling render executable")
        subprocess.run(command)
        print("executable finished.")

        # destroy temp camera file if called
        if not self.temp_camera_file == self.camera_file_name:
            if os.path.exists(self.temp_camera_file):
                os.remove(self.temp_camera_file)
                if verbose:
                    print("removed temporary camera file")

    def plot(self, save_location, cmap="Greys", vmin=-13, vmax=-10, remove_data=False):
        
        save_location = save_location.removesuffix(".png") # strip as needed

        # define persistent figure
        fig = plt.figure(figsize=(10.0/3,10.0/3))
        ax = fig.add_subplot()
        plt.subplots_adjust(hspace=0, wspace=0)
        X = np.linspace(0,1,self.num_pixels_X+1)
        Y = np.linspace(0,1,self.num_pixels_Y+1)
        XX, YY = np.meshgrid(X, Y, indexing="ij")

        num_images = len(self.cameras)
        for i in range(num_images):

            load_str = self.save_str + str(i).zfill(str_zfill) + ".npy"
            save_str = save_location + str(i).zfill(str_zfill) + ".png"

            img = np.load(load_str)

            pc = ax.pcolormesh(XX, YY, np.log10(img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
            fig.savefig(save_str, dpi=300, bbox_inches="tight")
            for handle in pc: 
                handle.remove()
            if (verbose):
                print("saved png at " + save_str)

            if (remove_data):
                os.remove(load_str)
                if (verbose):
                    print("removed data file at " + load_str)

        plt.close("all")



