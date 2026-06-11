import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess, copy, pathlib
import glob
import pandas as pd

# define global constants
str_zfill = 5       # num zeros for zpadding strings
epsilon = 1e-6      # small value for off-axis casts

# define unit conversions
kpc_to_m = 1e3 * 3.086e+16
Myr_to_s = 1e6 * 365 * 24 * 60 * 60
c_light = 3e8

# supress div zero warnings from log10 usage
np.seterr(divide="ignore")

def set_plot_defaults(use_tex = True):
    """
        Assigns default matplotlib settings before figure creation
        
        Parameters
        ----------
        use_tex : bool
            Boolean to use full tex rendering or MathTex
    """
    ## FIGURE

    # test for tex environment
    if use_tex:
        plt.rcParams["text.usetex"] = "True"
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        # FONT
        plt.rcParams['font.serif']=['cm']
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']=['cm']
    else:
        plt.rcParams["text.usetex"] = "False"

    plt.rcParams['font.size']=8 # defval 18
    plt.rcParams['xtick.labelsize']=8
    plt.rcParams['ytick.labelsize']=8
    plt.rcParams['legend.fontsize']=8
    plt.rcParams['axes.titlesize']=8
    plt.rcParams['axes.labelsize']=8
    plt.rcParams['axes.linewidth']=1.5
    plt.rcParams["lines.linewidth"] = 2.2
    ## TICKS
    plt.rcParams['xtick.top']='False'
    plt.rcParams['xtick.bottom']='True'
    plt.rcParams['xtick.minor.visible']='True'
    plt.rcParams['xtick.direction']='out'
    plt.rcParams['ytick.left']='True'
    plt.rcParams['ytick.right']='True'
    plt.rcParams['ytick.minor.visible']='True'
    plt.rcParams['ytick.direction']='out'
    plt.rcParams['xtick.major.width']=1.5
    plt.rcParams['xtick.minor.width']=1
    plt.rcParams['xtick.major.size']=3
    plt.rcParams['xtick.minor.size']=9/4
    plt.rcParams['ytick.major.width']=1.5
    plt.rcParams['ytick.minor.width']=1
    plt.rcParams['ytick.major.size']=3
    plt.rcParams['ytick.minor.size']=9/4

class Camera:
    """
        Class allowing for the definitions of general images planes, and the construction of text files containing
        the camera properties to be read at cpp runtime

        Parameters
        ----------
        origin : 3-vector
            Central Camera position 
        normal : 3-vector
            Camera look direction
        num_pixels_X : int
            Number of pixels in X direction
        num_pixels_Y :  int
            Number of pixels in Y direction
        length_X : float
            Spatial extent of screen in X direction
        length_Y : float
            Spatial extent of screen in Y direction
        bias : 3-vector
            Orientation vector, set Y direction for image plane
        tilt : float
            Angle to rotate X, Y axes about the normal
        num_pixels : int
            Total number of pixels 
        t_obs : float
            Time for observation (in Myr)
        theta : float
            Polar coordinate for camera origin
        phi : float
            Azimuthal coordinate for camera origin
        r : float
            Radial coordinate for camera origin
         
        Methods
        -------
        header_str()
            Packages Camera properties for write to camera text file
        set_sph_pos()
            Set Camera origin in spherical polar coordinates
        set_target()
            Set the position for the Camera to target (sets normal)
        __str__()
            Prints the Camera properties to the command line
    """

    def __init__(self, origin = np.array([1.0,0.0,0.0]), normal = np.array([-1.0,0.0,0.0]), bias = np.array([0.0,0.0,1.0]),
                    num_pixels_X = 512, num_pixels_Y = 512, length_X = 1.0, length_Y = 1.0, tilt = 0.0, t_obs = 0.0,
                    theta = 0.5 * np.pi - epsilon, phi = epsilon, r = 2.0):
        # stash inputs

        # test vector types
        for vec_type_input, var_name in zip([origin, normal, bias],["origin", "normal", "bias"]):
            if not isinstance(vec_type_input, np.ndarray):
                raise Exception("input {0} must be array type".format(var_name))

        self.origin = origin
        self.normal = normal
        self.num_pixels_X = num_pixels_X
        self.num_pixels_Y = num_pixels_Y
        self.length_X = length_X
        self.length_Y = length_Y
        self.bias = bias
        self.tilt = tilt
        self.num_pixels = num_pixels_X * num_pixels_Y
        self.t_obs = t_obs
        self.theta = theta
        self.phi = phi
        self.r = r

    def __str__(self):
        """Print summary of camera properties"""
        print_str = "printing camera data...\n"
        print_str += "origin = ({0},{1},{2})\n".format(*self.origin)
        print_str += "normal = ({0},{1},{2})\n".format(*self.normal)
        print_str += "bias = ({0},{1},{2})\n".format(*self.bias)
        print_str += "(lx,ly) = ({0},{1})\n".format(self.length_X, self.length_Y)
        return print_str

    def header_str(self):
        """Package camera properties into string for write to camera text file"""
        return "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}\n".format(*self.origin, *self.normal, *self.bias, self.tilt, self.length_X, self.length_Y, self.t_obs)
    
    def set_sph_pos(self, r = None, theta = None, phi = None, target_origin = False):
        """
            set the camera position in spherical polar coordinates
            if coordinates not specified, use internal attributes

            Parameters
            ----------
            r : float
                Radial coordinate
            theta : float
                Polar coordinate
            phi : float
                Azimuthal coordinate
            target_origin : bool
                Boolean to auto-call set_target()
        """

        # if not passed as arg, use internal values
        if theta is None: theta = self.theta
        if phi is None: phi = self.phi
        if r is None: r = self.r

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        self.origin = r * np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
        if (target_origin): self.set_target(np.array([0.0, 0.0, 0.0])) # target coordinate origin

    def set_target(self, target):
        """
            Set look direction for camera and auto-normalise
        
            Parameters
            ----------
            target :  3-vector
                Position for camera to target
        """
        self.normal = target - self.origin
        self.normal = self.normal / np.linalg.norm(self.normal)
    
class Scene:
    """
        The Scene class is used to call the main cpp executable as a subprocess (with optional flags).
        The class also supports the generation of .png files from the raw .npy output data, and the ability
        to make/remake the cpp executable from within Python.

        Parameters
        ----------
        load_str : str
            Path to the simulation data to render, should by single .npy file or directory (unlabelled vs labelled mode)
        save_dir : str
            Path to the output write space, should be directory (will auto mkdir if possible)
        camera_file_name : str
            Path to camera file name, if file is to persist (else auto deleted)
        cameras : str
            List of **Camera** objects, to specify image properties
        
        Methods
        -------
        render()
            Calls the cpp executable as a subprocess, running the main render routine
        plot()
            Converts the raw .npy images into .png figures
        build_camera_file()
            Generates a text file contaning all camera properties for read at cpp runtime
        make_clean()
            Calls the 'make clean' routine from the main Makefile
        make()
            Calls the 'make' routine from the main Makefile
        print_command()
            Prints full command line invokation of cpp executable to terminal
        __str__()
            Prints the Scene properties to the command line

    """
    def __init__(self, load_str, save_dir, cameras = None, camera_file_name = None): 

        # parse load/save strings        
        self.load_str = load_str # must be .npy file, or directory
        self.save_dir = save_dir # may not exist, should be directory
        self.camera_file_name = camera_file_name 

        # check existence and types
        if not os.path.exists(self.load_str):
            raise Exception("unable to locate {0} for loading".format(self.load_str))#
        elif not (os.path.isdir(self.load_str) or self.load_str.endswith(".npy")):
            raise Exception("load_str ({0}) must be .npy file or directory (for unlabelled/labelled runs respecitvely).")
        if not os.path.exists(self.save_dir):
            print("unable to locate {0} for saving".format(self.save_dir))
            os.mkdir(self.save_dir)

        # handle None, list/array or single pass for cameras
        if cameras is None:
            self.cameras = [Camera()] # initialise single default camera
        elif isinstance(cameras, list) or isinstance(cameras, np.ndarray):
            self.cameras = cameras # pass list type input directly
        else:
            self.cameras = [cameras] # package singel type input
        
        # handle None pass to camera_file_name
        if self.camera_file_name is None:
            self.temp_camera_file = "temp_camera_file.txt"
        else:
            self.temp_camera_file = self.camera_file_name

        # check camera dimensions are consistent across all cameras
        self.num_pixels_X = cameras[0].num_pixels_X
        self.num_pixels_Y = cameras[0].num_pixels_Y
        for camera in self.cameras:
            if (camera.num_pixels_X != self.num_pixels_X) or (camera.num_pixels_Y != self.num_pixels_Y):
                raise Exception("all Camera objects must have consistent image dimensions.")

    def __str__(self):
        """Prints the Scene properties to the command line"""
        print_str = "printing scene data...\n"
        print_str += "load_str = {0}\n".format(self.load_str)
        print_str += "save_dir = {0}\n".format(self.save_dir)
        print_str += "num_cameras = {0}\n".format(len(self.cameras))
        print_str += "camera_file_name = {0}\n".format(self.temp_camera_file)

    def build_camera_file(self):
        """Generates a text file contaning all camera properties for read at cpp runtime"""
        with open(self.temp_camera_file, "w") as f:
            # add header with const image dimensions (zero packed for istringstream read)
            f.write("{0} {1} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(self.cameras[0].num_pixels_X, self.cameras[0].num_pixels_Y))
            # add line for each camera to run render with
            for camera in self.cameras:
                f.write(camera.header_str())

    def make_clean(self):
        """Calls the 'make clean' routine from the main Makefile"""
        subprocess.run(["make","clean"], check = True)       

    def make(self):
        """Calls the 'make' routine from the main Makefile"""
        subprocess.run(["make"], check = True)

    def print_command(self, command):
        """Prints full command line invokation of cpp executable to terminal"""
        i = 0
        while i < np.size(command): 
            # handle end case
            if i == np.size(command) - 1:
                print(command[i])
                i += 1
            elif command[i].startswith("-") and not command[i+1].startswith("-"):
                print("\n{0} {1} ".format(command[i],command[i+1]), end='')
                i += 2
            else:
                print("{0} ".format(command[i]), end='')
                i += 1
        print("\n")

    def render(self, save_profile = False, verbose = False, check_make = True, force_make = False, 
                max_mem = None, relativistic = False, doppler_index = None, power_law_index = None, append = False,
                lookback = False, flexload = False, verbose_cpp = False):

        """
            Given a constructed Scene, format a subprocess invokation of the main cpp executable with any 
            specified flags. Remove any temporary files post render.

            Parameters
            ----------
            relativistic : bool
                Runs render using relativistic boosting (default False)
            lookback : bool
                Runs render using finite speed of light implementation (default False)
            flexload : bool
                Attempts to skip out-of-bounds snapshots in when running with lookback (default False)
            power_law_index : None
                Sets power law slope for rest frame emission (default None, autos to -0.6 in cpp)
            doppler_index : None
                Sets exponent for Doppler factor (overwritten by power_law_index, default None)
            verbose : bool
                Prints progress of pythonic execution to terminal (default False)
            verbose_cpp :  bool
                Prints progress of cpp execution to terminal (default False)
            save_profile : string
                Path to profiling log of cpp executable, if None, runs without profiling (default None)
            append : bool
                Sums render output to existing .npy files (default False)
            check_make : bool 
                Checks to see if executable requires remake (default False)
            force_make : bool
                Makes explicit call to the ``make clean``, ``make`` methods (default False)
            max_mem : int
                Sets maximum allowed VRAM occupancy for execution (default None)
        """

        # prepare camera space
        self.build_camera_file()
        if (verbose):
            print("generated camera file at {0}.".format(self.temp_camera_file))

        # check executable exists, or build
        pysrc_dir = pathlib.Path(__file__).parent.resolve()
        path_to_executable = os.path.join(pysrc_dir, "..", "bin","cudart")
        if not os.path.isfile(path_to_executable):
            if (verbose):
                print("unable to located executable, forcing remake...")
            self.make()
        elif check_make:
            if verbose:
                print("checking for updates since last make...")
            self.make()
        elif force_make:
            if verbose:
                print("forcing remake of executable...")
            self.make_clean()
            self.make

        # prepare command line argument to invoke .cpp executable, with proper flags  
        command = [path_to_executable, "-i", self.load_str, "-s", self.save_dir,"-c",self.temp_camera_file]
        # run executable with nvprof
        if save_profile: 
            profile_location = os.path.join(self.save_dir, "profiling")
            command = ["nsys", "profile", "--stats=true","--export=text","--output={0}".format(profile_location)] + command
        # pass verbose flag 
        if verbose_cpp: 
            command = command + ["-v"]
        # pass VRAM limit 
        if max_mem is not None:
            command = command + ["-m", str(max_mem)]
        # run using relativistic boosting
        if relativistic:
            command = command + ["-r"]
        # run using finite speed of light (required directory-type input)
        if lookback:
            if not os.path.isdir(self.load_str):
                raise Exception("if using lookback mode, input must be directory.")
            command = command + ["-l"]
            # run using the flexload snapshot/camera skipping routine
            if flexload:
                command = command + ["-f"]
        # run using specific power-law for rest-frame emission
        if power_law_index is not None:
            command = command + ["-p", str(power_law_index)]
        elif doppler_index is not None: # power_law has priority over doppler
            command = command + ["-d", str(doppler_index)]
        # when saving raw images, add to existing files in save space
        if append:
            command = command + ["-a"]

        # invoke executable
        if (verbose):
            print("calling render executable...")
            self.print_command(command)
        subprocess.run(command, check = True)
        if (verbose): print("executable finished.")

        # destroy temp camera file if not specified at Scene init
        if self.camera_file_name is None:
            if os.path.exists(self.temp_camera_file):
                os.remove(self.temp_camera_file)
                if verbose: print("removed temporary camera file.")

        # parse nsys output, cleanup
        if save_profile:
            nsys_comamnd = ["nsys", "profile", "--report=osrt_sum", "--format=column", profile_location + ".sqlite"]
            subprocess.run(command, check = True)

    def plot(self, fig_save_dir = None, cmap = "afmhot", vmin = -6, vmax = 0, remove_raw_npy = False, verbose = False, log_data = True):
        
        """
            Generates .png figures from raw .npy images
        
            Parameters
            ----------
            fig_save_dir: str
                Path to save location for .png files (autos to .npy dir, will auto mkdir if possible)
            cmap : str
                Matplotlib colormap (default "afmhot")
            vmin : float
                Minimum value for colormap scaling (default -6)
            vmax : float
                Maxmimum value for colormap scaling (default 0)
            log_data : bool
                Boolean to plot image data in logspace (default True)
            verbose : bool
                Prints progress of the plotting process (default False)
            remove_raw_npy : bool
                Removes .npy files after .png generation (default False)
        """
        
        # build write space if needed
        if fig_save_dir is None: # if None, use same directory as raw .npy files
            fig_save_dir = self.save_dir
        elif not os.path.isdir(fig_save_dir):
            os.mkdir(fig_save_dir)

        # define persistent figure to reuse for each image
        fig = plt.figure(figsize=(10.0/3,10.0/3))
        ax = fig.add_subplot()
        ax.set_facecolor("k")
        plt.subplots_adjust(hspace=0, wspace=0)
        X = np.linspace(0,1,self.num_pixels_X+1)
        Y = np.linspace(0,1,self.num_pixels_Y+1)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        # iterate over raw images
        num_images = len(self.cameras)
        for i in range(num_images):
            # build load, save paths
            load_str = os.path.join(self.save_dir, "raw{0}.npy".format(str(i).zfill(str_zfill)))
            save_str = os.path.join(fig_save_dir, "img{0}.png".format(str(i).zfill(str_zfill)))

            # load image data
            img = np.load(load_str)
            if log_data: img = np.log10(img) # if flagged, apply log10 to img data
            pc = ax.pcolormesh(XX, YY, img, vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
            
            # save figure, and cleanup
            fig.savefig(save_str, dpi=300, bbox_inches="tight")
            pc.remove()
            if (verbose): print("saved png at {0}".format(save_str))

            if (remove_raw_npy):
                os.remove(load_str)
                if (verbose): print("removed data file at {0}".format(load_str))

        plt.close("all")

class Mesh:

    """
        The Mesh class is used when running cuDART in labelled mode, acting as a wrapper for multiple sub-domains termed MeshBlocks
    
        Parameters
        ----------
        data_dir : str
            Path to directory, where labelled data will be build (will auto mkdir if possilbe)
        
        Attributes
        ----------
        num_mb : int
            Number of subdomains (MeshBlocks) within Mesh
        mb_headers : list
            List of strings containing MeshBlock metadata

        Methods
        -------
        add_meshblock()
            Package a sub-domain and add to the labelled dataset
        write_header()
            Generate a header text file with subdomain metadata
    """

    def __init__(self, data_dir):
        # check inputs and stash
        self.data_dir = data_dir                    # main directory for data
        if not os.path.isdir(self.data_dir):        # check existance, build if needed
            os.mkdir(self.data_dir)
        self.num_mb = 0                             # initially Mesh is empty
        self.mb_headers = []                        # list for MeshBlock header stash
    
    def add_meshblock(self, mb_data, xl, xr):
        """
            Package MeshBlock data and append to Mesh

            Parameters
            ----------
            mb_data : numpy array
                rank 3 or rank 4 numpy array containing subdomain simulation data
            xl : 3-vector
                spatial coordinate of lower subdomain vertex (xmin, ymin, zmin)
            xr : 3-vector
                spatial coordinate of upper subdomain vertex (xmax, ymax, zmax)
        """
        mb_data = mb_data.astype(np.float32)        # enforce cast to float32 type
        npy_str = os.path.join(self.data_dir, "meshblock" + str(self.num_mb).zfill(str_zfill) + ".npy")
        np.save(npy_str, mb_data)                   # save data to main dir
        mb_shape = np.shape(mb_data)[:3]            # "shape" is spatial only, ignore 4th axis
        mb_size = np.size(mb_data)                  # "size" is for memory usage, include all axes
        mb_header = [mb_size,*mb_shape,*xl,*xr]     
        self.mb_headers.append(mb_header)           # stash header
        self.num_mb += 1                            # update MeshBlock count

    def write_header(self):
        """Generate a header text file with subdomain metadata"""
        print(self.mb_headers)
        header_str = os.path.join(self.data_dir, "header.txt")
        with open(header_str, "w") as f:
            for mb_header in self.mb_headers:
                f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(*mb_header))