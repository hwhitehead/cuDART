import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess, copy
import glob
import pandas as pd

host_dir = "/mnt/users/hww27/cuDART"
str_zfill = 3

def set_plot_defaults(use_tex = True):
    """
    assign default plot settings before figure creation
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

class Profiler:

    def __init__(self, data_dir, prof_dir):
        self.data_dir = data_dir
        self.prof_dir = prof_dir

    def build(self, D_span = [64,128,256,512], save_boosted = True):

        self.D_span = D_span

        for D in D_span:
            shape = (D, D, D)
            data = np.ones(shape=shape, dtype=np.float32)
            save_str = os.path.join(self.data_dir, "unboosted_" + str(D) + ".npy")
            np.save(save_str, data.astype(np.float32))

            if save_boosted:
                shape = (D, D, D, 4)
                data = np.ones(shape=shape, dtype=np.float32)
                save_str = os.path.join(self.data_dir, "boosted_" + str(D) + ".npy")
                np.save(save_str, data.astype(np.float32))

    def run(self, N_span = [64,128,256,512], D_span = [64,128,256,512], num_iter = 10, rand_view = True):

        epsilon = 1e-4

        template_camera = Camera()
        template_camera.tilt = 0.0
        template_camera.length_X = 0.5
        template_camera.length_Y = 0.5
        
        if rand_view:
            phi = np.random.uniform(epsilon, 2 * np.pi - epsilon, num_iter)
            cos_theta = np.random.uniform(-1.0 + epsilon ,1.0 - epsilon, num_iter)
            theta = np.arccos(cos_theta)
            cameras = []
            for i in range(num_iter):
                camera = copy.deepcopy(template_camera)
                camera.set_sph_pos(r = 2.0, theta = theta[i], phi = phi[i], target_origin = True)
            cameras.append(camera)
        else:
            template_camera.set_sph_pos(r = 2.0, theta = 0.5 * np.pi - epsilon, phi = epsilon, target_origin = True)
            cameras = [template_camera] * num_iter

        
        for label, relativistic in zip(["unboosted_", "boosted_"], [False, True]):
            for j, D in enumerate(D_span):
                for i, N in enumerate(N_span):
                    # target data
                    npy_load_str = os.path.join(self.data_dir, label + str(D) + ".npy")
                    npy_save_str = os.path.join(self.data_dir, "scratch") # overwrite output, TODO: add off switch to write
                    
                    # update camera
                    for camera in cameras:
                        camera.num_pixels_X = int(N)
                        camera.num_pixels_Y = int(N)
                    
                    scene = Scene(npy_load_str, npy_save_str, cameras)
                    save_profile = os.path.join(self.prof_dir, "profile_N{0}D{1}b{2}.txt".format(N, D, relativistic))
                    scene.render(verbose = False, relativistic = relativistic, save_profile = save_profile)

                    print("finished N = " + str(N) + ", D = " + str(D) + " relativistic = " + str(relativistic))
                    print("\n\n\n\n\n")

    def time_from_prof(self, load_str):

        df = pd.read_csv(load_str, skiprows=3)
        df = df.fillna(0)

        # find time unit
        Avg = df["Avg"]
        time_type = Avg.iloc[0]

        # find average time
        task_names = df["Name"]
        row = np.where(task_names == "render_from_mesh(Camera, float*, Mesh**, bool)")[0][0]
        avg_time = float(df["Avg"].iloc[row])

        if (time_type == "ms"):
            return 1e-3 * avg_time
        elif (time_type == "us"):
            return 1e-6 * avg_time
        elif (time_type == "s"):
            return avg_time
        else:
            raise Exception("unable to parse time type in " + load_str)

    def save_timings(self, save_str, N_span = [64,128,256,512], D_span = [64,128,256,512]):

        dd, ii = np.meshgrid(D_span, N_span, indexing="ij")
        log_data_dims = np.log10(D_span)
        log_image_dims = np.log10(N_span)

        avg_times = np.zeros(shape=(np.size(N_span), np.size(D_span), 2)) # N, D, ub/b

        for i, N in enumerate(N_span):
            for j, D in enumerate(D_span):
                for label, relativistic in zip(["unboosted_", "boosted_"], [False, True]):
                    load_str = os.path.join(self.prof_dir, "profile_N" + str(N) + "D" + str(D) + "b" + str(relativistic) + ".txt")
                    render_time = self.time_from_prof(load_str)
                    if relativistic:
                        avg_times[i,j,1] = render_time
                    else:
                        avg_times[i,j,0] = render_time

        np.save(save_str, avg_times)

    def plot_timings(self, load_str, save_str, N_span = [64,128,256,512], D_span = [64,128,256,512]):

        avg_times = np.load(load_str) * 1e-3 # to ms

        print(avg_times)

        log_N = np.log10(N_span)
        log_D = np.log10(D_span)

        set_plot_defaults()
        height_ratios = np.array([1.0, 1.0])
        width_ratios = np.array([2.0])
        h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
        L = 20.0 / 3
        fig = plt.figure(figsize=(L, L * h_over_w))
        gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])

        cmap = plt.get_cmap("plasma")
        N_colors = [cmap(x) for x in np.linspace(0, 0.999, np.size(N_span))]
        D_colors = [cmap(x) for x in np.linspace(0, 0.999, np.size(D_span))]
        ax0.set_title(r"Render Image $N^2$ from domain $D^3$")

        ax0.set_xlabel(r"$\log_{10}(N)$")
        ax0.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
        
        labels = ["D = {0}".format(D) for D in D_span]
        for i, label in enumerate(labels):
            ax0.plot([],[],label=label, color=D_colors[i])
        ax0.legend(loc="upper left", frameon=False)
        for j, D in enumerate(D_span):
            ax0.plot(log_N, np.log10(avg_times[:, j, 0]), linestyle="solid", color=D_colors[j])
            ax0.plot(log_N, np.log10(avg_times[:, j, 1]), linestyle="dashed", color=D_colors[j])

        ax1.set_xlabel(r"$\log_{10}(D)$")
        ax1.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
        labels = ["N = {0}".format(N) for N in N_span]
        for i, label in enumerate(labels):
            ax1.plot([],[],label=label, color=N_colors[i])
        ax1.legend(loc="upper left", frameon=False)
        for i, N in enumerate(N_span):
            ax1.plot(log_D, np.log10(avg_times[i, :, 0]), linestyle="solid", color=N_colors[i])
            ax1.plot(log_D, np.log10(avg_times[i, :, 1]), linestyle="dashed", color=N_colors[i])

        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        plt.close("all")

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

    def __str__(self):
        retstr = "origin = ({0},{1},{2})\n".format(*self.origin)
        retstr += "normal = ({0},{1},{2})\n".format(*self.normal)
        retstr += "bias = ({0},{1},{2})\n".format(*self.bias)
        retstr += "(lx,ly) = ({0},{1})\n".format(self.length_X, self.length_Y)
        return retstr

class Scene:
    """ 
    this class provides a simple way for the user to call cuDART and process the results
    """
    def __init__(self, npy_load_str, npy_save_str, cameras = None, camera_file_name = None): 

        # parse load/save strings        
        self.npy_load_str = npy_load_str
        self.npy_save_str = npy_save_str.removesuffix(".png")

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

    def render(self, save_profile = None, verbose = False, check_make = True, force_make = False, plot = False, max_mem = None, relativistic = False):

        # prepare camera space
        self.build_camera_file()
        if (verbose):
            print("generated camera file at " + self.temp_camera_file)

        # check savespace exists
        save_dir = os.path.dirname(self.npy_save_str)
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
        command = [path_to_executable, "-i", self.npy_load_str, "-s", self.npy_save_str,"-c",self.temp_camera_file]
        if save_profile is not None: 
            command = ["nvprof", "--csv", "--log-file", save_profile]+ command
        if verbose:
            command = command + ["-v"]
        if max_mem is not None:
            command = command + ["-m", str(max_mem)]
        if relativistic:
            command = command + ["-r"]
        print("calling render executable")
        subprocess.run(command, check = True)
        print("executable finished.")

        # destroy temp camera file if called
        if not self.temp_camera_file == self.camera_file_name:
            if os.path.exists(self.temp_camera_file):
                #os.remove(self.temp_camera_file) # TEMP
                if verbose:
                    print("removed temporary camera file")

    def plot(self, save_location, cmap="Greys", vmin=-13, vmax=-10, remove_raw_images=False, verbose=False, log_c=True):
        
        # TODO: add labelling options, axes?
        
        save_location = save_location.removesuffix(".png") # strip as needed

        # check savespace exists
        save_dir = os.path.dirname(save_location)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # define persistent figure
        fig = plt.figure(figsize=(10.0/3,10.0/3))
        ax = fig.add_subplot()
        ax.set_facecolor("k")
        plt.subplots_adjust(hspace=0, wspace=0)
        X = np.linspace(0,1,self.num_pixels_X+1)
        Y = np.linspace(0,1,self.num_pixels_Y+1)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        num_images = len(self.cameras)
        for i in range(num_images):

            load_str = self.npy_save_str + str(i).zfill(str_zfill) + ".npy"
            save_str = save_location + str(i).zfill(str_zfill) + ".png"

            img = np.load(load_str)
            if log_c: img = np.log10(img)
            pc = ax.pcolormesh(XX, YY, img, vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
            fig.savefig(save_str, dpi=300, bbox_inches="tight")
            pc.remove()
            if (verbose):
                print("saved png at " + save_str)

            if (remove_raw_images):
                os.remove(load_str)
                if (verbose):
                    print("removed data file at " + load_str)

        plt.close("all")

    def calc_lightcurve(self, save_location):

        save_location = save_location.removesuffix(".png") # strip as needed

        num_raws = len(self.cameras)
        lum_ar = np.zeros(shape=(num_raws))
        for i in range(num_raws):

            raw_str = self.npy_save_str + str(i).zfill(str_zfill) + ".npy"
            raw_img = np.load(raw_str)
            lum = np.sum(raw_img)
            lum_ar[i] = lum

        return lum_ar

    def plot_wlightcurve(self, save_location, cmap="Greys", vmin=-13, vmax=-10, remove_raw_images=False, verbose=False):
        
        save_location = save_location.removesuffix(".png") # strip as needed
        lum_ar = self.calc_lightcurve(save_location)

        # check savespace exists
        save_dir = os.path.dirname(save_location)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # define persistent figure
        height_ratios = np.array([0.3,1])
        width_ratios = np.array([1,0.05])
        set_plot_defaults(False)
        h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
        fig = plt.figure(figsize=(10.0/3,10.0/3 * h_over_w))
        gs = fig.add_gridspec(2,2,height_ratios=height_ratios,width_ratios=width_ratios)
        ax = fig.add_subplot(gs[1,0])
        tax = fig.add_subplot(gs[0,0])
        cax = fig.add_subplot(gs[1,1])
        ax.set_facecolor("k")
        plt.subplots_adjust(hspace=0, wspace=0)
        X = np.linspace(0,1,self.num_pixels_X+1)
        Y = np.linspace(0,1,self.num_pixels_Y+1)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # plot header
        num_images = len(self.cameras)
        x_span = np.linspace(0,1,num_images)
        tax.set_xlim([0,1])
        delta = (lum_ar - np.mean(lum_ar)) / np.mean(lum_ar)
        tax.plot(x_span, delta, color='k')
        #tax.plot(x_span, np.log10(lum_ar), color='k')
        #tax.set_ylim([22, 24.2])
        #tax.set_xlabel(r"log$_{10}$ Luminosity [arb.]")
        tax.set_xlabel(r"$\left(L_\nu - \langle L_\nu \rangle\right)/ \langle L_\nu \rangle$")
        tax.axhline(y=0, color='k', linestyle="dashed", zorder=-10)
        #tax.set_ylim([-0.4,0.4])
        #tax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        #tax.set_yticklabels(["-40%", "-20%", "mean", "+20%", "+40%"])
        tax.xaxis.set_label_position("top")
        tax.xaxis.set_ticks([])

        # plot colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, cax=cax, orientation="vertical")
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r"$\log_{10}\left(I_{\nu}\right)$")

        # plot inserts
        for i in range(num_images):

            load_str = self.npy_save_str + str(i).zfill(str_zfill) + ".npy"
            save_str = save_location + str(i).zfill(str_zfill) + ".png"

            img = np.load(load_str)
            tstamp = tax.axvline(x=x_span[i], color='k', alpha=0.2)

            pc = ax.pcolormesh(XX, YY, np.log10(img), vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
            fig.savefig(save_str, dpi=300, bbox_inches="tight")
            pc.remove()
            tstamp.remove()
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
        mb_shape = np.shape(mb_data)[:3]
        mb_size = np.size(mb_data)
        mb_header = [mb_size,*mb_shape,*xl,*xr]
        self.mb_headers.append(mb_header)
        self.num_mb += 1

    def write_header(self):

        header_str = os.path.join(self.data_dir, "header.txt")
        with open(header_str, "w") as f:
            for mb_header in self.mb_headers:
                f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(*mb_header))

class BSpline:

    # referencing: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/CURVE-INT-global.html

    def __init__(self, p, D_array, mode="chord", len_power=0.5):
        # load data
        self.D_array = D_array
        self.n = np.shape(D_array)[0] - 1
        if (p > self.n):
            raise Exception("degree must be less than or equal to number of data points")
        self.p = p # order
        self.m = self.n + self.p + 1
        num_middle = (self.m + 1) - 2 * (self.p + 1)
        self.u_list = [0] * (self.p + 1) + np.linspace(0, 1, num_middle + 2)[1:-1].tolist() + [1] * (self.p + 1)

        # package data
        self.set_spacing(mode, len_power)
        self.build_N_array()

        # solve data
        self.solve_P()

    def set_spacing(self, mode, len_power):
        if mode == "uniform":
            self.t_list = np.linspace(0, 1, self.n+1)
        elif mode in ["chord", "centripetal"]:
            if mode == "chord":
                len_power = 1.0
            sides = self.D_array[1:,:] - self.D_array[:-1,:] # D_{k+1} - D_k
            lengths = np.sum(np.power(np.abs(sides),len_power), axis=1)
            total_length = np.sum(lengths)
            t_list = np.zeros(shape=self.n+1)
            t_list[-1] = 1
            for i in range(1,self.n):
                t_list[i] = t_list[i-1] + lengths[i] / total_length
            self.t_list = t_list
        else:
            raise Exception("unable to recognised mode, select form [\"uniform\",\"chord\",\"centripetal\"]")

        self.tl = self.t_list[0]
        self.tr = self.t_list[-1]

    def build_N_row(self, u):

        # init row as zero
        N_row = np.zeros(shape=(self.n+1))

        # handle edge cases
        if u == self.u_list[0]:
            N_row[0] = 1.0
            return N_row
        elif u == self.u_list[-1]:
            N_row[self.n] = 1.0
            return N_row

        k = np.argmax(self.u_list > u) - 1

        # loop over degrees
        N_row[k] = 1.0
        for d in range(1,self.p+1):
            N_row[k-d] = N_row[k-d+1] * (self.u_list[k+1] - u) / (self.u_list[k+1] - self.u_list[k-d+1])
            for i in range(k-d+1,k):
                N_row[i] = N_row[i] * (u - self.u_list[i]) / (self.u_list[i+d] - self.u_list[i])
                N_row[i] += N_row[i+1] * (self.u_list[i+d+1] - u) / (self.u_list[i+d+1] - self.u_list[i+1])
            N_row[k] = N_row[k] * (u - self.u_list[k]) / (self.u_list[k+d] - self.u_list[k])

        return N_row

    def build_N_array(self):
        N_array = np.zeros(shape=(self.n+1, self.n+1))
        for row, t in enumerate(self.t_list):
            N_row = self.build_N_row(t)
            N_array[row,:] = N_row
        self.N_array = N_array

    def solve_P(self):
        # solve lin equation set for each column
        P_array = np.zeros(shape=(self.n+1, 3))
        for i in range(3):
            D_column = self.D_array[:,i]
            P_column = np.linalg.solve(self.N_array, D_column)
            P_array[:,i] = P_column

        self.P_array = P_array

    def eval_spline(self, t_span):

        C = np.zeros(shape=(np.size(t_span),3))
        for i, t in enumerate(t_span):
            N_coeffs = self.build_N_row(t)
            C_i = np.zeros(shape=(3))
            for j in range(self.n+1):
                C_i += N_coeffs[j] * self.P_array[j,:]
            C[i,:] = C_i

        return C

class GuidedCamera:

    def __init__(self, checkpoints = None, targets = None):
        self.checkpoints = checkpoints
        self.targets = targets

    def generate_cameras(self, p = 3, mode = "chord", num_img = 100, template_camera = None, camera_times = None):
        if template_camera is None:
            template_camera = Camera()
            template_camera.num_pixels_X = 512
            template_camera.num_pixels_Y = 512
            template_camera.tilt = 0
            template_camera.length_X = 0.66
            template_camera.length_Y = 0.66

        if self.checkpoints is None:
            raise Exception("require checkpoints set before camera generation")

        # if self.targets is None:
        #     self.targets = [0,0,0]
        # elif np.shape(self.targets)[0] > 1:
        #     raise Exception("multi target currently unsupported")

        if camera_times is None: # even spacing
            self.camera_times = np.linspace(0, 1, num_img)
        else: # normalise
            max_time = np.max(camera_times)
            self.camera_times = np.array(camera_times) / max_time

        self.origin_spline = BSpline(p, self.checkpoints, mode)
        origins = self.origin_spline.eval_spline(self.camera_times)
        cameras = []
        for i in range(num_img):
            camera = copy.deepcopy(template_camera)
            camera.origin = origins[i,:]
            camera.set_target(self.targets[0]) # single target for now
            cameras.append(camera)

        return cameras