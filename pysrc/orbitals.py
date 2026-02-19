import numpy as np
import os, sys
from scipy.special import factorial

sys.path.append("..")
from pysrc import *

class Orbital:

    def __init__(self, n=1, l=0, m=0, data_dir = "/scratch/github/cuDART_wdir/orbitals"):

        # check inputs
        quantum_numers = [n, l, m]
        if any(not isinstance(x, int) for x in quantum_numers):
            raise Exception("quantum numbers must all be integers")

        if (l > n - 1):
            raise Exception("l = {0} incompatible with n = {1}, maximal l value {2}".format(l, n, n - 1))
        if (m > l or m < -l):
            raise Excpetion("m = {0} incompatible with l = {1}, m must be in [{2},{3}]".format(m, l, -l, -l))

        if os.path.isdir(data_dir):
            self.data_dir = data_dir
        else:
            raise Exception("unable to locate directory at {0}".format(data_dir))

        self.n = n
        self.l = l
        self.m = m
        self.a0 = 1

        self.update_funcs()
        self.npy_str = None

    def update_funcs(self):

        self.select_laguerre()
        self.select_legendre()
        self.build_radial_prefacs()
        self.build_angular_prefacs()

    def save_mesh(self, D = 256, half_len = 5):

        self.half_len = half_len
        self.D = D
        self.npy_str = os.path.join(self.data_dir, "n{0}l{1}m{2}D{3}L{4}.npy".format(self.n, self.l, self.m, D, half_len))
        span = np.linspace(-half_len, half_len, D)
        xx, yy, zz = np.meshgrid(span, span, span, indexing="ij")
        rr = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        cos_theta = zz / rr
        PP = self.eval_pdf(rr, cos_theta)
        np.save(self.npy_str, PP)

    def select_legendre(self):

        if self.l == 0:
            self.eval_legendre = lambda x : 1
        elif self.l == 1:
            if self.m == 0:
                self.eval_legendre = lambda x : x
            else:
                self.eval_legendre = lambda x : -np.sqrt(1 - x ** 2)
        elif self.l == 2:
            if self.m == 0:
                self.eval_legendre = lambda x : 0.5 * (3 * x ** 2 - 1)
            elif np.abs(self.m) == 1:
                self.eval_legendre = lambda x : - 3 * x * np.sqrt(1 - x ** 2)
            else:
                self.eval_legendre = lambda x : 3 * (1 - x ** 2)
        # l > 2 currently unsupported as n_max = 3

    def select_laguerre(self):

        n = self.n - self.l - 1
        a = 2 * self.l + 1

        if n == 0:
            self.eval_laguerre = lambda x : 1
        elif n == 1:
            self.eval_laguerre = lambda x : - x + a + 1
        elif n == 2:
            self.eval_laguerre = lambda x : 0.5 * (x**2 - 2 * (a + 2) * x + (a + 1) * (a+2))
        elif n == 3:
            self.eval_laguerre = lambda x :(-x ** 3 + 3 * (a + 3) * x ** 2 - 3 * (a + 2) * (a + 3) * x + (a + 1) * (a + 2) * (a + 3)) / 6
        else:
            raise Exception("n > 3 currently unsupported for generalised Laguerre polynomials")

    def build_radial_prefacs(self):

        self.r_fac = 2 / (self.n * self.a0)
        radial_prefac = factorial(self.n - self.l - 1) / (2 * self.n * factorial(self.n + self.l))
        radial_prefac *= np.power(self.r_fac, 5)
        self.radial_prefac = radial_prefac

    def build_angular_prefacs(self):

        self.angular_prefac = (2 * self.l + 1) / (2 * np.pi)
        self.angular_prefac *= factorial(self.l - self.m)
        self.angular_prefac /= factorial(self.l + self.m)

    def eval_R_sqr(self, r):

        return self.radial_prefac * np.exp(-self.r_fac * r) * np.power(r, 2 * self.l) * self.eval_laguerre(self.r_fac * r) ** 2

    def eval_Y_sqr(self, cos_theta):
        return self.angular_prefac * self.eval_legendre(cos_theta) ** 2

    def eval_pdf(self, r, cos_theta):
        return self.eval_R_sqr(r) * self.eval_Y_sqr(cos_theta)

    def plot_slice(self, save_str, axis = 2, axis_pos = 0, npy_str = None, half_len = None, D = None):

        if half_len is None:
            if self.half_len is None:
                raise Exception("no set half_len in orbital")
            else:
                half_len = self.half_len

        if D is None:
            if self.D is None:
                raise Exception("no set D in orbital")
            else:
                D = self.D

        if npy_str is None:
            if self.npy_str is None:
                npy_str = os.path.join(self.data_dir, "n{0}l{1}m{2}D{3}L{4}.npy".format(self.n, self.l, self.m, D, half_len))
            else:
                npy_str = self.npy_str

        try:
            data = np.load(npy_str)
        except:
            raise Exception("unable to load data at {0}".format(npy_str))

        set_plot_defaults()
        fig = plt.figure()
        ax = fig.add_subplot()
        span = np.linspace(-half_len, half_len, D)
        try:
            idx = np.where(span > axis_pos)[0][0]
        except:
            raise Exception("axis_pos out of bounds")
        if axis == 0:
            slicer = np.s_[idx,:,]
        elif axis == 1:
            slicer = np.s_[:,idx,:]
        else:
            slicer = np.s_[:,:,idx]

        sliced_data = data[slicer]
        XX, YY = np.meshgrid(span, span, indexing="ij")
        ax.pcolormesh(XX, YY, sliced_data)
        ax.set_aspect("equal")
        fig.savefig(save_str, dpi=600, bbox_inches="tight")
        plt.close("all")

    def render(self, save_dir, num_img = 100, N = 256, npy_load_str = None, half_len = None, D = None):

        if half_len is None:
            if self.half_len is None:
                raise Exception("no set half_len in orbital")
            else:
                half_len = self.half_len

        if D is None:
            if self.D is None:
                raise Exception("no set D in orbital")
            else:
                D = self.D

        if npy_load_str is None:
            if self.npy_str is None:
                npy_load_str = os.path.join(self.data_dir, "n{0}l{1}m{2}D{3}L{4}.npy".format(self.n, self.l, self.m, D, half_len))
            else:
                npy_load_str = self.npy_str

        npy_save_str = os.path.join(save_dir, "raw")
        png_save_str = os.path.join(save_dir, "img")

        # build template camera
        template_camera = Camera()
        template_camera.num_pixels_X = N
        template_camera.num_pixels_Y = N
        template_camera.tilt = 0.0
        template_camera.length_X = 1.0
        template_camera.length_Y = 1.0

        # build camera array, inherit from template
        phi = epsilon
        theta_ar = np.linspace(epsilon, np.pi - epsilon, num_img, endpoint=False)
        cameras = []
        for theta in theta_ar:
            camera = copy.deepcopy(template_camera)
            camera.set_sph_pos(r=2.0, theta=theta, phi=phi, target_origin=True)
            cameras.append(camera)
        print("initialised cameras")

        # generate scene
        scene = Scene(npy_load_str, npy_save_str, cameras)
        print("built scene")

        # render and save images
        scene.render(verbose=True)
        print("finished rendering raw images")

        scene.plot(png_save_str, cmap="afmhot", verbose=True, remove_raw_images=remove_raw_images, vmin=None, vmax=None)
        print("finished rendering rasterised images")

        print("unlablled render example finished.")