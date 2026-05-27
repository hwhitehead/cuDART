# this file is designed to contains functions soon to be deprecated

class Profiler:

    def __init__(self, data_dir, prof_dir):
        self.data_dir = data_dir
        self.prof_dir = prof_dir

    def build(self, D_span = [64,128,256,512], save_boosted = True):

        self.D_span = D_span

        for D in D_span:
            shape = (D, D, D)
            data = np.full(shape=shape, fill_value=0.5, dtype=np.float32)
            save_str = os.path.join(self.data_dir, "unboosted_" + str(D) + ".npy")
            np.save(save_str, data.astype(np.float32))

            if save_boosted:
                shape = (D, D, D, 4)
                data = np.full(shape=shape, fill_value=0.5, dtype=np.float32)
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

        # read average duration for render_from_mesh kernel in seconds

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

    def plot_timings(self, load_str, save_str, N_span = [64,128,256,512], D_span = [64,128,256,512], num_iter = None):


        avg_times = np.load(load_str) * 1e3 # to ms

        log_N = np.log10(N_span)
        log_D = np.log10(D_span)
        N_labels = [str(N) for N in N_span]
        D_labels = [str(D) for D in D_span]

        set_plot_defaults()
        height_ratios = np.array([1.0, 1.0, 1.0])
        width_ratios = np.array([2.0, 0.05])
        h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
        L = 20.0 / 3
        fig = plt.figure(figsize=(L, L * h_over_w))
        gs = fig.add_gridspec(np.size(height_ratios), np.size(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        ax2 = fig.add_subplot(gs[2,0])
        cax0 = fig.add_subplot(gs[0,1])
        cax1 = fig.add_subplot(gs[1,1])
        cax2 = fig.add_subplot(gs[2,1])

        plasma = plt.get_cmap("plasma")
        viridis = plt.get_cmap("viridis")
        N_colors = [viridis(x) for x in np.linspace(0, 0.999, np.size(N_span))]
        D_colors = [plasma(x) for x in np.linspace(0, 0.999, np.size(D_span))]
        
        title_str = r"Render Image $N^2$ from domain $D^3$"
        if num_iter is not None and num_iter > 1:
            title_str += " (averaged over {0} calls)".format(num_iter)

        axr = ax0.twinx()
        axr.yaxis.set_visible(False)
        axr.plot([],[],color=plasma(0), linestyle="solid", label="Unboosted")
        axr.plot([],[],color=plasma(0), linestyle="dashed", label="Boosted")
        axr.legend(loc="upper left", frameon=False)

        ax0.set_title(title_str)
        ax0.set_xlabel(r"$\log_{10}(N)$")
        ax0.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
        for j, D in enumerate(D_span):
            ax0.plot(log_N, np.log10(avg_times[:, j, 0]), linestyle="solid", color=D_colors[j])
            ax0.plot(log_N, np.log10(avg_times[:, j, 1]), linestyle="dashed", color=D_colors[j])
        ax0.set_xticks(log_N, labels=N_labels)
        ax0.set_xlim([log_N[0], log_N[-1]])
        ax0.set_ylim([-1, 2.5])

        ax1.set_xlabel(r"$\log_{10}(D)$")
        ax1.set_ylabel(r"$\log_{10}(\tau [\mathrm{ms}])$")
        for i, N in enumerate(N_span):
            ax1.plot(log_D, np.log10(avg_times[i, :, 0]), linestyle="solid", color=N_colors[i])
            ax1.plot(log_D, np.log10(avg_times[i, :, 1]), linestyle="dashed", color=N_colors[i])
        ax1.set_xticks(log_D, labels=D_labels)
        ax1.set_xlim([log_D[0], log_D[-1]])
        ax1.set_ylim([-1, 2.5])

        ax2.set_xlabel(r"$\log_{10}(D)$")
        ax2.set_ylabel(r"$\log_{10}(\mathrm{MP}/\mathrm{s})$")
        for i, N in enumerate(N_span):
            MP_ps = 1e-6 * N ** 2 / (1e-3 * avg_times[i, :, :])
            ax2.plot(log_D, np.log10(MP_ps[:, 0]), linestyle="solid", color=N_colors[i])
            ax2.plot(log_D, np.log10(MP_ps[:, 1]), linestyle="dashed", color=N_colors[i])
        ax2.set_xticks(log_D, labels=D_labels)
        ax2.set_xlim([log_D[0], log_D[-1]])

        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=log_D[0], vmax=log_D[-1]))
        fig.colorbar(sm, cax=cax0, orientation="vertical")
        cax0.set_ylabel(r"Domain Size $D$")
        cax0.set_yticks(log_D, labels=D_labels)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=log_N[0], vmax=log_N[-1]))
        fig.colorbar(sm, cax=cax1, orientation="vertical")
        cax1.set_ylabel(r"Image Size $N$")
        cax1.set_yticks(log_N, labels=N_labels)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=log_N[0], vmax=log_N[-1]))
        fig.colorbar(sm, cax=cax2, orientation="vertical")
        cax2.set_ylabel(r"Image Size $N$")
        cax2.set_yticks(log_N, labels=N_labels)

        # add trendlines
        fine_N_span = np.linspace(log_N[0], log_N[-1], 100)
        fine_D_span = np.linspace(log_D[0], log_D[-1], 100)
        m_N = 2
        m_D = 1
        y_int_N = -2.25
        y_int_D = -1.125
        ax0.plot(fine_N_span, fine_N_span * m_N + (y_int_N - m_N * fine_N_span[0]), color='k', linestyle="dotted", zorder=-10, alpha=0.5, label=r"$\frac{d\log \tau}{d\log N} = 2$")
        ax1.plot(fine_D_span, fine_D_span * m_D + (y_int_D - m_D * fine_D_span[0]), color='k', linestyle="dotted", zorder=-10, alpha=0.5, label=r"$\frac{d\log \tau}{d\log D} = 1$")
        ax0.legend(loc="lower right", frameon=False)
        ax1.legend(loc="lower right", frameon=False)

        plt.subplots_adjust(wspace=0)
        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        plt.close("all")

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