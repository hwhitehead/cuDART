import numpy as np
import sys, os
import h5py

class BlackHole:
    """
    this class loads BH data for a single BH into memory from a txt or npy file, labelling and retaining header info
    """
    def __init__(self, bh_index, inp, archive=False): # old n_data = 16 for archival data
        # pass user input, accept inp as path to bh_data.txt or preloaded npy object
        if archive:
            n_data = 15 # TEMP
            self.var_list = ["m", "x", "y", "z", "vx", "vy", "vz", "ax_gas", "ay_gas", "az_gas", "ax_acc", "ay_acc", "az_acc", "J", "m_obs"]# TEMP, "m_acc"]
        else:
            n_data = 23
            self.var_list =  ["m", "x", "y", "z", "vx", "vy", "vz"]
            self.var_list += ["ax_gas", "ay_gas", "az_gas", "ax_acc", "ay_acc", "az_acc"]
            self.var_list += ["m_app", "m_nom", "m_acc", "t_init"] 
            self.var_list += ["J", "Jx", "Jy", "Jz"]
            self.var_list += ["edd_frac", "eta_j"]
        
        if isinstance(inp, str):
            bh_data = np.load(inp)
            # load header data
            self.head_list = ["b", "rho0", "Omega0", "xl", "xr", "yl", "yr", "zl", "zr"]
            for i, head in enumerate(self.head_list):
                setattr(self, head, bh_data[0, i])
            bh_data = bh_data[1:, :]

            # load bh data
            self.t = bh_data[:, 0]
            js = 1 + bh_index * n_data
            for i, var in enumerate(self.var_list):
                setattr(self, var, bh_data[:, js + i])

            del bh_data
        else:
            # try:
            self.Omega0 = inp.Omega0
            self.t = getattr(inp, "t")
            self.m = getattr(inp, "m{0}".format(bh_index+1))
            self.x = getattr(inp, "x{0}".format(bh_index+1))
            self.y = getattr(inp, "y{0}".format(bh_index+1))
            self.z = getattr(inp, "z{0}".format(bh_index+1))
            self.vx = getattr(inp, "vx{0}".format(bh_index+1))
            self.vy = getattr(inp, "vy{0}".format(bh_index+1))
            self.vz = getattr(inp, "vz{0}".format(bh_index+1))
            empty = np.zeros_like(self.m)
            self.ax_gas = empty
            self.ay_gas = empty
            self.az_gas = empty
            self.ax_acc = empty
            self.ay_acc = empty
            self.az_acc = empty
            self.J = empty
            self.m_obs = np.array(self.m)
            # except:
            #     print("unable to parse NBody object as input")

    def clip(self, start_index=None, stop_index=None):

        if start_index is not None:
            if stop_index is None:
                slicer = np.s_[start_index:]
            else:
                slicer = np.s_[start_index:stop_index]
        elif stop_index is not None:
            slicer = np.s_[:stop_index]
        else:
            return

        self.t = self.t[slicer]
        for attr in self.var_list:
            setattr(self, attr, getattr(self, attr)[slicer])

class AthenaData:
    """
    this class loads single HDF5 snapshots into python memory retaining labels
    it also features regularisation routines to compile MeshBlocks into homogenous meshes
    """

    def __init__(self, h_str, user_vars="T"):
        print("Loading {0} as AthenaData".format(h_str))
        self.coord_str = ["x", "y", "z"]
        # rename variables for user ease
        self.variable_dict = {
            # primitive
            "rho": "rho",
            "press": "P",
            "vel1": "vx",
            "vel2": "vy",
            "vel3": "vz",
            "r0": "C_J",
            # conservative
            "dens": "rho",  # degenerate for non-relativistic simulations
            "Etot": "E",
            "mom1": "Mx",
            "mom2": "My",
            "mom3": "Mz",
            "s0": "DC_J",
        }
        # accept user specified labels for user_out_var
        if user_vars is not None:
            for i, user_var in enumerate(user_vars):
                self.variable_dict.update({"user_out_var" + str(i): user_var})
        # load data from HDF5 file
        with h5py.File(h_str, 'r') as f:
            # copy topline attributes
            for attr in list(f.attrs):
                setattr(self, attr, f.attrs.get(attr))
            # copy coordinates
            for attr in ("x1f", "x2f", "x3f", "x1v", "x2v", "x3v", "Levels", "LogicalLocations"):
                setattr(self, attr, np.array(f[attr]))
            # dupe coordinates for ease of access
            self.x = self.x1v
            self.y = self.x2v
            self.z = self.x3v
            # copy hydro data
            variable_names = np.array([x.decode("ascii", "replace") for x in f.attrs["VariableNames"][:]])
            dataset_sizes = f.attrs['NumVariables'][:]
            dataset_names = np.array([x.decode('ascii', 'replace') for x in f.attrs['DatasetNames'][:]])
            for dataset_index, dataset_name in enumerate(dataset_names):
                variable_begin = sum(dataset_sizes[:dataset_index])
                variable_end = variable_begin + dataset_sizes[dataset_index]
                variable_names_local = variable_names[variable_begin:variable_end]
                for variable_index, variable_name in enumerate(variable_names_local):
                    if variable_name in self.variable_dict: # if new label exists, relabel
                        attr_name = self.variable_dict[variable_name]
                    else:
                        attr_name = variable_name
                    setattr(self, attr_name, np.array(f[dataset_name][variable_index, ...]))

    def homogenize(self, level=None, homo_vars=None, verbose = False, bounds=None):
        mb_size = self.MeshBlockSize
        max_level = np.max(self.Levels)
        # test restriction limits
        if level is None:
            level = max_level
        if level > max_level:
            warnings.warn("target level {0} exceeds maximum mesh level {1}".format(level, max_level))
        else:
            max_restrict = 2 ** (max_level - level)
            for d in range(0, 3):
                if mb_size[d] != 1 and mb_size[d] < max_restrict:
                    limit = max_level - int(np.log2(mb_size[d]))
                    warnings.warn(
                        "target level " + str(level) + " too low for restriction routine, must be >= " + str(limit))

        # define dimensions of output array
        nx_vals = []
        for d in range(3): # handle slicing here?
            if mb_size[d] == 1: # do not expand along unexpanded dimension
                nx_vals.append(self.RootGridSize[d])
            else:
                nx_vals.append(self.RootGridSize[d] * 2 ** level)
        nx1 = nx_vals[0]
        nx2 = nx_vals[1]
        nx3 = nx_vals[2]

        if verbose:
            print("starting homogenization routine...")
            print("root grid                            [nk, nj, ni] = [{0}, {1}, {2}]".format(*self.RootGridSize[-1::-1]))
            print("max mesh level                                    = {0}".format(max_level))
            print("homogenous level                                  = {0}".format(level))
            print("master grid                          [nk, nj, ni] = [{0}, {1}, {2}]".format(nx3, nx2, nx1))

        # populate coordinate arrays
        data = {}
        for d, (nx, c) in enumerate(zip(nx_vals, self.coord_str)):
            xmin = getattr(self, "RootGridX" + str(d+1))[0]
            xmax = getattr(self, "RootGridX" + str(d+1))[1]
            data[c] = np.linspace(xmin, xmax, nx+1)

        # account for domain selection
        index_lim = np.zeros(shape=(3,2), dtype=np.int32)
        index_lim[0, 1] = nx1
        index_lim[1, 1] = nx2
        index_lim[2, 1] = nx3
        trims = np.array([False, False, False])
        slices = np.array([False, False, False])
        err_string = "{0} must be {1} than {2} in order to overlap domain"
        if np.any(bounds is not None): # trim domain to spec
            # test user input
            if np.shape(bounds) != (3, 2):
                raise Exception("Invalid pass to bounds, require input shape (3,2)")
            # test if bounds in domain
            for d, c in enumerate(self.coord_str):
                bound = bounds[d, :]
                if bound[0] is not None and bound[0] >= data[c][0]:
                    if bound[0] >= data[c][-1]:
                        raise Exception(err_string.format(c + "_min", "less", data[c][-1]))
                    index_lim[d, 0] = np.where(data[c] <= bound[0])[0][-1]
                    trims[d] = True
                if bound[1] is not None and bound[1] <= data[c][-1]:
                    if bound[1] <= data[c][0]:
                        raise Exception(err_string.format(c, "_max", "greater", data[c][0]))
                    index_lim[d, 1] = np.where(data[c] >= bound[1])[0][0]
                    trims[d] = True
                if nx_vals[d] != 1: # if extended dimension, check for slice
                    if bound[0] == bound[1] and (bound[0] is not None) and (bound[1] is not None): # select slice
                        slices[d] = True
                        index_lim[d, 1] += 1 # bump to allow for single value

        # trim data arrays
        for d, c in enumerate(self.coord_str):
            if trims[d]:
                data[c] = data[c][index_lim[d, 0]:index_lim[d, 1] + 1]

        # unpack indices
        i_min = index_lim[0, 0]
        i_max = index_lim[0, 1]
        j_min = index_lim[1, 0]
        j_max = index_lim[1, 1]
        k_min = index_lim[2, 0]
        k_max = index_lim[2, 1]

        # identify output variables to merge
        if homo_vars is None:  # merge all variables
            homo_vars = []
            for q, (x, variable_name) in enumerate(self.variable_dict.items()):
                if hasattr(self, variable_name):
                    homo_vars.append(variable_name)
        elif isinstance(homo_vars, str):  # accept single variable specified
            homo_vars = [homo_vars]
        elif not (isinstance(homo_vars, list) or isinstance(homo_vars, np.ndarray)): # accept list of str
            raise Exception("invalid pass to homo_vars")

        # purge improper variables
        for merge_var in homo_vars:
            if not hasattr(self, merge_var):
                homo_vars.remove(merge_var)
                print("removing {0}" + str(merge_var) + " from list")

        # build output array
        for merge_var in homo_vars:
            data.update({merge_var: np.zeros((k_max - k_min, j_max - j_min, i_max - i_min))})

        if verbose:
            print("apply bounds [xmin, xmax, ymin, ymax, zmin, zmax] = [{0}, {1}, {2}, {3}, {4}, {5}]".format(*np.ravel(bounds)))
            print("homogenous grid                      [nk, nj, ni] = [{0}, {1}, {2}]".format(k_max - k_min, j_max - j_min, i_max - i_min))
            print("homogenizing hydro variables                       ", homo_vars)

        # iterate over mb
        for mb_num in range(self.NumMeshBlocks):
            mb_level = self.Levels[mb_num]
            mb_location = self.LogicalLocations[mb_num, :]

            # apply prolongation to coarse, copy same-level
            if mb_level <= level:
                # scale multiplier
                s = 2 ** (level - mb_level)
                # destination indices in merged
                il_d = (mb_location[0] * mb_size[0] * s
                        if nx1 > 1 else 0)
                jl_d = (mb_location[1] * mb_size[1] * s
                        if nx2 > 1 else 0)
                kl_d = (mb_location[2] * mb_size[2]* s
                        if nx3 > 1 else 0)
                iu_d = il_d + mb_size[0] * s if nx1 > 1 else 1
                ju_d = jl_d + mb_size[1] * s if nx2 > 1 else 1
                ku_d = kl_d + mb_size[2] * s if nx3 > 1 else 1

                # Calculate (prolongated) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min

                # insert values
                for merge_var in homo_vars:
                    mb_data = getattr(self, merge_var)[mb_num, ...]
                    if s > 1: # level != mb_level, prolongate data and insert
                        if nx1 > 1:
                            mb_data = np.repeat(mb_data, s, axis=2)[:, :, il_s:iu_s]
                        if nx2 > 1:
                            mb_data = np.repeat(mb_data, s, axis=1)[:, jl_s:ju_s, :]
                        if nx3 > 1:
                            mb_data = np.repeat(mb_data, s, axis=0)[kl_s:ku_s, :, :]
                        data[merge_var][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = mb_data
                    else: # level match, insert directly
                        data[merge_var][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = mb_data[kl_s:ku_s,
                                                                   jl_s:ju_s,
                                                                   il_s:iu_s]
            else: # restrict fine data
                # Calculate scale
                s = 2 ** (mb_level - level)

                # Calculate destination indices, without selection
                il_d = mb_location[0] * mb_size[0] // s if nx1 > 1 else 0
                jl_d = mb_location[1] * mb_size[1] // s if nx2 > 1 else 0
                kl_d = mb_location[2] * mb_size[2] // s if nx3 > 1 else 0
                iu_d = il_d + mb_size[0] // s if nx1 > 1 else 1
                ju_d = jl_d + mb_size[1] // s if nx2 > 1 else 1
                ku_d = kl_d + mb_size[2] // s if nx3 > 1 else 1

                # Calculate (restricted) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min

                # Account for restriction in source indices
                if nx1 > 1:
                    il_s *= s
                    iu_s *= s
                if nx2 > 1:
                    jl_s *= s
                    ju_s *= s
                if nx3 > 1:
                    kl_s *= s
                    ku_s *= s

                # Apply subsampling
                # Calculate fine-level offsets (nearest cell at or below center)
                o1 = s // 2 - 1 if nx1 > 1 else 0
                o2 = s // 2 - 1 if nx2 > 1 else 0
                o3 = s // 2 - 1 if nx3 > 1 else 0

                # Assign values
                for merge_var in homo_vars:
                    data[merge_var][kl_d:ku_d,
                    jl_d:ju_d,
                    il_d:iu_d] = getattr(self, merge_var)[mb_num, kl_s + o3:ku_s:s,
                                 jl_s + o2:ju_s:s, il_s + o1:iu_s:s]

        if verbose:
            print("finished homogenizing.")

        return data

    def build_mesh(self, data_dir, verbose = False, nzfill = None, homogenize = False, origin = np.array([0.0, 0.0, 0.0]), bounds = None, tracer_type="P", homo_level=3):

        if nzfill is None:
            nzfill = int(np.ceil(np.log10(self.NumMeshBlocks)))
        mesh = Mesh(data_dir, nzfill = nzfill)

        # determine bounds
        if bounds is None:
            bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        else:
            bounds = np.array(bounds)
        for dim in range(3):
            bounds[dim][:] += origin[dim]

        # import data
        if homogenize:
            if tracer_type in self.variable_dict.values():
                homo_data = self.homogenize(level = homo_level, homo_vars=[tracer_type], bounds=bounds, verbose = verbose)
                mb_data = homo_data[tracer_type]
            elif tracer_type == "vel_z":
                homo_data = self.homogenize(level = homo_level, homo_vars=["vz"], bounds=bounds, verbose = verbose)
                mb_data = np.abs(homo_data["vz"])
            else:
                raise Exception("did not recognise pass to tracer_type")
            # important: transpose data with label intact
            mb_data = np.array(np.transpose(mb_data), order="C")
            xl = np.array([np.min(homo_data["x"]), np.min(homo_data["y"]), np.min(homo_data["z"])])
            xr = np.array([np.max(homo_data["x"]), np.max(homo_data["y"]), np.max(homo_data["z"])])
            mesh.add_meshblock(mb_data, xl, xr)
        else:
            for n in range(0, self.NumMeshBlocks):
                xl = np.array([self.x1f[n,0], self.x2f[n,0], self.x3f[n,0]])
                xr = np.array([self.x1f[n,-1], self.x2f[n,-1], self.x3f[n,-1]])
                if not self.in_bounds(xl, xr, bounds): continue

                if tracer_type in self.variable_dict.values():
                    mb_data = getattr(self, tracer_type)[n, ...]
                elif tracer_type == "vel_z":
                    mb_data = np.abs(getattr(self, "vz")[n, ...])
                else:
                    raise Exception("did not recognise pass to tracer_type")
                # important: transpose data with label intact
                mb_data = np.array(np.transpose(mb_data), order="C")
                mesh.add_meshblock(mb_data, xl, xr)

        mesh.write_header()
        return mesh

    def in_bounds(self, xl, xr, bounds):
        for dim in range(np.size(xl)):
            if xl[dim] > bounds[dim][1]:
                return False
            elif xr[dim] < bounds[dim][0]:
                return False

        return True 