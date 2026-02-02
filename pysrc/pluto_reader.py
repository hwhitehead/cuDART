# python toolkit for reading particle data from PLUTO into cuDART legible .npy arrays
# this toolkit is built directly upon pre-existing routines developed by E. Elley at https://github.com/emmaelley/jet_analyst (private)

import pyPLUTO.ploadparticles as pr
import pandas as pd
import numpy as np
import os, sys, configparser, gc
import vtk
from vtk.util import numpy_support
from scipy.ndimage import gaussian_filter

class VTKLoader:
	def __init__(self,data_dir):
		"""Loads 3D vtk data.

			**Inputs**:
				data_dir -- path to the directory which has the data files\n
	
			**Outputs**:
				
				vtkload object whose keys are arrays of data values.

		"""
		self.Dt = 0.0
		self.n1 = 0
		self.n2 = 0
		self.n3 = 0
		self.x1 = []
		self.x2 = []
		self.x3 = []
		self.dx1 = []
		self.dx2 = []
		self.dx3 = []
		self.datadir = data_dir

		#reads grid file to get x1 x2 and x3 arrays 
		self.ReadGridFile(os.path.join(data_dir,'grid.out'))

		#read vtk.out file to get timing and variables
		self.ReadVarFile(os.path.join(data_dir,'vtk.out'))

		#set up variables as instance attributes
		for var in self.vars:
			setattr(self,var,[])

		#load variables in yourself (dont be lazy)
		#this is to avoid loading all the vars if you don't need them. 
		#Currently is quite a chunky method. 
		#maybe ill make this automatic when its more efficient 
		
		
	def ReadGridFile(self, gridfile):
		xL = []
		xR = []
		nmax = []
		gfp = open(gridfile, "r")
		for i in gfp.readlines():
				if len(i.split()) == 1:
						try:
								int(i.split()[0])
								nmax.append(int(i.split()[0]))
						except:
								pass
						
				if len(i.split()) == 3:
						try:
								int(i.split()[0])
								xL.append(float(i.split()[1]))
								xR.append(float(i.split()[2]))
						except:
								if (i.split()[1] == 'GEOMETRY:'):
										self.geometry=i.split()[2]
								pass
						
		self.n1, self.n2, self.n3 = nmax
			
		n1 = self.n1
		n1p2 = self.n1 + self.n2
		n1p2p3 = self.n1 + self.n2 + self.n3
		self.x1 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1)])
		self.dx1 = np.asarray([(xR[i]-xL[i]) for i in range(n1)])
		self.x2 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1, n1p2)])
		self.dx2 = np.asarray([(xR[i]-xL[i]) for i in range(n1, n1p2)])
		self.x3 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1p2, n1p2p3)])
		self.dx3 = np.asarray([(xR[i]-xL[i]) for i in range(n1p2, n1p2p3)])



		# Create the xr arrays containing the edges positions
		# Useful for pcolormesh which should use those
		self.x1r = np.zeros(len(self.x1)+1) ; self.x1r[1:] = self.x1 + self.dx1/2.0 ; self.x1r[0] = self.x1r[1]-self.dx1[0]
		self.x2r = np.zeros(len(self.x2)+1) ; self.x2r[1:] = self.x2 + self.dx2/2.0 ; self.x2r[0] = self.x2r[1]-self.dx2[0]
		self.x3r = np.zeros(len(self.x3)+1) ; self.x3r[1:] = self.x3 + self.dx3/2.0 ; self.x3r[0] = self.x3r[1]-self.dx3[0]

	def ReadVarFile(self, varfile):
		""" Read variable names from the outfiles.

		**Inputs**:

		varfile -- name of the out file which has variable information.

		"""
		vfp = open(varfile, "r")
		varinfo = vfp.readline().split()
		self.filetype = varinfo[4]
		self.endianess = varinfo[5]
		self.vars = varinfo[6:]
		vfp.close()

	def ReadTimeInfo(self, timefile, ns):
		""" Read time info from the outfiles.

		**Inputs**:

		timefile -- name of the out file which has timing information.

		"""
		f_var = open(timefile, "r")
		tlist = []
		for line in f_var.readlines():
				tlist.append(line.split())
		self.SimTime = float(tlist[ns][1])
		self.Dt = float(tlist[ns][2])

	def loadVariable(self,var, ns):
		"""loads in data from VTK file on that variable
		Reads a 3D VTK file and converts it into a NumPy array.

		Args:
			var (string): one of the instance variables: rho,prs,tr1 etc etc
		"""

		#first check that var is actually one of the vars:
		var_exists=False
		for existing_var in self.vars:
			if var==existing_var:
				var_exists=True
				break

		#makes the 0001 string from nstep
		self.NStepStr = str(ns)
		while len(self.NStepStr) < 4:
				self.NStepStr = '0'+self.NStepStr

		if var_exists:
			filename = self.datadir+'{}.{}.vtk'.format(var,self.NStepStr)

			# Initialize the VTK reader
			reader = vtk.vtkRectilinearGridReader()
			reader.SetFileName(filename)
			reader.Update()

			# Get the VTK data object from the reader
			vtk_data = reader.GetOutput()
			
			if vtk_data is None:
				raise ValueError(f"Could not read the VTK file or file is empty: {filename}")
			
			# Attempt to get point data
			vtk_array = vtk_data.GetPointData().GetScalars()
			
			if vtk_array is None:
				# If no point data, check for cell data
				vtk_array = vtk_data.GetCellData().GetScalars()
			
			if vtk_array is None:
				raise ValueError(f"No scalar data (point or cell) found in the file: {filename}")
			
			# Determine the data type of the VTK array
			vtk_type = vtk_array.GetDataType()
			numpy_type = numpy_support.get_vtk_to_numpy_typemap().get(vtk_type, None)
			
			if numpy_type is None:
				raise ValueError(f"Unsupported VTK array type {vtk_type}")
			
			# Convert the VTK array to a NumPy array
			data_array = numpy_support.vtk_to_numpy(vtk_array)
			
			# Extract the dimensions of the 3D data
			dims = vtk_data.GetDimensions()
			dims = (dims[0]-1,dims[1]-1,dims[2]-1)
			# Reshape the NumPy array to the original 3D shape
			data_array = data_array.reshape(dims, order='F')

			print(data_array)
			
			setattr(self,var,data_array)
		else:
			print('Variable does not belong to this instance.')

class Frequencies:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_file)
        self.frequencies = {}
        for key in config['frequencies']:
            self.frequencies[key] = int(config['frequencies'][key])

class Units:
    """Read in conversions to go from simulation units to defaults for plotting.

    Defaults for plotting are cgs for pressure and density, fraction of c for speeds, kpc for distances, Myr for times.
    """
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.prs = float(config['conversions']['prs'])
        self.rho = float(config['conversions']['rho'])
        self.v = float(config['conversions']['v'])
        self.tracer = float(config['conversions']['tr1'])
        self.b = float(config['conversions']['b'])
        self.undersampling_factor = float(config['conversions']['undersampling_factor'])
        self.time = float(config['conversions']['time'])

class PlutoParticleReader:

    def __init__(self, load_dir, config_file):
        self.load_dir = load_dir
        self.units = Units(config_file)
        self.all_frequencies = Frequencies(config_file).frequencies
        self.vtk_loader = self.invoke_vtk_loader()

    def invoke_vtk_loader(self):

        loader = VTKLoader(self.load_dir)

        self.xr = {1 : loader.x1r, 2 : loader.x2r, 3 : loader.x3r}
        self.dx = {1 : loader.dx1[0], 2 : loader.dx2[0], 3 : loader.dx3[0]}

        self.cell_volume = self.dx[1] * self.dx[2] * self.dx[3]
        self.volume_factor = self.cell_volume*3.0856775807e21**3

        self.domain = {}
        self.domain["xbeg"] = self.xr[1].min()
        self.domain["xend"] = self.xr[1].max()
        self.domain["ybeg"] = self.xr[2].min()
        self.domain["yend"] = self.xr[2].max()
        self.domain["zbeg"] = self.xr[3].min()
        self.domain["zend"] = self.xr[3].max()
        return loader

    def emm_to_npy(self, snapshot_num, save_dir, frequencies = ["1000MHz"], sparse_step = 10, verbose = True, num_pfiles = None, apply_blur = False, blur_kwargs = None, apply_mirror = True, force_gc = True, apply_boost = False):

        if not isinstance(frequencies, list):
            if frequencies.lower == "all":
                frequncies = self.all_frequencies
            else:
                raise Exception("invalid pass to 'frequencies', accept list or 'all'")

        # Calculate the midpoints of the cells in each direction
        midpoints1 = ((self.xr[1][::sparse_step][:-1] + self.xr[1][::sparse_step][1:]) / 2.0).round(3)
        midpoints2 = ((self.xr[2][::sparse_step][:-1] + self.xr[2][::sparse_step][1:]) / 2.0).round(3)
        midpoints3 = ((self.xr[3][::sparse_step][:-1] + self.xr[3][::sparse_step][1:]) / 2.0).round(3)

        if num_pfiles is None:
            num_pfiles = self.count_particle_files(snapshot_num)

        if apply_boost:
            vel_strs = ["vx1", "vx2", "vx3"]
            for i in range(num_pfiles):  
                P = pr.ploadparticles(ns=snapshot_num, w_dir=self.load_dir, datatype='flt', ptype='LP',chnum=i)  # Should be safe to change this to other datatypes, but untested.
                emissivities = []
                for frequency in frequencies:
                    emm_local = np.reshape(P.color[:, self.all_frequencies["J_" + frequency]], P.x1.shape)
                    emm_local = emm_local * self.units.undersampling_factor * self.volume_factor
                    emissivities.append(emm_local)

                if i == 0:
                    particles = pd.DataFrame(np.array([P.x1, P.x2, P.x3] + [P.__dict__[var] for var in vel_strs] + emissivities).T,
                                                columns=["x1", "x2", "x3"] + vel_strs + ["emm_freq_" + frequency[2:] for frequency in frequencies], dtype=pd.Float32Dtype())
                else:
                    particles_section = pd.DataFrame(np.array([P.x1, P.x2, P.x3] + [P.__dict__[var] for var in vel_strs] + emissivities).T,
                                                columns=["x1", "x2", "x3"] + vel_strs + ["emm_freq_" + frequency[2:] for frequency in frequencies], dtype=pd.Float32Dtype())
                    particles = pd.concat([particles, particles_section])  # Append the particles from this file to the existing DataFrame

            # Separate the particles by position into cells and label these cells by their midpoints.
            particles['x1bin'] = pd.cut(particles.x1, self.xr[1][::sparse_step], labels=midpoints1).astype(float).round(3)
            particles['x2bin'] = pd.cut(particles.x2, self.xr[2][::sparse_step], labels=midpoints2).astype(float).round(3)
            particles['x3bin'] = pd.cut(particles.x3, self.xr[3][::sparse_step], labels=midpoints3).astype(float).round(3)

            # Average over the particles in each cell and create dataframe of the results. If a cell has no particles we fill the cell with 0, because we are about to add it to an array of zeros
            
            # extract velocity data
            vel_npy_data = {}
            for vel_str in ["vx1", "vx2", "vx3"]:
                piv = pd.pivot_table(particles, index='x3bin', columns=['x1bin', 'x2bin'], aggfunc={vel_str : 'mean'}).fillna(0.0)
                piv.columns = piv.columns.droplevel(0)

                # To get the shape of the numpy array consistent for every frame, we first make an array of zeros of the correct shape and then add the pivot table made in the last step to this
                X, Y = np.meshgrid(midpoints1, midpoints2)
                every_XY_pair = [(float(X[i, j]), float(Y[i, j])) for i in range(X.shape[0]) for j in range(X.shape[1])]
                zeros = pd.DataFrame(0, columns=every_XY_pair, index=midpoints3)
                piv.columns = piv.columns.to_flat_index()
                piv_full = zeros.add(piv, fill_value=0.0)
                if force_gc:
                    del piv
                    gc.collect()
                piv_full.index.set_names('x3bin', inplace=True)
                piv_full.columns.set_names('(x1bin,x2bin)', inplace=True)

                vel_data = piv_full.to_numpy(dtype=np.float32)
                if force_gc:
                    del piv_full
                    gc.collect()
                dim = np.shape(vel_data)[0]
                vel_data = vel_data.reshape((dim,dim,dim))
                vel_data = np.einsum("kji->ijk", vel_data)

                # apply post-processing
                if apply_blur:
                    vel_data = self.blur_data(vel_data, blur_kwargs)

                if apply_mirror:
                    if vel_str == "vx3":
                        vel_data = self.mirror_data(vel_data, flip = True)
                    else:
                        vel_data = self.mirror_data(vel_data)

                vel_npy_data[vel_str] = vel_data
            if verbose:
                print("extracted velocity data")
        
        # extract emissivity data
        for frequency in frequencies: 
            piv = pd.pivot_table(particles, index='x3bin', columns=['x1bin', 'x2bin'], aggfunc={"emm_freq_" + frequency[2:]: 'mean'}).fillna(0.0)
            piv.columns = piv.columns.droplevel(0)

            # To get the shape of the numpy array consistent for every frame, we first make an array of zeros of the correct shape and then add the pivot table made in the last step to this
            X, Y = np.meshgrid(midpoints1, midpoints2)
            every_XY_pair = [(float(X[i, j]), float(Y[i, j])) for i in range(X.shape[0]) for j in range(X.shape[1])]
            zeros = pd.DataFrame(0, columns=every_XY_pair, index=midpoints3)
            piv.columns = piv.columns.to_flat_index()
            piv_full = zeros.add(piv, fill_value=0.0)
            if force_gc:
                del piv
                gc.collect()
            piv_full.index.set_names('x3bin', inplace=True)
            piv_full.columns.set_names('(x1bin,x2bin)', inplace=True)

            emm_data = piv_full.to_numpy(dtype=np.float32)
            if force_gc:
                del piv_full
                gc.collect()
            dim = np.shape(emm_data)[0]
            emm_data = emm_data.reshape((dim,dim,dim))
            emm_data = np.einsum("kji->ijk", emm_data)

            # post processes
            if apply_blur:
                emm_data = self.blur_data(emm_data)

            if apply_mirror:
                emm_data = self.mirror_data(emm_data)
                
            save_str = os.path.join(save_dir, "emm_" + frequency + ".npy")
            if apply_boost:
                # stack with velocity data and save
                boosted_shape = np.append(np.array(np.shape(emm_data)), [4])
                boosted_data = np.zeros(shape=boosted_shape, dtype=np.float32)
                boosted_data[..., 0] = emm_data
                boosted_data[..., 1] = 0.0 #np.abs(vel_npy_data["vx1"])
                boosted_data[..., 2] = 0.0 #np.abs(vel_npy_data["vx2"])
                boosted_data[..., 3] = 0.0 #np.abs(vel_npy_data["vx3"])
                np.save(save_str, boosted_data.astype(np.float32))
            else:
                # save emissivity alone
                np.save(save_str, emm_data.astype(np.float32))
            
            if verbose:
                print("saved emissivity data at " + frequency)

    def count_particle_files(self, snapshot_num, num_zfill = 4):

        file_list = os.listdir(self.load_dir)
        sub_string = "particles." + str(snapshot_num).zfill(num_zfill)
        particle_file_count = sum(1 for file in file_list if sub_string in file)
        return particle_file_count

    def mirror_data(self, input_ar, flip = False):

        input_shape = np.array(np.shape(input_ar))
        mirrored_shape = input_shape * np.array([1,1,2])
        mirrored_ar = np.zeros(shape=mirrored_shape, dtype=np.float32)
        mirrored_ar[:,:,input_shape[2]:] = input_ar
        if flip:
            mirrored_ar[:,:,:input_shape[2]] = -input_ar[:,:,::-1]
        else:
            mirrored_ar[:,:,:input_shape[2]] = input_ar[:,:,::-1]
        return mirrored_ar

    def blur_data(self, input_ar, blur_kwargs = None):
        
        if blur_kwargs is None:
            sigma = 2
            window = 2
        else:
            try:
                sigma = blur_kwargs["sigma"]
                window = blur_kwargs["window"]
            except:
                raise Exception("blur_kwarg must be a dictionary containing keyed values for 'sigma' and 'window'")
        truncate = (((window) / 2) - 0.5) / sigma
        return gaussian_filter(input_ar, sigma = sigma, truncate = truncate)

        

        
    