# cuDART: CUDA + DDA Accelerated Ray Tracing (v0.9)

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/lookback/gallery/superluminal.gif width="800" alt=animated/>
</p>
<p align="center"">
  <em> Animation showing synthetic radio observations of relativistic twin ejecta launched at angles of &pi/2 &pi/4 to the line-of-sight (left and right respectively). In both cases, each blob has the same absolute velocity, but appear to move differently. cuDART automatically accounts for relativistic beaming (the ejectum pointed toward the observer is brighter) and superluminal motion (as the approaching ejectum travels towards the observer, its apparent transverse velocity exceeds the speed of light).</em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/lookback/gallery/magnetised_jets.png width = "600"/>
</p>
<p align="center"">
  <em> Static images of a highly magnetised, variable power jet launched from an Active Galactic Nucleus, viewed from three different orientations. Relativistic beaming results in a brighter advancing jet and dimmer receding jet; this effect is strongest when the jet is more closely aligned with the line-of-sight. Simulation data produced as part of paper currently in prep.</em>
</p>

This repository contains a lightweight set of tools for raytracing heterogenous orthogonal meshes, intended for visualisation of line-of-sight quantities in simulated data, such as optically thin emission, surface density etc. Such visualisations, especially from arbitrary viewpoints, have the potential to be very expensive due to the large number of cells that a line-of-sight may intersect with. In `cuDART` two acceleration structures are implemented to triviliase this computation: GPU acceleration and DDA, the Digital Differential Analyzer. DDA allows for iterative low-cost propagation of rays through regular meshes, previously implemented in Python [here](https://github.com/hwhitehead/DART), but now utilising the CUDA toolkit to perform ray propagation and summation exceptionally quickly. As well as properties independent of line-of-sight, such as density, `cuDART` supports relativistic beaming of emissivity when provided with velocity data. `cuDART` supports using a finite-speed-of-light in intensity calculations, allowing for geometric effects such as superluminal motion to be recovered. The workhorse of the code is written in C++/CUDA, but Python scripts are provided for user ease on the frontend. 

## Usage

Broadly speaking, `cuDART` can be split into a frontend system (written in Python) and a backend (written in C++/CUDA). The user can interact with the backend directly, or use Python to make, setup, run the executable and generate plotted images. 

### Inputs
On each render call, `cuDART` will trace either a single file, representing a single mesh with homogenous spatial resolution (unlabeled mode), or a directory containing multiple files, each containing their own homogenous-resolution meshblock, though resolution between meshblocks can be different (labelled mode). If multiple files are read, information as to the spatial distribution of these meshblocks must also be provided, see `render_unlabelled_example`, `build_labelled_example` and `render_labelled_example` for demonstration. If the user wants to implement a finite communication time for the trace e.g. using a finite speed of light for a intensity calculation, then the input to `cuDART` should be a directory containing multiple snapshots in time. Each snapshot may be labelled or unlabelled e.g. a single file or a subdirectory.

## Data Formats
`cuDART` accepts data in the form of `.npy` files, which *must* contain an array of `float32` entries. If not flagged for relativisitic beaming, the array should be of shape `(nx,ny,nz)`, with each entry populated by data to be traced at each spatial position. If relativistic beaming is included, an extra rank is required to pass data to trace, and velocity data in units of the the speed of light in each cardinal direction e.g. as $\beta_x = v_x / c$. The resulting array will have shape `(nx,ny,nz,4)` where the fourth index covers `(tracer,beta_x,beta_y,beta_z)` for each spatial position. In unlabelled mode, `cuDART` will read a single homogenous `.npy` file, assuming equal spacing in all directions. In labelled mode, `cuDART` accepts an arbitrary number of `.npy` files with different mesh resolutions. Labelled mode requires a header file to specific spatial locations of subgrids (see [examples.py](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py)).

## Performance
`cuDART` is bottlenecked primarily by I/O; the actual tracing of meshes and image writes are performed exceptionally quickly. The main overhead occurs at executable intialisation, due to the cost of launching a GPU context and reading the `.npy` file(s) into host memory, usually taking O(1)s. CUDA-type operations are MUCH faster, copying the data into device memory and generating an image from this data takes only O(100)ms. For sensible image dimsions, the data transfer to device is more expensive than the trace operation itself. As such, peak efficiency with `cuDART` is achieved when many images are taken using the same data set e.g. many lines-of-sight. Comparing a 100 image render to a single image, `cuDART` transitions from 16% of the runtime attributed to the render kernel up to 95%. See [this](https://github.com/hwhitehead/cuDART/blob/lookback/gallery/profiling.png) figure for a comparison of runtimes across image and domain size, and with/without relativistic boosting. Users acting as admin may wish to look into [persistence daemons](https://docs.nvidia.com/deploy/driver-persistence/overview.html) for the NVIDIA kernel; especially for short/simple renders the majority of the runtime can be occupied by the application start latency which can be bypassed by ensuring a kernel persists between executions.  

### Comparison to DART

Performance comparisons can be made with a pervious implementation of the DDA algorithm, the single-core Pythonic `DART` (hosted privately [here](https://github.com/hwhitehead/DART)). Extrpolating scaling tests performed on `DART`, tracing a $2048^2$$ image from a $750^3$ image would take O(100)s. The same trace in `cuDART` takes O(100)ms, a factor 1000 speedup. This marks an extreme case, as the high image resolution supersamples the domain resolution. `DART` itself uses `numba`, a Python JIT compiler to accelerate its own calculations, without this module the code is 100 times slower. As such `cuDART` boasts a $10^3$ efficiency boost over JIT-compiled `DART`, and a $10^5$ boost over the raw Python.

`DART` features routines not present in the current `cuDART` implementation, such as the ability to "bake" rays and reuse them on different domains with identical dimensions. This allowed for a 30-60% saving in render cost for repeated imaging of different domains. Such features are not present in `cuDART` primarily because baking significantly increases the memory overhead which is more problematic for GPU codes. Additional costs associated with the absence of this feature are efficiently amortized by the efficiency of the render algorithm.

## Technical Notes

### Requirements

To run `cuDART`, the following is required:
- A CUDA-capable GPU
- The `nvcc` CUDA compiler
- Python (optional, but recommended frontend, requires basic libraries such as `numpy` and `matplotlib`)

### Portability

The user can run [configure.py]() which uses the `-gencode` flag at make to avoid just-in-time ([JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation)) machine-specification code compilation. Passing the `--arch` or `--gpu` arguments to `configure.py` allow the user to target their specific GPU type or architecture; if no argument is passed then GPU agnostic code will be pre-compiled. `cuDART` is not compatible with GPU architectures pre-dating Ampere e.g. Volta/Pascal, but is compatible with all modern architectures Ampere onwards.

### Inherited Libraries
To support interaction with commonly used simulation file types, this repository uses the [libnpy](https://github.com/llohse/libnpy) library to support the import of `.npy` files. In addition to the verbatim use of this libary, much of the code structure has been informed by pre-existing publically available codebases. Most notably, as with the original Pythonic [DART](https://github.com/hwhitehead/DART) repository, the underlying DDA algorithm was written with help of [this](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/grid.html) excellent guide on acceleration structures in C. [This](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) developer blog on raytracing in CUDA helped introduce me to memory management and CUDA Makefiles, though the primary Makefile structure is actually inherited from the [Athena++](https://github.com/PrincetonUniversity/athena) repository. 

### Development

While `cuDART` is functional and tested in its current form, development on this code is ongoing. The latest code version is v0.9, v1.0 will be delayed until documentation has been improved.

