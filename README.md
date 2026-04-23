# cuDART: CUDA + DDA Accelerated Ray Tracing (v0.7)

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/comp.gif width="800" alt=animated/>
</p>
<p align="center"">
  <em> Animation depicting multiple views for a jet launched from an Active Galactic Nucleus, showing images rendered with unboosted and boosted data. Top panel shows that as the orientation of the jet changes, the unboosted luminosity is fixed but the boosted luminosity varies. Inset panel depicts a real observation of Hercules A. Simulation data featured in <a href="https://ui.adsabs.harvard.edu/abs/2026MNRAS.tmp..127E/abstract">this paper</a>.</em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/append/docs/magnetised_jet.png width = "600" alt=animated/>
</p>
<p align="center"">
  <em> Static iamges of a highly magnetised, variable power jet launched from an Active Galactic Nucleus, viewed from three different orientations. Relativistic beaming results in a brighter advancing jet and dimmer receding jet; this effect is strongest when the jet is more closely aligned with the line-of-sight. Simulation data produced as part of paper currently in prep.</em>
</p>


<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/rotate.gif width = "600" alt=animated/>
</p>
<p align="center"">
  <em> Animation of a supernova-jet simulation snapshot, featuring 200 viewpoints each yielding a 2048<sup>2</sup> image. Raw image data for each frame generated in ~150ms, during which 4 million rays were cast through a simulation domain hosting over 400 million (750<sup>3</sup>) cells. Simulation data featured in <a href="https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.4011G/abstract">this paper.</a></em>
</p>

This repository contains a lightweight set of tools for raytracing heterogenous orthogonal meshes, intended for visualisation of line-of-sight quantities in simulated data, such as optically thin emission, surface density etc. Such visualisations, especially from arbitrary viewpoints, have the potential to be very expensive due to the large number of cells that a line-of-sight may intersect with. In `cuDART` two acceleration structures are implemented to triviliase this computation: GPU acceleration and DDA, the Digital Differential Analyzer. DDA allows for iterative low-cost propagation of rays through regular meshes, previously implemented in Python [here](https://github.com/hwhitehead/DART), but now utilising the CUDA toolkit to perform ray propagation and summation exceptionally quickly. As well as properties independent of line-of-sight, such as density, `cuDART` supports relativistic beaming of emissivity when provided with velocity data. The workhorse of the code is written in C++/CUDA, but Python scripts are provided for user ease on the frontend. 

## Usage

Broadly speaking, `cuDART` can be split into a frontend system (written in Python) and a backend (written in C++/CUDA). The user can interact with the backend directly, or use Python to make, setup, run the executable and generate plotted images. 

### Frontend
Exemplar usage of the Python frontend is included as [examples.py](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py), implementing classes and routines imported from [cudart.py](https://github.com/hwhitehead/cuDART/blob/main/pysrc/cudart.py). `cuDART` operates in two modes, able to read "unlabelled data" in single homogenous meshes, or "labelled data" where an arbitrary number of subregions of heterogenous resolution can be render simultaneously. For simple implementations of these two modes see `render_unlabelled_example`, `build_labelled_example` and `render_labelled_example`. Further documentation is included [here](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py) 

### Backend
The `bin/cudart` executable accepts the following flags:
- `-i <dir>`    specifies the preperatory direction to read
- `-c <file>`   specifies the input `.txt` file specifying the camera(s) (dimension, position and orientation)
- `-s <file>`   specifies the raw img `.npy` save location (appended numerically for multiple traces)
- `-r`          flags render for relativistic beaming (requires velocity data)
- `-m <value>`  species the maximum allowed VRAM usage
- `-v`          flags for verbose execution (prints progress to stdout)

Upon execution:
1. Data to visualise is loaded from the input `.npy` file to the host, and copied to the device
2. Memory is allocated on the device to store the image data
3. Containers for the data (`MeshBlock`) and cameras (`Camera`) are allocated and initiliased
4. The `render` kernel is called, calculating values for each pixel on the device
5. The populated image buffer is copied to the host, and saved to the output `.npy` file
6. Steps 4 and 5 are repeated for all cameras specified in the `.txt` file
7. The program frees all associated memory registers (device and host) and terminates 

## Data Formats
`cuDART` accepts data in the form of `.npy` files, which *must* contain an array of `float32` entries. If not flagged for relativisitic beaming, the array should be of shape `(nx,ny,nz)`, with each entry populated by data to be traced at each spatial position. If relativistic beaming is included, an extra rank is required to pass data to trace, and velocity data in units of the the speed of light in each cardinal direction e.g. as $\beta_x = v_x / c$. The resulting array will have shape `(nx,ny,nz,4)` where the fourth index covers `(tracer,beta_x,beta_y,beta_z)` for each spatial position. In unlabelled mode, `cuDART` will read a single homogenous `.npy` file, assuming equal spacing in all directions. In labelled mode, `cuDART` accepts an arbitrary number of `.npy` files with different mesh resolutions. Labelled mode requires a header file to specific spatial locations of subgrids (see [examples.py](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py)).

## Performance
`cuDART` is bottlenecked primarily by I/O; the actual tracing of meshes and image writes are performed exceptionally quickly. The main overhead occurs at executable intialisation, due to the cost of launching a GPU context and reading the `.npy` file(s) into host memory, usually taking O(1)s. CUDA-type operations are MUCH faster, copying the data into device memory and generating an image from this data takes only O(100)ms. For sensible image dimsions, the data transfer to device is more expensive than the trace operation itself (see profiling [here](https://github.com/hwhitehead/cuDART/blob/main/docs/profiling.txt)). As such, peak efficiency with `cuDART` is achieved when many images are taken using the same data set e.g. many lines-of-sight. Comparing a 100 image render to a single image, `cuDART` transitions from 16% of the runtime attributed to the render kernel up to 95%. Users acting as admin may wish to look into [persistence daemons](https://docs.nvidia.com/deploy/driver-persistence/overview.html) for the NVIDIA kernel; especially for short/simple renders the majority of the runtime can be occupied by the application start latency which can be bypassed by ensuring a kernel persists between executions.  

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/profiling.png width = "600" alt=animated/>
</p>
<p align="center"">
  <em> Scaling tests varying the size of both the domain and image, for a single homogenous meshblock. For reasonable sized domains, render runtime (per frame) scales linearly with both the side length of the domain, and number of pixels in the image. Adding boosting to the calculation increases runtime by 0.5dex.</a></em>
</p>

### Comparison to DART

Performance comparisons can be made with a pervious implementation of the DDA algorithm, the single-core Pythonic `DART` (hosted privately [here](https://github.com/hwhitehead/DART)). Extrpolating scaling tests performed on `DART`, tracing a $2048^2$$ image from a $750^3$ image would take O(100)s. The same trace in `cuDART` takes O(100)ms, a factor 1000 speedup. This marks an extreme case, as the high image resolution supersamples the domain resolution. `DART` itself uses `numba`, a Python JIT compiler to accelerate its own calculations, without this module the code is 100 times slower. As such `cuDART` boasts a $10^3$ efficiency boost over JIT-compiled `DART`, and a $10^5$ boost over the raw Python.

`DART` features routines not present in the current `cuDART` implementation, such as the ability to "bake" rays and reuse them on different domains with identical dimensions. This allowed for a 30-60% saving in render cost for repeated imaging of different domains. Such features are not present in `cuDART` primarily because baking significantly increases the memory overhead which is more problematic for GPU codes. Additional costs associated with the absence of this feature are rapidly amortized by the efficiency of the render algorithm.

## Technical Notes

### Requirements

To run `cuDART`, the following is required:
- A CUDA-capable GPU
- The `nvcc` CUDA compiler
- Python (optional frontend, requires basic libraries such as `numpy` and `matplotlib`)

### Portability

By default, the [Makefile](https://github.com/hwhitehead/cuDART/blob/main/Makefile) uses the `-gencode` flag to avoid just-in-time ([JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation)) machine-specification code compilation, targeting Turing architecture appropriate for the GeForce RTX 2080 Ti machines used during development of this code. The users should tailor (or remove) these flags as appropriate for their runtime environment: see [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation) for the full NVIDIA GPU/Virtual Architecture feature lists.

### Inherited Libraries
To support interaction with commonly used simulation file types, this repository uses the [libnpy](https://github.com/llohse/libnpy) library to support the import of `.npy` files. In addition to the verbatim use of this libary, much of the code structure has been informed by pre-existing publically available codebases. Most notably, as with the original Pythonic [DART](https://github.com/hwhitehead/DART) repository, the underlying DDA algorithm was written with help of [this](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/grid.html) excellent guide on acceleration structures in C. [This](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) developer blog on raytracing in CUDA helped introduce me to memory management and CUDA Makefiles, though the primary Makefile structure is actually inherited from the [Athena++](https://github.com/PrincetonUniversity/athena) repository. 

### Development

While `cuDART` is functional and tested in its current form, development on this code is ongoing. Future plans include supporting additional file types and increased camera flexibility. The latest code version is v0.5, v1.0 will be delayed until this repository is made public. 

