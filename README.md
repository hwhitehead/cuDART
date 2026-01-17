# cuDART: CUDA + DDA Accelerated Ray Tracing (v0.4)

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/rotate.gif alt=animated/>
</p>
<p align="center"">
  <em> Animation of a supernova-jet simulation snapshot, featuring 200 viewpoints each yielding a 2048<sup>2</sup> image. Raw image data for each frame generated in ~150ms, during which 4 million rays were cast through a simulation domain hosting over 400 million (750<sup>3</sup>) cells.</em>
</p>

This repository contains a lightweight set of tools for raytracing regular orthogonal meshes, intended for visualisation of line-of-sight quantities in simulated data, such as optically thin emission, surface density etc. Such visualisations, especially from arbitrary viewpoints, have the potential to be very expensive due to the large number of cells that a line-of-sight may intersect with. In `cuDART` two acceleration structures are implemented to triviliase this computation: GPU acceleration and DDA, the Digital Differential Analyzer. DDA allows for iterative low-cost propagation of rays through regular meshes, previously implemented in Python [here](https://github.com/hwhitehead/DART), but now utilising the CUDA toolkit to perform ray propagation and summation exceptionally quickly. The workhorse of the code is written in C++/CUDA, but Python scripts are provided for user ease on the frontend. 

## Usage

Broadly speaking, `cuDART` can be split into a frontend system (written in Python) and a backend (written in C++/CUDA). The user can interact with the backend directly, or use Python to make, setup, run the executable and generate plotted images. 

### Frontend
Exemplar usage of the Python frontend is included as [example.py](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py), implementing classes and routines imported from [cudart.py](https://github.com/hwhitehead/cuDART/blob/main/pysrc/cudart.py). The function `demo_scene_gen` documents the class usage: 
- The user defines strings pointing to the data to load (`npy_load_str`), a save location for the raw images (`npy_save_str`) and a save location for the matplotlib rendered images (`png_save_str`)
- The user defines a template `Camera` object, with specified dimension, position and orientation
- The user generates a list of `Camera`s with unique positions
- The user generates a `Scene` object from the target strings and `Camera` list
- The user calls the `.render()` routine from `Scene`, which generates a `.txt` file of camera positions and uses `subprocess.run` to call the executable. The `Scene` will automatically ensure the target is compiled before execution.
- The user calls the `.plot()` routine from `Scene`, which loads the raw images and plots them at `png_save_str` using `matplotlib`
- Optionally, the user can delete the raw images after plotting, along with the camera `.txt` file

### Backend
The `bin/cudart` executable accepts the following flags:
- `-i <file>`   specifies the input `.npy` file to trace
- `-c <file>`   specifies the input `.txt` file specifying the camera(s) (dimension, position and orientation)
- `-s <file>`   specifies the raw img `.npy` save location (appended numerically for multiple traces)
- `-m <value>`  species the maximum allowed VRAM usage (see "Partitioning" below)
- `-v` flags for verbose execution (prints progress to stdout)

Upon execution:
1. Data to visualise is loaded from the input `.npy` file to the host, and copied to the device
2. Memory is allocated on the device to store the image data
3. Containers for the data (`MeshBlock`) and cameras (`Camera`) are allocated and initiliased
4. The `render` kernel is called, calculating values for each pixel on the device
5. The populated image buffer is copied to the host, and saved to the output `.npy` file
6. Steps 4 and 5 are repeated for all cameras specified in the `.txt` file
7. The program frees all associated memory registers (device and host) and terminates 

## Performance
`cuDART` is bottlenecked primarily by I/O; the actual tracing of meshes and image writes are performed exceptionally quickly. The main overhead occurs at executable intialisation, due to the cost of launching a GPU context and reading the `.npy` file into host memory, usually taking O(1)s. CUDA-type operations are MUCH faster, copying the data into device memory and generating an image from this data takes only O(100)ms. For sensible image dimsions, the data transfer to device is more expensive than the trace operation itself (see profiling [here](https://github.com/hwhitehead/cuDART/blob/main/docs/profiling.txt)). As such, peak efficiency with `cuDART` is achieved when many images are taken using the same data set e.g. many lines-of-sight. Comparing a 100 image render to a single image, `cuDART` transitions from 16% of the runtime attributed to the render kernel up to 95%. 

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

### Partitioning

For sensible data-to-image resolution ratios, the vast majority of VRAM required by `cuDART` will be occupied by data to be traced. As the render kernel requires access to all of this data `cuDART` may become VRAM limited. `cuDART` will automatically partition the input data into equal length subsections of contiguous memory if the data size exceeds either the available VARM on the device, or a user specified ceiling (using `-m` at runtime). The render kernel is then called on each of these subsections in turn, transfering the next section to the device each tine. The increase in runtime scales roughly linearly with the number of partitions required.

### Inherited Libraries
To support interaction with commonly used simulation file types, this repository uses the [libnpy](https://github.com/llohse/libnpy) library to support the import of `.npy` files. In addition to the verbatim use of this libary, much of the code structure has been informed by pre-existing publically available codebases. Most notably, as with the original Pythonic [DART](https://github.com/hwhitehead/DART) repository, the underlying DDA algorithm was written with help of [this](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/grid.html) excellent guide on acceleration structures in C. [This](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) developer blog on raytracing in CUDA helped introduce me to memory management and CUDA Makefiles, though the primary Makefile structure is actually inherited from the [Athena++](https://github.com/PrincetonUniversity/athena) repository. 

### Development

While `cuDART` is functional and tested in its current form, development on this code is ongoing. Future plans include supporting additional file types and increased camera flexibility. The latest code version is v0.4, v1.0 will be delayed until this repository is made public. 

