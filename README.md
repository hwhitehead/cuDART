# cuDART: CUDA + DDA Accelerated Ray Tracing (v0.4)

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/rotate.gif alt=animated/>
</p>
<p align="center"">
  <em> Animation of a supernova-jet simulation snapshot, featuring 200 viewpoints each yielding a 2048^2 (4M) pixel image. Total runtime 50s, during which over 800M rays were cast through a 750^3 (400M) cell domain.</em>
</p>

This repository contains a lightweight set of tools for raytracing Cartesian meshes, designed for visualisation of line-of-sight quantities in simulated data, such as optically thin emission, surface density etc. The principle acceleration structure for this method is the Digitial Differential Analyzer (DDA) which allows for iterative low-cost propagation of rays through regular meshes. Previously implemented in Python [here](https://github.com/hwhitehead/DART), this repository utilises the CUDA toolkit to perform ray propagation and summation exceptionally quickly. The workhorse of the code is written in C++/CUDA, but Python scripts are provided for user ease on the frontend. 

## Usage

Broadly speaking, `cuDART` can be split into a frontend system (written in Python) and a backend (written in C++/CUDA). The user can interact with the backend directly, or use Python to make, setup, run the executable and generate plotted images. 

### Frontend
Exemplar usage of the Python frontend is included as [example.py](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py), implementing classes and routines imported from [cudart.py](https://github.com/hwhitehead/cuDART/blob/main/pysrc/cudart.py). The function `demo_scene_gen` documents the class usage: 
- The user defines strings pointing to the data to load (`npy_load_str`), a save location for the raw images (`npy_save_str`) and a save location for the matplotlib rendered images (`png_save_str`)
- The user defines a template `Camera` object, with specified dimension, position and orientation
- The user generates a list of `Camera`s with unique positions
- The user generates a `Scene` object from the target strings and `Camera` list
- The user calls the `.render()` routine from `Scene`, which generates a `.txt` file of camera positions and uses `subprocess.run` to call the executable
- The user calls the `.plot()` routine from `Scene`, which loads the raw images and plots them using `matplotlib`
- Optionally, the user can delete the raw images after plotting, along with the camera `.txt` file

### Backend
The `bin/cudart` executable accepts the following flags:
- `-i` specifies the input `.npy` file to trace
- `-c` specifies the input `.txt` file specifying the camera(s)
- `-s` specifies the raw img `.npy` save location (appended numerically for multiple traces)
- `-v` flags for verbose execution (prints progress to stdout)

Upon execution:
1. Data is loaded from the input `.npy` file to the host, and copied to the device
2. Data is allocated on the device to store the image data
3. Containers for the data (`MeshBlock`) and cameras (`Camera`) are initiliased
4. The `render` kernel is called, calculating values for each pixel using the device
5. The image buffer is copied to the host, and saved to the output `.npy` file
6. Steps 4 and 5 are repeated for all cameras specified in the `.txt` file

## Requirements

To run `cuDART`, the following is required:
- A CUDA-capable GPU
- The `nvcc` CUDA compiler
- Python (optional frontend, requires basic libraries such as `numpy` and `matplotlib`)

## Performance
`cuDART` is bottlenecked primarily by I/O; the actual tracing of meshes and image writes are performed exceptionally quickly. The main overhead occurs at executable intialisation, due to the cost of launching a GPU context and reading the `.npy` file into host memory. CUDA-type operations are MUCH faster, for a 1.7GB file copying the data into device memory takes ~300ms and generating a 4MP image from this data takes only 70ms (see profiling [here](https://github.com/hwhitehead/cuDART/blob/main/docs/profiling.txt)). As such, peak efficiency with `cuDART` is achieved when many images are taken using the same data set e.g. many lines-of-sight. Comparing a 100 image render to a single image, `cuDART` transitions from 16% of the runtime dedicated to the render kernel to 95%.  

## Portability

By default, the [Makefile](https://github.com/hwhitehead/cuDART/blob/main/Makefile) uses the `-gencode` flag to avoid JIT machine-specification code compilation, targeting Turing architecture appropriate for the 2060 Ti machines used to develop this code. The users should tailor (or remove) these flags as appropriate for their runtime environment; see [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation) for the full NVIDIA GPU/Virtual Architecture feature lists.

## Inherited Libraries
To support interaction with commonly used simualtion file types, this repository uses the [libnpy](https://github.com/llohse/libnpy) library to support the import .npy files. In addition to this direct import, much of the code structure has been informed by pre-existing publically available codebases. Most notably, as with the original Pythonic DART repository, the underlying ray marching algorithm was written with help of [this](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/grid.html) excellent guide on acceleration structures in C. [This](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) developer blog on raytracing in CUDA helped introduce me to memory management and CUDA Makefiles, though the primary Makefile structure is actually inherited from the [Athena++](https://github.com/PrincetonUniversity/athena) repository. 