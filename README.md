# cuDART: CUDA + DDA Accelerated Ray Tracing
This repository contains a lightweight set of tools for raytracing Cartesian meshes, designed for visualisation of line-of-sight quantities in simulated data, such as optically thin emission, surface density etc. The principle acceleration structure for this method is the Digitial Differential Analyzer (DDA) which allows for iterative low-cost propagation of rays through regular meshes. Previously implemented in Python [here](https://github.com/hwhitehead/DART), this repository utilises the CUDA toolkit to perform ray propagation and summation exceptionally quickly. The workhorse of the code is written in C++/CUDA, but Python scripts are provided for user ease on the frontend. 

## Overview
The canon of `cuDART` execution is as follows:
- The user defines camera position(s) in a `.txt` file, assisted by a Python frontend
- The user calls the `bin/cudart` executable, specifying input (`-i`) and output (`-s`) `.npy` files and tageting the camera `.txt` file (`-c`)
- `cuDART` executes:
    * Data is loaded from the input `.npy` file to the host, and copied to the device
    * Data is allocated on the device to store the image data
    * Containers for the data (`MeshBlock`) and camera (`Camera`) are initiliased
    * The `render` kernel is called, calculating pixel values synchronously on the device
    * The image buffer is copied to the host, and saved to the output `.npy` file
- The user converts the output `.npy` file to an image using Python.

## Requirements

To run `cuDART`, the following is required:
- A CUDA-capable GPU
- The `nvcc` CUDA compiler
- Python (optional frontend, requires basic libraries such as `numpy`)

## Inherited Libraries
To support interaction with commonly used simualtion file types, this repository uses the [libnpy](https://github.com/llohse/libnpy) library to support the import .npy files. In addition to this direct import, much of the code structure has been informed by pre-existing publically available codebases. Most notably, as with the original Pythonic DART repository, the underlying ray marching algorithm was written with help of [this](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/grid.html) excellent guide on acceleration structures in C. [This](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) developer blog on raytracing in CUDA helped introduce me to memory management and CUDA Makefiles, though the primary Makefile structure is actually inherited from the [Athena++](https://github.com/PrincetonUniversity/athena) repository. 

## Performance
`cuDART` is bottlenecked primarily by I/O; the actual tracing of meshes and image writes are performed exceptionally quickly. The main overhead occurs at executable intialisation, due to the cost of launching a GPU context and reading the `.npy` file into host memory. CUDA-type operations are MUCH faster, fora  1.7GB file copying the data into device memory takes ~300ms; generating a 4MP image from this data takes only 70ms (see profiling [here](https://github.com/hwhitehead/cuDART/docs/profiling.txt)). As such, peak efficiency with `cuDART` is achieved when many iamges are taken using the same data set e.g. many lines-of-sight. Comparing a 100 image render to a single image, `cuDART` transitions from 16% of the runtime dedicated to the render kernel to 95%.  