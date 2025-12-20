# cuDART
This repository contains routines that allow for the DART ray-tracing simualtion visualisation code (originally implemented purely in Python with Numba acceleration [here](https://github.com/hwhitehead/DART)) to be accelerated with the CUDA API.

## Included Libraries
To support interaction with commonly used simualtion file types, this repository uses external libraries to import .npy files.
- linpy (available [here](https://github.com/llohse/libnpy))
