Exemplar usage of the Python frontend is included as [examples.py](https://github.com/hwhitehead/cuDART/blob/main/scripts/examples.py), implementing classes and routines imported from [cudart.py](https://github.com/hwhitehead/cuDART/blob/main/pysrc/cudart.py). `cuDART` operates in two modes, able to read "unlabelled data" in single homogenous meshes, or "labelled data" where an arbitrary number of subregions of heterogenous resolution can be render simultaneously. For simple implementations of these two modes see `render_unlabelled_example`, `build_labelled_example` and `render_labelled_example`. 


In `build_labelled_example`:
- The user constructs a `Mesh` a master object that contains an arbitray number of `MeshBlocks` which each contain a sub-region of the simulation domain
- The user loads and contains sub-regions into `MeshBlocks`. These regions must be defined spatially in the scene, but do not need to have a single resolution. The generation of each `MeshBlock` is accompanied by the creation of a new `.npy` file hosted in a preperatory directory.
- The user invokes `write_header` which generates a `.txt` file contianing labels for the `.npy` `MeshBlock` files, including spatial positions and sizes


Both render calls implement a similar routine, the only difference is that for labelled data, the input string points to a directory containing the labelled data, whereas for unlablled data the string points directly to a `.npy` file to read and automatically label at runtime.
- The user defines strings pointing to the input and an output directory for raw `.npy` images and rasterised `.png` imagess
- The user defines an arbitrary number of `Camera` objects, each will generated a unique image of the domain
- The user generates a `Scene` object, passing to it the preperatory directory, output directory and `Camera`s.
- The user invokes `render` routine from `Scene`, which generates a `.txt` file of camera positions and uses `subprocess.run` to call the `bin/cudart` executable. The `Scene` will automatically ensure the target is compiled before execution.
- The user calls the `.plot()` routine from `Scene`, which loads the raw images and plots them at `png_save_str` using `matplotlib`
- Optionally, the user can delete the raw images after plotting, along with the camera `.txt` file
