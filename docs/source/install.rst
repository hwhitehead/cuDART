####################
Setup and Quickstart
####################

:code:`cuDART` is written in Python/C++, and runs on the GPU. As such, to function it requires

* A CUDA-capable GPU (from Turing/Ampere onwards)
* The :code:`nvcc` compiler (`documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>`_)
* Python: optional, but highly recommended for front-end

.. _setup_general:

General Setup
=============

First, clone a local copy of the :code:`cuDART` codebase

.. code-block:: bash

    $ git clone git@github.com:hwhitehead/cuDART.git

.. _setup_cpp:

C++ Setup
---------

:code:`cuDART` is packaged with a template Makefile, you should configure a specific Makefile before building

.. code-block:: bash

    $ export CUDART_DIR=/path/to/cuDART
    $ cd $CUDART_DIR
    $ python3 configure.py --arch=GPU_ARCH --gpu=GPU_MODEL
    $ make clean
    $ make

where here the :code:`--arch` and :code:`--gpu` flags can be used to target specific GPU architecture or models. 
These flags allows :code:`cuDART` to build machine-specific code and avoid just-in-time compilation on deployment. If no arguments are passed to :code:`configure.py`,
:code:`cuDART` will pre-compile GPU-agnostic code valid for any NVIDIA GPU at least as modern as Ampere (`list of GPU architectures <https://en.wikipedia.org/wiki/Category:Nvidia_microarchitectures>`_).

.. _setup_python:

Python Setup
------------

If you wish to use the Python front end (as is recommended), the required modules can be found at :code:`pysrc/requirements.txt`.
You can also build a virutal environment specifically for :code:`cuDART` usage and auto install these modules

.. code-block:: bash

    $ cd $CUDART_DIR/pysrc
    $ python3 -m venv cudart_venv
    $ source cudart_venv/bin/activate
    $ (cudart_venv) python3 -m pip install -r requirements.txt

Quickstart
==========

:code:`cuDART` comes packaged with regression testing routines to construct mock data and peform a range of rendering operations. 
These routines can be found at :code:`scripts/regression.py`. These tests can be invoked from the command line as 

.. code-block:: bash

    $ cd $CUDART_DIR
    $ python3 scripts/regression.py <flags>

where the choices for flags are 

* :code:`-build` generates a series of unlabelled mock simulation snapshots in time 
* :code:`-build_labelled` generates a series of mock simulation snapshots in time, splitting the simulation into sub-domains for use in labelled model
* :code:`--build_mode=<sphere, sphere_rest, jet>` specifies the simulation type (sphere in lab frame, sphere in rest frame or jet)
* :code:`-render` calls the render routine without lookacbk, rendering a series of images from a single snapshot, varying the camera position
* :code:`-render_lookback` calls the render routine with lookback, rendering a series of images from multiple snapshots, varying the camera time
* :code:`-render_comp` calls the render routine both with and without lookback, generating a :code:`.png` figure to compare the result
* :code:`--data_dir=<data_dir>` accepts a path to specify the directory in which to generate mock snapshots, and to read from for rendering
* :code:`--save_dir=<save_dir>` accepts a path to specificy the directory in which to save raw :code:`.npy` renders, and :code:`.png` figures
* :code:`-profile` generates a profiling report for a previously concluded render execution
* :code:`-verbose` prints additional progress reports to the terminal

For example, if the user wishes to generate an unlabelled mock data set at :code:`$DATA_DIR`, render it using the lookback method and save the resulting
:code:`.npy` raw images and :code:`.png` figures at :code:`$SAVE_DIR`, they might run the following

.. code-block:: bash

    $ cd $CUDART_DIR
    $ python3 scripts/regression.py -verbose -build -render_lookback --data_dir=$DATA_DIR --save_dir=$SAVE_DIR 

As the build routines share the :code:`data_dir` write space, the two build flags cannot be used together. 
Similarly, the render routines share the :code:`save_dir` write space, so are also exclusive. 

Building Documentation
======================

Users can build their own version of the documentation by creating a Python virtual environment and building the proper :code:`html` files using Sphinx. 

.. code-block:: bash

    $ cd $CUDART_DIR/docs
    $ python3 -m venv sphinx_venv
    $ source sphinx_venv/bin/activate
    $ (sphinx_venv) python3 -m pip install -r requirements.txt

The documentation (built at :code:`docs/build/html`) can then be constructed and opened from the terminal using 

.. code-block:: bash
    
    cd $CUDART_DIR/docs
    make html
    chromium build/html/index.html

Replace :code:`chromium` with your own browser. The documentation can also be opened using :code:`CTRL + O`/:code:`CMD + O` in browser and selecting the :code:`.html` file.