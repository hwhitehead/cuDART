Installation and Requirements
#############################

:code:`cuDART` is written in C++, and runs on the GPU. As such, to function it requires

* A CUDA-capable GPU (from Ampere onwards)
* The :code:`nvcc` compiler
* Python (optional, but highly recommended for front-end), with the following modules
    * common modules such as :code:`numpy`, :code:`matplotlib`
    * if using data from :code:`PLUTO`, the :code:`pyPLUTO` module (available `here <https://github.com/GiMattia/PyPLUTO>`_)
    * if using data in the HDF5 format, the :code:`h5py` module

Setup
=====

First, clone a local copy of the :code:`cuDART` codebase

.. code-block:: bash

    $ git clone git@github.com:hwhitehead/cuDART.git

:code:`cuDART` is packaged with a template makefile, the user should configure before building

.. code-block:: bash

    $ expot CUDART_DIR=/path/to/cuDART
    $ cd $CUDART_DIR
    $ python3 configure.py --arch=GPU_ARCH --gpu=GPU_MODEL
    $ make clean
    $ make

where here the :code:`--arch` and :code:`--gpu` flags allow the user to target their own GPU architecture or model. 
This allows :code:`cuDART` to build machine-specific code and avoid just-in-time compilation on deployment. If no arguments are passed to :code:`configure.py`
:code:`cuDART` will pre-compile GPU-agnostic code valid for any NVIDIA GPU at least as modern as Ampere. 

Running :code:`cuDART`
==============