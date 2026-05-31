#############################
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

    $ export CUDART_DIR=/path/to/cuDART
    $ cd $CUDART_DIR
    $ python3 configure.py --arch=GPU_ARCH --gpu=GPU_MODEL
    $ make clean
    $ make

where here the :code:`--arch` and :code:`--gpu` flags allow the user to target their own GPU architecture or model. 
This allows :code:`cuDART` to build machine-specific code and avoid just-in-time compilation on deployment. If no arguments are passed to :code:`configure.py`,
:code:`cuDART` will pre-compile GPU-agnostic code valid for any NVIDIA GPU at least as modern as Ampere. 

Regression
==========

:code:`cuDART` comes packaged with regression testing routines to construct mock data and peform a range of rendering operations. 
These routines can be found at :code:`scripts/regression.py`. These tests can be invoked from the command line as 

.. code-block:: bash

    $ cd $CUDART_DIR
    $ python3 scripts/regression.py <flags>

where the choices for flags are 

* :code:`-b` calls the :code:`build_unlabelled_regresssion_suite` function, generating a series of mock simulation snapshots in time
* :code:`-bl` calls the :code:`build_labelled_regression_suite` function, generation a series of mock simulation snapshots in time, splitting the simulation into sub-domains for use in labelled model
* :code:`--data_dir=<data_dir>` accepts a path to specify the directory in which to generate mock snapshots, and to read from for rendering
* :code:`--save_dir=<save_dir>` accepts a path to specificy the directory in which to save raw :code:`.npy` renders, and :code:`.png` figures
* :code:`-r` calls the :code:`run_nolookback_test` routine, which renders a series of images from a single snapshot, varying the camera position
* :code:`-rl` calls the :code:`run_lookback_test` routine, which renders a series of images from multiple snapshots, varying the camera time
* :code:`-v` results in additional verbose prints to the terminal

For example, if the user wishes to generate an unlabelled mock data set at :code:`$DATA_DIR`, render it using the lookback method and save the resulting
:code:`.npy` raw images and :code:`.png` figures at :code:`$SAVE_DIR`, they might run the following

.. code-block:: bash

    $ cd $CUDART_DIR
    $ python3 scripts/regression.py -v -bl -rl --data_dir=$DATA_DIR --save_dir=$SAVE_DIR 

As both the labelled and unlabelled build routines share the :code:`data_dir` write space, the two flags cannot be used together. 
Similarly, the lookback and no-lookback render routines share the :code:`save_dir` write space, so are also exclusive. 
