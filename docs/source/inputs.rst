#############
Input Formats
#############

:code:`cuDART` reads simulation data in the form of :code:`.npy` files which MUST have :code:`dtype=np.float32`.
The formatting of these files depends on the intended operation mode. Each :code:`.npy` file should have the shape

* :code:`(nx,ny,nz)`: if running without relativisitc boosting
* :code:`(nx,ny,nz,4)`: if running with relativistic boosting

where here :code:`nx`, :code:`ny` and :code:`nz` are the number of cells in each cardinal direction. If running without relativistic boosting, the array should
contian the data that will be summed along the line-of-sight such as density, emissivity. If running with relativistic boosting, the fourth index spans the quantity
to be traced (the rest-frame emissivity) and the velocity in each of the cardinal directions.

The input data should have column-major ordering (C-style). It is worth noting that Python prefers to edit stride meta-data rather than explicitly rearrange its memory,
which can result in initially column-major ordered arrays losing this property. It is left to the user to ensure that their input data has true C-style ordering, this
can be forced in Python using :code:`input_data = np.array(input_data, order="C")`.

The input data may be unlabelled, or labeled:

* :ref:`Unlabelled <inputs_unlabelled>` mode: each simulation snapshot is stored a single :code:`.npy` file
* :ref:`Labelled <inputs_labelled>` mode: each simulation snapshot is stored within a directory, containing multiple :code:`.npy` files for each sub-domain, and a header file :code:`header.txt` which contains additional information

.. _inputs_unlabelled:

Unlabelled Mode
---------------

Unlabelled mode is designed for rendering simulation domains that have a single homogenous resolution. In this case,
the input can be a single :code:`.npy` file; the code will automatically assume that the simulation cells are cubic and that the longest domain side
has a length of unity. If the domain does not feature cubic cells, or contains multiple sub-domains in different resolutions,
then labelled mode should be used. The data directory containing unlabelled simulation data might take the form

| data_directory
| ├── header.txt
| ├── snapshot00000.npy
| ├── snapshot00001.npy
| ├── snapshot00002.npy
| ├── ...

Note the single header file, which should contain information about time and length units. This file is only necessary if rendering with lookback.

.. _inputs_labelled:

Labelled Mode
-------------

Labelled mode is designed to render simulation domains composed of multiple sub-domains which may not have the same resolution. 
In this case, the data directory should contain a :code:`.npy` file for each subdomian, and an additional header file to label the spatial position/extents 
of each sub-domain (as well as meta-data concerning the number of cells in each domain). The directory containing labelled simulation data might take the form

| data_directory
| ├── header.txt
| ├── snapshot00000
| │   ├── header.txt
| │   ├── meshblock00000.npy
| │   ├── meshblock00001.npy
| │   ├── ...
| ├── snapshot00001
| │   ├── header.txt
| │   ├── meshblock00000.npy
| │   ├── meshblock00001.npy
| │   ├── ...
| ├── ...

Header files can be automatically formatted using the :code:`Mesh` class (see `Example Usage <example>`). 
Note the additional header file at the top of the tree, this file contains information about time and length units. This file is only necessary if rendering with lookback.

Converting Simulation Data
--------------------------
While useful for manipulation in Python, :code:`.npy` files are generally not the standard outputs from hydrodynamic simulation codes. To support conversion 
between standard simulation file formats and the :code:`.npy` files required by :code:`cuDART`, additional functions are included in :code:`pysrc`.
Currently conversion support is available for the :code:`.vtk` files output from :code:`PLUTO`, and the :code:`.hdf5` files output from :code:`Athena++`.
We recommend the user develops their own routines for conversion between simulation output and :code:`.npy` file as approriate for their own use case.