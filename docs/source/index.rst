.. cuDART documentation master file, created by
   sphinx-quickstart on Thu May 28 13:59:00 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cuDART documentation
====================

.. figure:: ../../gallery/magnetised_jets.png
    :width: 800px

:code:`cuDART` is a relativistic ray-tracing code designed to generate synthentic observations via line-of-sight summation of optically thin emission. 
Generating such visualisations, especially from arbitrary viewpoints, can prove very expensive due to the large number of cells that a line-of-sight may intersect with. 
In :code:`cuDART` two acceleration structures are implemented to triviliase this computation: GPU acceleration and DDA, the Digital Differential Analyzer. 
DDA allows for iterative low-cost propagation of rays through regular meshes, :code:`cuDART` uses the CUDA toolkit to perform ray propagation and summation exceptionally quickly. 
:code:`cuDART` allows the computations including relativistic boosting, and supports a finite speed of light, allowing for geometric effects such as superluminal motion to be recovered.

The workhorse of the code is written in C++/CUDA, but Python scripts are provided for user ease on the frontend. The code is available on `GitHub <https://github.com/hwhitehead/cuDART>`_. 
Please report any issues and suggestions for improvement here. Development for :code:`cuDART` is use-case driven; if there are projects that you think may benefit from this code, but requires additional features
please get in touch with the authors.

---------------------------------------
Publications
---------------------------------------

Earlier iterations of this codebase have been use to generate synethic observations for the following publications:

* Gasealahwe et al. (2025): `A relativistic jet from a neutron star breaking out of its natal supernova remnant <https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.4011G/abstract>`_,
* Elley et al. (2026): `The impact of flickering variability and magnetisation on the dynamics, stability and morphology of radio-loud AGN jets <https://ui.adsabs.harvard.edu/abs/2026arXiv260513469E/abstract>`_.

---------------------------------------
Development
---------------------------------------

Authors and contributors to the :code:`cuDART` code and their institutions are:

Henry Whitehead
    DPhil candidate, Department of Physics, Astrophysics, University of Oxford, Denys Wilkinson Building, Keble Road, Oxford, OX1 3RH, UK

.. toctree::
    :titlesonly:
    :glob:
    :hidden:
    :caption: Documentation

    install
    *

