Codebase API
############

.. _api_main:

The :code:`cuDART` codebase can be split into two halves, the backend which is written in C++/CUDA and runs on both the CPU and GPU, and the frontend which is written in Python and runs only on the CPU. 
While the user can interact directly with the backend, the frontend is designed to automatically generate the input parameter text files that the backend reads as run time, such as unit specifications and camera properties.

.. toctree::

    api/python_api.rst
    api/cpp_api.rst