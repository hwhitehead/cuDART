.. _cpp_api_header:

C++ API
#######

While it is perfectly possible (and recommended) to interact with cuDART entirely through its Pythonic frontend,
the user can also interface directly with the C++ executable through the command line. The C++ API can be found entirely within the :code:`src` directory.
Here we describe the functionality of the main classes used within cuDART.

1. The :ref:`vec3 <cpp_api_vec3>` class
2. The :ref:`Ray <cpp_api_ray>` class
3. The :ref:`Camera <cpp_api_camera>` class
4. The :ref:`MeshBlock <cpp_api_meshblock>` class
5. The :ref:`Mesh <cpp_api_mesh>` class 

Outside of this class structure, the main code routines can be split into 

1. :ref:`Kernels <cpp_api_kernels>`
2. :ref:`Tools <cpp_api_tools>`

Included within the C++ API is :code:`npy.hpp`, which provides functionality for reading and writing :code:`.npy` files with C++. 
This library is imported near verbatim from the `libnpy <https://github.com/llohse/libnpy>`_ repository created by Leon Merten Lohse (`github <https://github.com/llohse>`_) 
under MIT `license <https://github.com/llohse/libnpy/blob/master/LICENSE>`_. Minor additions have been made to bypass the need to load via a :code:`std::vector` object. 
We do not include documentation for this library here, please refer to the original libnpy repository.


For brevity, this page gives only simple descriptions of classes, member and non-member functions. For a full description of class attributes and function parameters, 
refer to the source code within the :code:`src` directory. Life is too short to use Doxygen, Breathe, Sphinx and ReadTheDocs to auto-document relatively 
straightforward C++ code...

Classes
-------

.. _cpp_api_vec3:

.. cpp:class:: vec3

    The :code:`vec3` class is used to support basic vector arithmetic in C++, such as normalisation, dot and cross products and rotations about 
    another vector. 

.. _cpp_api_ray:

.. cpp:class:: Ray

    The :code:`Ray` class describes the line-of-sight for a pixel, defined by an origin and a normal vector

    .. cpp:member:: vec3 march(float s)

        Return the point given by traversing a distance :code:`s` from the origin along the normal vector

.. _cpp_api_camera:

.. cpp:class:: Camera

    The :code:`Camera` class is responsible for loading in camera properties from a text file and generating ray origins for each pixels in an image

    .. cpp:member:: void build_camera()

        Generates an image plane using internal positional and orientional arguments 

    .. cpp:member:: vec3 calc_pixel_origin(const int i, const int j)

        Returns the ray origin for a pixel given its 2D indices in the image plane

.. _cpp_api_meshblock:

.. cpp:class:: MeshBlock

    The :code:`MeshBlock` class is the keystone of the C++ API, responsible for containing a simulation sub-domain of fixed resolution and providing 
    methods to calculate line-of-sight summations through this sub-domain using the DDA alogorithm.

    .. cpp:member:: bool calc_mb_intercept(const Ray *r, float &s_entry, float &s_exit)

        Determine if intersection occurs between a :code:`Ray` and the :code:`MeshBlock`; if so then store the entry and exit locations as 
        parameterised by :code:`s_entry` and :code:`s_exit`
    
    .. cpp:member:: float calc_trace(const Ray &r, TraceArgs trace_args)

        Calculate a summation through the :code:`MeshBlock` sub-domain, on a path given by a :code:`Ray`'s line of sight. Apply runtime flags as stored within the :code:`TraceArgs` struct.

.. _cpp_api_mesh:

.. cpp:class:: Mesh 

    Akin to the Pythonic :ref:`Mesh <python_api_mesh>` class, in the C++ code the :code:`Mesh` class is a container for all sub-domains, wrapping all
    :code:`MeshBlocks` within a single snapshot of the simulation. The :code:`Mesh` able to accept lines-of-sight generated within 
    the :code:`Camera` class and iterate over all :code:`MeshBlocks` to perform the path summation. 
    Unlike the Pythonic Mesh, in C++ the :code:`Mesh` class is used regardless of running in labelled or unlabelled mode. In the unlabelled case, the :code:`Mesh`
    will contain only a single :code:`MeshBlock`

    .. cpp:function:: float calc_trace(const Ray &r, TraceArgs trace_args)

        Calculate the total summation through the full domain for a given :code:`Ray`'s line-of-sight, summing over contributions from each :code:`MeshBlock`
    
.. _cpp_api_kernels:

Kernels
-------

Kernels are functions called from the host, but deployed on the device. All kernels have :code:`__global__` specifier.

.. cpp:function:: render_from_mesh(Camera camera, float *img, Mesh **mesh, TraceArgs trace_args)

    Principle render kernel, calculates pixel values on a thread-by-thread basis

.. cpp:function:: init_mesh(Mesh **mesh, MeshBlock **mb_list, int num_meshblocks)

    Allocated and initialises a :code:`Mesh` container on the device

.. cpp:function:: free_mesh(Mesh **mesh, int num_meshblocks) 

    Deletes :code:`Mesh` container from device, frees each :code:`MeshBlock`

.. cpp:function:: init_meshblock(MeshBlockInfo mb_info, MeshBlock **mb_list, float *data) 

    Builds :code:`MeshBlock` on device, stash within the :code:`MeshBlock` list 

.. cpp:function:: wipe_img(Camera camera, float *img)

    Sets all values in the image buffer to zero as prep for next render

.. _cpp_api_tools:

Tools
-----

Here we list some of the more important non-method, non-kernel functions

.. cpp:function:: std::vector<Camera> load_cameras(char *camera_char, bool verbose) 

    Builds an array of :code:`Camera` objects loaded line-by-line from a text file

.. cpp:function:: std::vector<MeshBlockInfo> load_unlabelled_meshblock(std::string input_str, float* &h_all_data, size_t &h_bytes, bool relativistic, bool verbose, bool host_malloc)

    Load unlabelled :code:`MeshBlock` from storage, allocate and load into host. Return :code:`MeshBlock` metadata.

.. cpp:function:: std::vector<MeshBlockInfo> load_labelled_meshblock(std::string input_str, float* &h_all_data, size_t &h_bytes, bool relativistic, bool verbose, bool host_alloc)

    Load labelled :code:`MeshBlock` from storage, read text file metadata, allocate and load into host. Return :code:`MeshBlock` metadata.

.. cpp:function:: void build_containers(std::vector<MeshBlockInfo> all_mb_info, float* &d_data, MeshBlock ** &mb_list, Mesh ** &mesh, bool verbose) 

    Allocate memory space on the device for the :code:`MeshBlock` list, initialise :code:`MeshBlock` objects on device
    Allocate and initialise :code:`Mesh` on device 