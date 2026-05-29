Python API
##########

There are three primary frontend classes:

1. The :ref:`Scene <python_api_scene>` class 
2. The :ref:`Camera <python_api_camara>` class
3. The :ref:`Mesh <python_api_mesh>` class

These classes are used to build input files to be read by the C++ backend, to invoke the backend as a subprocess and to process the outputs. 

Throughout this documentation ``3-vector`` refers to a Numpy array of length three, representing a vector in Cartesian space with ordering :code:`(x,y,z)`.

.. _python_api_scene:

.. autoclass:: cudart.Scene

    .. automethod:: Scene.build_camera_file

    .. automethod:: Scene.render
    
    .. automethod:: Scene.plot

    .. automethod:: Scene.make_clean

    .. automethod:: Scene.make

.. _python_api_camara:

.. autoclass:: cudart.Camera

    .. automethod:: Camera.set_sph_pos

.. _python_api_mesh:

.. autoclass:: cudart.Mesh

    .. automethod:: Mesh.add_meshblock

    .. automethod:: Mesh.write_header

