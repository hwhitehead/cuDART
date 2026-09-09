.. _example_header:

Example Operation
##################

This page contains code snippets intended to illustrate how a user might generate their own renders using cuDART. 
Example usage is given for

| :ref:`A <example_unlabelled>`: Formatting an unlabelled simulation snapshot
| :ref:`B <example_labelled>`: Formatting a labelled simulation snapshot 
| :ref:`C <example_lookback_init>`: Preparing a master header for lookback rendering
| :ref:`D <example_no_lookback_render>`: Rendering without lookback
| :ref:`E <example_lookback_render>`: Rendering with lookback

Additional example usage can be found within the regression testing suite, available at :code:`scripts/regression.py`.
For a more detailed description of the functions and classes used here, see the :ref:`API documentation <api_header>`. 

.. _example_unlabelled:

A: Formatting an unlabelled data set
------------------------------------

Here we show how a user might format a single homogenous simulation mesh, representing a single snapshot in time.

.. code-block:: python 

    import os, sys
    pysrc = os.path.join(os.environ["CUDART_DIR"], "pysrc")
    if pysrc not in sys.path: sys.path.append(pysrc)
    from cudart import *
    import numpy as np

    # specify size of domain
    nx, ny, nz = 100, 100, 200                                      # spatial dimensions         
    data_dims = np.array([nx, ny, nz, 4])                           # data dimensions

    # load simulation data for domain
    emissivity = np.load("path/to/emissivity/data")                 # emissivity data
    vel_x = np.load("path/to/velocity_x/data")                      # velocity, in units c
    vel_y = np.load("path/to/velocity_y/data")
    vel_z = np.load("path/to/velocity_z/data")

    # package data into single array
    mesh_data = np.zeros(shape=data_dims)
    mesh_data[..., 0] = emissivity
    mesh_data[..., 1] = vel_x
    mesh_data[..., 2] = vel_y
    mesh_data[..., 3] = vel_z
    mesh_data = mesh_data.astype(np.float32)                        # enforce cast to float32
    np.save("/path/to/save/location.npy",mesh_data)

Note that no spatial labels are saved; cuDART will automatically center the data around the origin, scaling the domain so that its longest size has length unity in code units.
The code also assumes that the simulation cells are cubic. If this is not the case, the user should apply labels (see Example :ref:`B <example_labelled>`).
If the user is not running with relativistic boosting, the velocity data does not need to be included. In this case, :code:`data_dims=np.array([nx,ny,nz])`.

The :ref:`regression suite<setup_regression>` contains a routine :code:`build_unlabelled_regression_suite()` which implements the above process self-consistently using a mock dataset.
See :code:`scripts/regression.py` for the specific form.

.. _example_labelled:

B: Formatting a labelled data set
---------------------------------

Here we show how a user might package a series of subdomains for a single snapshot in time, into a labelled directory containing files for each sub-domain.

.. code-block:: python 

    import os, sys
    pysrc = os.path.join(os.environ["CUDART_DIR"], "pysrc")
    if pysrc not in sys.path: sys.path.append(pysrc)
    from cudart import *
    import numpy as np

    # construct Mesh object to hold sub-domains
    mesh = Mesh("path/to/data/dir")

    # iterate over sub-domains, labelled as MeshBlocks
    num_meshblocks = 10 
    for n in range(num_meshblocks):

        # specify size of sub-domain - need not be consistent across MeshBlocks
        nx, ny, nz = 100, 100, 200                              # spatial dimensions         
        data_dims = np.array([nx, ny, nz, 4])                   # data dimensions

        # load simulation data for sub-domain
        emissivity = np.load("path/to/emissivity/data")         # emissivity data
        vel_x = np.load("path/to/velocity_x/data")              # velocity, in units c
        vel_y = np.load("path/to/velocity_y/data")
        vel_z = np.load("path/to/velocity_z/data")

        # package data into single array
        meshblock_data = np.zeros(shape=data_dims)
        meshblock_data[..., 0] = emissivity
        meshblock_data[..., 1] = vel_x
        meshblock_data[..., 2] = vel_y
        meshblock_data[..., 3] = vel_z
        meshblock_data = meshblock_data.astype(np.float32)      # enforce cast to float32

        # add MeshBlock to Mesh, with spatial metadata
        xl = np.array([x_min, y_min, z_min])                    # vector position of lower meshblock corner
        xr = np.array([x_max, y_max, z_max])                    # vector position of upper meshblock corner
        mesh.add_meshblock(mb_data, xl, xr)

    # build header file for Mesh
    mesh.write_header()

If the user is not running with relativistic boosting, the velocity data does not need to be included. In this case, :code:`data_dims=np.array([nx,ny,nz])`.

The :ref:`regression suite<setup_regression>` contains a routine :code:`build_labelled_regression_suite()` which implements the above process self-consistently using a mock dataset.
See :code:`scripts/regression.py` for the specific form.

.. _example_lookback_init:

C. Generating a master header for lookback rendering
----------------------------------------------------

If the user wishes to render with lookback, the finite time delay routine that requires reading in multiple simulation snapshots, then an additional master header file
must be provided that provides information about the simulation snapshot cadence and length units. The master header file should be placed in the directory containing all simulation snapshots,
for more information on the required formatting of the simulation snapshots, see :ref:`Input Formats <inputs_header>`.

.. code-block:: python

    # build header data
    master_header_str = os.path.join("path/to/all/snapshots", "header.txt")
    with open(master_header_str, "w") as f:
        f.write("{0} {1} {2} {3}".format(num_snapshots, max_snapshot_size, dt_in_Myr, L_in_kpc))

The master header file is a plain text file containing a single line with four space-seperated values, which are:

- :code:`num_snapshots`: the total number of simulation snapshots within the data directory
- :code:`max_snapshot_size`: the total number of floats in the largest simulation snapshot
- :code:`dt_in_Myr`: the time interval between simulation snapshots (in Myr)
- :code:`L_in_kpc`: the conversion factor from code units to real length units (in kpc)

The :ref:`regression suite<setup_regression>` contains routines :code:`build_unlabelled_regression_suite()` and :code:`build_labelled_regression_suite()` which implements the above process self-consistently using mock datasets.
See :code:`scripts/regression.py` for the specific form.

.. _example_no_lookback_render:

D. Rendering from a single pre-formatted snapshot
-------------------------------------------------

Here we show how a user might use a single simulation snapshot in time to generate multiple rendered images using a list of cameras.

.. code-block:: python

    import os, sys
    pysrc = os.path.join(os.environ["CUDART_DIR"], "pysrc")
    if pysrc not in sys.path: sys.path.append(pysrc)
    from cudart import *
    import numpy as np

    # specify load/save string
    path_to_single_snapshot = "path/to/single/snapshot"         # single file, or directory (for unlabelled/labelled data)
    path_to_save_npy = "path/to/npy/dir"                        # directory for raw image outputs (.npy)
    path_to_save_png = "path/to/png/dir"                        # directory for figure image outputs (.png)

    # generate a template camera using properties consistent between images
    template_camera = Camera()
    template_camera.length_X = 1.0                              # image plane size in X direction
    template_camera.length_Y = 1.0                              # image plane size in Y direction
    template_camera.num_pixels_X = 2048                         # num pixels in X direction
    template_camera.num_pixels_Y = 2048                         # num pixels in Y direction 

    # generate array of cameras (or use just one)
    # in this example, vary the polar angle describing the camera position 
    num_imgs = 100
    phi = epsilon                                               # epsilon is a small value
    theta_ar = np.linspace(epsilon, np.pi - epsilon, num_imgs)  # evenly space over polar angle
    cameras = []
    for i, theta in enumerate(theta_ar):
        camera = copy.deepcopy(template_camera)
        camera.set_sph_pos(theta = theta, phi = phi, target_origin = True)
        cameras.append(camera)

    # generate Scene, the main class for rendering
    scene = Scene(load_str = path_to_single_snapshot, save_dir = path_to_save_npy, cameras = cameras)

    # use Scene to call the backend executable 
    scene.render(lookback = False)                  

    # convert raw .npy files into .png figures
    scene.plot(fig_save_dir = path_to_save_png)     

Note that both the :code:`Scene.render` and :code:`Scene.plot` routines have many other possible arguments, see the full API :ref:`here <python_api_scene>`.

The :ref:`regression suite<setup_regression>` contains routines :code:`render_single_snapshot()` and :code:`render_without_lookback()` which implements the above process self-consistently using a mock dataset.
See :code:`scripts/regression.py` for the specific form.

.. _example_lookback_render:

E. Rendering from multiple pre-formatted snapshots
--------------------------------------------------

If the user wishes to account for a finite speed of light (e.g. using the lookback routine), then multiple snapshots must be read by cuDART. 
Multiple cameras (and hence images) can still be specified.

.. code-block:: python

    import os, sys
    pysrc = os.path.join(os.environ["CUDART_DIR"], "pysrc")
    if pysrc not in sys.path: sys.path.append(pysrc)
    from cudart import *
    import numpy as np

    # specify load/save string
    path_to_all_snapshots = "path/to/all/snapshots"             # directory, containing all snapshots + master header file
    path_to_save_npy = "path/to/npy/dir"                        # directory for raw image outputs (.npy)
    path_to_save_png = "path/to/png/dir"                        # directory for figure image outputs (.png)

    # generate a template camera using properties consistent between images
    template_camera = Camera()
    template_camera.length_X = 1.0                              # image plane size in X direction
    template_camera.length_Y = 1.0                              # image plane size in Y direction
    template_camera.num_pixels_X = 2048                         # num pixels in X direction
    template_camera.num_pixels_Y = 2048                         # num pixels in Y direction 
    template_camera.theta = np.pi / 2 - epsilon                 # polar camera position
    template_camara.phi = epsilon                               # azimuthal camera position
    template_camera.r                                           # origin-camera seperaton
    template_camera.set_sph_pos                                 # init position

    # generate array of cameras (or use just one)
    # in this example, vary the time at which the observation is made
    num_imgs = 100
    cameras = []
    t_first = 0                                                 # all times in Myr
    t_last = 1 
    t_ar = np.linspace(t_first, t_last, num_imgs)
    for i, t_obs in enumerate(t_ar):
        camera = copy.deepcopy(template_camera)
        camera.t_obs = t_obs
        cameras.append(camera)

    # generate Scene, the main class for rendering
    scene = Scene(load_str = path_to_all_snapshots, save_dir = path_to_save_npy, cameras = cameras)

    # use Scene to call the backend executable 
    scene.render(lookback = True)                  

    # convert raw .npy files into .png figures
    scene.plot(fig_save_dir = path_to_save_png)     

Note that both the :code:`Scene.render` and :code:`Scene.plot` routines have many other possible arguments, see the full API :ref:`here <python_api_scene>`.
The time and length units for the system are inherited from the :code:`header.txt` file that should exist at :code:`path_to_all_snapshots/header.txt` (see Example :ref:`C <example_lookback_init>`).

The :ref:`regression suite<setup_regression>` contains a routine :code:`render_with_lookback()` and which implements the above process self-consistently using a mock dataset.
See :code:`scripts/regression.py` for the specific form.
