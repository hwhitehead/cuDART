##################
Example Operation
##################

This page contains code snippets intended to illustrate how a user might generate their own renders using :code:`cuDART`. 
Example usage is given for

A. Formatting an unlabelled simulation snapshot (:ref:`Example <example_section_A>`)
B. Formatting a labelled simulation snapshot (:ref:`Example <example_section_B>`)
C. Rendering a single pre-formatted simulation snapshot (:ref:`Example <example_section_C>`)
D. Rendering multiple pre-formatted simulation snapshots with the lookback routine (:ref:`Example <example_section_D>`)

Additional example usage can be found within the regression testing suite, available at :code:`scripts/regression.py`.
For a more detailed description of the functions and classes used here, see the :ref:`API documentation <api_main>`. 

.. _example_section_A:

A: Formatting an unlabelled data set
------------------------------------

Here we show how a user might format a single homogenous simulation mesh, representing a single snapshot in time.

.. code-block:: python 

    pysrc = os.path.join("path/to/pysrc")
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

Note that no spatial labels are saved; :code:`cuDART` will automatically center the data around the origin, scaling the domain so that its longest size has length unity in code units.
The code also assumes that the simulation cells are cubic. If this is not the case, the user should apply labels (see Example B).
If the user is not running with relativisitc boosting, the velocity data does not need to be included. In this case, :code:`data_dims=np.array([nx,ny,nz])`.

.. _example_section_B:

B: Formatting a labelled data set
---------------------------------

Here we show how a user might package a series of subdomains for a single snapshot in time, into a labelled directory containing files for each sub-domain.

.. code-block:: python 

    pysrc = os.path.join("path/to/pysrc")
    from cudart import *
    import numpy as np

    # construct Mesh object to hold sub-domains
    mesh = Mesh("path/to/data/dir")

    # iterate over sub-domains, labelled MeshBlock
    num_meshblocks = 10 
    for n in range(num_meshblocks):

        # specify size of sub-domain
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

        # add MeshBlock to Mesh, with spatial information
        xl = np.array([0.0, 0.0, 0.0])                          # vector position of lower meshblock corner
        xr = np.array([1.0, 1.0, 1.0])                          # vector position of upper meshblock corner
        mesh.add_meshblock(mb_data, xl, xr)

    # build header file for Mesh
    mesh.write_header()

If the user is not running with relativisitc boosting, the velocity data does not need to be included. In this case, :code:`data_dims=np.array([nx,ny,nz])`.

.. _example_section_C:

C. Rendering from a single pre-formatted snapshot
-------------------------------------------------

Here we show how a user might use a single simulation snapshot in time to generate multiple rendered images using a list of cameras.

.. code-block:: python

    pysrc = os.path.join("path/to/pysrc")
    from cudart import *
    import numpy as np

    # specify load/save string
    path_to_snapshot = "path/to/single/snapshot"                # single file, or directory (unlabelled/labelled)
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
    scene = Scene(load_str = path_to_snapshot, save_dir = path_to_save_npy, cameras = cameras)

    # use Scene to call the .cpp executable 
    scene.render(lookback = False)                  

    # convert raw .npy files into .png figures
    scene.plot(fig_save_dir = path_to_save_png)     

Note that both the :code:`Scene.render` and :code:`Scene.plot` routines have many other possible arguments, see full documentation.

.. _example_section_D:

D. Rendering from multiple pre-formatted snapshots
--------------------------------------------------

If the user wishes to account for a finite speed of light, then multiple snapshots must be read by :code:`cuDART`. 
Multiple cameras (and hence images) can still be specified.

.. code-block:: python

    pysrc = os.path.join("path/to/pysrc")
    from cudart import *
    import numpy as np

    # specify load/save string
    path_to_snapshot = "path/to/all/snapshots"                  # directory, containing all snapshots + header file
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
    scene = Scene(load_str = path_to_snapshot, save_dir = path_to_save_npy, cameras = cameras)

    # use Scene to call the .cpp executable 
    scene.render(lookback = True)                  

    # convert raw .npy files into .png figures
    scene.plot(fig_save_dir = path_to_save_png)     

Note that both the :code:`Scene.render` and :code:`Scene.plot` routines have many other possible arguments, see full documentation.
The units for the origin-camera distance are inherited from the :code:`header.txt` file that should exist at :code:`path/to/all/snapshots/header.txt`.
This header file should feature a single line of text in the form :code:`num_snapshots snapshot_size dt L_domain` where :code:`num_snapshots` is the total number 
of snapshots within the data dir, :code:`snapshot_size` is the maximum snapshot size (by cells), :code:`dt` is the (fixed) time between snapshots in Myr and 
:code:`L_domain` is the code unit for length (in kpc). 