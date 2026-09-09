.. _render_header:

Generating Synthetic Observations
#################################

In this section, we break down how users can create their own synthetic observations using the cuDART codebase. 
When using the Pythonic frontend, the execution chronology can be split up into the following steps:

**Preprocessing**:

1. Convert simulation data into a cuDART legible format (see :ref:`Input Formats<inputs_header>` page).
    
    a. If using a **single** homogenous domain for each snapshot, data requires no additional labelling (see :ref:`unlabelled formatting<inputs_unlabelled>` and :ref:`example <example_unlabelled>`).
    b. If using **multiple** sub-domains for each snapshot, data must be labelled, with a header file for each snapshot containing spatial metadata (see :ref:`labelled formatting<inputs_labelled>` and :ref:`example <example_labelled>`).

2. Determine if you require the render to account for light time delay (termed "lookback", see :ref:`methodology<calculation_lookback>`). 

    a. If running **without** lookback, only a single simulation snapshot is required for the render
    b. If running **with** lookback, multiple simulation snapshots are required for the render, and a master header file is required to store unit conversion and timestep metadata (see :ref:`header formatting<example_lookback_init>`).

**Rendering**:

1. Define one or more observation orientations using the :code:`Camera` class. Each :code:`Camera` will yield its own observation (see :ref:`Camera API <python_api_camara>`).
2. Initiliase a :code:`Scene` object with a list of :code:`Camera` objects, and paths to the :code:`data_dir` / :code:`save_dir` directories for input simulation data and render outputs respecitvely (see :ref:`Scene API <python_api_scene>`). 
    
    a. If running **without** :code:`lookback` enabled, the input path should point to a single snapshot e.g. :code:`data_dir = path_to_all_snapshots/snapshotXXXXX`
    b. If running **with** :code:`lookback` enabled, the input path should point to the simulation master directory e.g. :code:`data_dir = path_to_all_snapshots`

3. Call the :code:`Scene.render()` method:

    a. When :code:`Scene.render()` is called, the frontend will invoke the render backend as a subprocess deployed to the GPU (see :ref:`C++ Execution Chronology <structure_header>` and :ref:`API <cpp_api_header>`).
    b. The render outputs will be writted as :code:`.npy` files at the :code:`save_dir` path stored with the :code:`Scene` object

4. If the user wishes to cast the :code:`.npy` images into human-readable figures, they can call the :code:`Scene.plot()` method, which uses :code:`matplotlib` to generate :code:`.png` files (see :ref:`Scene API <python_api_scene>`). 

For exemplar implementations of this chronology, see the :ref:`Example Operation <example_header>` page and the regression testing routines in :code:`scripts/regression.py`.

.. toctree::

    render/inputs.rst
    render/example.rst