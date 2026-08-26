.. _structure_header:

Code Structure
##############

Given the efficiency of the render algorithm and the inescapable overheads presented by reading data from storage,
cuDART prioritises minimising the number of read/writes made from storage to memory, and the number of transfers between host RAM and device VRAM. 
Below, we show the full execution chronology for runs performed both with and without the lookback routine. We do not show the 
optional Pythonic frontend (see :ref:`documentation <python_api_header>`), except for the figure generation step at the end.

.. _strucutre_png:

.. figure:: ../../gallery/code_structure.png
    :width: 800px

The main difference between the runtime chronology between no-lookback and lookback executions is the use of a persistent image buffer when running with lookback,
and the additional iteration over each snapshot. 

Chronology without Lookback
---------------------------

Initialisation: memory is allocated on host and device for a single simulation snapshot, and a single image (not depicted).

A. The code utilises the :code:`libnpy` library to read a single snapshot of the simulation state into memory from a :code:`.npy` file. This is generally the slowest step, with duration largely dependent on the file system
B. The simulation snapshot is copied from the host memory to the device, and containerised into a :code:`Mesh` with child :code:`MeshBlocks`
C. The render kernel is executed on the simulation snapshot, for a single observation/image. The image is written to a unique memory space on the device
D. The image is copied back to the host 
E. The image is written to storage using :code:`libnpy`

    * Steps C-E are repeated for all cameras

F. Optionally, the raw :code:`.npy` images are converted into :code:`.png` figures using :code:`matplotlib`

Chronology with Lookback
------------------------

A. The code utilises the :code:`libnpy` library to read a single snapshot of the simulation state into memory from a :code:`.npy` file. This is generally the slowest step, with execution largely dependent on the file system
B. The simulation snapshot is copied from the host memory to the device, and containerised into a :code:`Mesh` with child :code:`MeshBlocks`
C. The render kernel is executed on the simulation snapshot, for a single observation/image 

    * Repeat render kernel for all observers
    * Once observations complete, load next simulation snapshot and repeat (A,B,C)
    
D. Once all simulation snapshots have been processed, copy image data from device to host
E. Write image data as :code:`.npy` files to storage
F. Optionally, convert raw images into :code:`.png` figures using the Pythonic frontend
