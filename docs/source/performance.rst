.. _performance_header:

Performance and Profiling
#########################

cuDART comes packaged with a lightweight wrapper to the :code:`nsys` (NVIDIA Nsight System) profiling toolkit.
To profile the execution of the C++ backend, set :code:`save_profile = True` when calling :code:`Scene.render`, this will prepend
the subprocess call to the C++ executable with a call to :code:`nsys profile`, generating a series of logfiles in the same output
directory as the rendered data (set using :code:`save_dir` on :code:`Scene` init). Results from these logfiles can then be printed to
the command line using the :code:`Profiler` class (see Pythonic API :ref:`here <python_api_header>`). Caution is warranted when interpreting 
the timings produced, as much of the GPU/CPU execution is asynchronous and so the total task duration will be in excess of the true wallclock.

The main contributors to runtime during the C++ execution of cuDART are

1. File read from storage to host (:code:`.npy` to RAM) 
2. Data transfer from host to device (RAM to VRAM)
3. Rendering the data on the device 
4. Writing the data to storage

Depending on the quality of the local storage, the cost of reading files can represent a significant fraction of the total runtime, making the user's
I/O environment source of performance discrepancy between systems. Given this uncertainty, on this page we report a series of performance metrics yielded 
renders produced using various GPUs on the Institute of Science and Techonology Austria's Scientific Computing Cluster. Each test was run using a single CPU/GPU, 
using mock simulation data with :math:`M` snapshots each containing :math:`D^3` cells, and :math:`N` renders composed of :math:`P^2` pixels. 

Brute Force Summation
---------------------

In the worst case scenario, where the render routine has no prior knowledge of the intersections between cells and pixel rays, a computation of the type used 
for performance testing here scales as :math:`O(D^3P^2NM)`. For a relatively modest input dimension of :math:`\{D,M,P.N\}=\{512,100,100,512\}` this amounts to a 
staggering :math:`O(10^{17})` potentially cell-ray intersections. In practice, cuDART massively reduces complexity by

1. Using DDA to iteratively progresss along each ray, considering only cells with spatial intersection. This reduces the intersection scaling from :math:`O(D^3)\rightarrow O(D)`.
2. Using the GPU to compute pixel values simultaneosuly, resulting in very weak scaling with :math:`P` when there are sufficient threads to cover all pixels
3. Using flexible laoding and traversal routines to skip calculations that cannot contribute to the path summation.

.. code-block:: bash

    [5/9] Executing 'osrt_sum' stats report

     Time (%)  Total Time (ns)  Num Calls     Avg (ns)         Med (ns)       Min (ns)     Max (ns)       StdDev (ns)             Name         
    --------  ---------------  ---------  ---------------  ---------------  ----------  -------------  ---------------  ----------------------
        55.5   95,895,779,402        985     97,356,121.2    100,173,532.0       1,639    343,590,736     18,279,051.9  poll                  
        26.4   45,575,994,575         92    495,391,245.4    500,116,603.0  66,370,191    500,148,166     45,220,066.5  pthread_cond_timedwait
        8.7   14,941,729,633        245     60,986,651.6        503,971.0       1,032    158,568,638     73,176,051.3  fclose                
        6.5   11,233,420,583        229     49,054,238.4          9,580.0       1,049    240,424,392     56,895,150.7  read                  
        1.3    2,206,125,504          2  1,103,062,752.0  1,103,062,752.0     164,077  2,205,961,427  1,559,734,264.1  pthread_cond_wait     
        1.0    1,697,223,170        100     16,972,231.7     17,459,862.0   7,806,869     22,508,151      1,873,314.0  writev                
        0.4      629,191,184        203      3,099,464.0        585,227.0     362,458     10,732,908      2,760,675.5  fopen64               
        0.2      403,721,065      1,682        240,024.4         38,301.0       1,278     22,268,038      1,080,346.4  ioctl                 
        0.0       31,135,402          3     10,378,467.3        265,814.0     250,071     30,619,517     17,529,265.0  pthread_join          
        0.0       13,264,971     10,016          1,324.4          1,314.0       1,004         10,691            184.1  fwrite                
        0.0        4,753,175         62         76,664.1          9,457.5       1,986      2,718,487        365,203.6  mmap64                
        0.0        2,629,082         23        114,307.9         91,631.0      17,069        688,046        152,696.7  sem_timedwait         
        0.0        2,293,004         54         42,463.0          2,703.5       1,111        837,620        161,370.8  fopen                 
        0.0        2,059,925        302          6,820.9          4,890.0       1,666         17,158          5,129.6  stat                  
        0.0        1,678,214         24         69,925.6          5,771.5       1,088        391,426        121,999.2  mmap                  
        0.0          522,341         15         34,822.7          3,651.0       2,262        437,260        111,451.5  munmap                
        0.0          291,296         74          3,936.4          3,417.0       1,631         14,248          2,061.6  open64                
        0.0          159,461          4         39,865.3         40,602.5      33,004         45,252          6,081.1  pthread_create        
        0.0          120,456          7         17,208.0         14,217.0       3,601         34,631         12,564.1  fread                 
        0.0           68,983         24          2,874.3          2,237.0       1,031         10,931          2,369.2  write                 
        0.0           58,746          2         29,373.0         29,373.0      26,101         32,645          4,627.3  pthread_mutex_lock    
        0.0           37,635          1         37,635.0         37,635.0      37,635         37,635              0.0  fgets                 
        0.0           25,877          6          4,312.8          4,757.0       1,388          6,310          1,923.5  open                  
        0.0           18,322          4          4,580.5          4,479.0       2,474          6,890          2,374.9  pipe2                 
        0.0           16,952          2          8,476.0          8,476.0       3,416         13,536          7,155.9  fcntl                 
        0.0           12,295          2          6,147.5          6,147.5       4,946          7,349          1,699.2  socket                
        0.0            8,109          1          8,109.0          8,109.0       8,109          8,109              0.0  connect               
        0.0            7,112          1          7,112.0          7,112.0       7,112          7,112              0.0  pthread_kill          
        0.0            5,448          2          2,724.0          2,724.0       2,578          2,870            206.5  pthread_cond_signal   
        0.0            4,326          1          4,326.0          4,326.0       4,326          4,326              0.0  pthread_cond_broadcast
        0.0            3,568          2          1,784.0          1,784.0       1,576          1,992            294.2  putc                  
        0.0            1,309          1          1,309.0          1,309.0       1,309          1,309              0.0  bind                  

    [6/9] Executing 'cuda_api_sum' stats report

    Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
    --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
        58.2   11,144,635,117        101  110,342,922.0  100,423,682.0   99,259,185  628,811,320  53,727,618.9  cudaMemcpy            
        39.6    7,581,715,798     20,300      373,483.5       42,247.0        4,860    1,770,121     340,740.3  cudaDeviceSynchronize 
        0.9      168,336,216          1  168,336,216.0  168,336,216.0  168,336,216  168,336,216           0.0  cudaMemGetInfo        
        0.7      135,058,079          1  135,058,079.0  135,058,079.0  135,058,079  135,058,079           0.0  cudaDeviceReset       
        0.4       79,756,286     20,300        3,928.9        3,452.0        3,126    4,319,755      30,344.4  cudaLaunchKernel      
        0.1       24,116,592        202      119,389.1      160,131.5        4,909      307,162     113,731.2  cudaMalloc            
        0.1       16,769,101        202       83,015.4       99,817.0        5,321    4,040,461     295,340.4  cudaFree              
        0.0           15,992          1       15,992.0       15,992.0       15,992       15,992           0.0  cuCtxSynchronize      
        0.0              767          1          767.0          767.0          767          767           0.0  cuModuleGetLoadingMode

    [7/9] Executing 'cuda_gpu_kern_sum' stats report

    Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                          Name                         
    --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------------------------------------------------
        94.8    7,147,447,424     10,000  714,744.7  720,574.5   624,158   984,030     54,893.3  render_from_mesh(Camera, float *, Mesh **, TraceArgs)
        5.2      391,459,852     10,000   39,146.0   38,976.0    38,048    52,896      1,485.1  wipe_img(Camera, float *)                            
        0.0        1,344,733        100   13,447.3   13,392.0    12,864    19,520        632.8  free_mesh(Mesh **, int)                              
        0.0          938,109        100    9,381.1    9,312.0     9,215    15,072        586.1  init_meshblock(MeshBlockInfo, MeshBlock **, float *) 
        0.0          754,110        100    7,541.1    7,488.0     7,392    11,680        420.3  init_mesh(Mesh **, MeshBlock **, int)                

    [8/9] Executing 'cuda_gpu_mem_time_sum' stats report

    Time (%)  Total Time (ns)  Count    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Operation          
    --------  ---------------  -----  -------------  -------------  -----------  -----------  ------------  ----------------------------
        94.4   10,501,701,437    100  105,017,014.4  100,269,273.5   99,147,370  147,464,338  13,151,318.0  [CUDA memcpy Host-to-Device]
        5.6      628,153,059      1  628,153,059.0  628,153,059.0  628,153,059  628,153,059           0.0  [CUDA memcpy Device-to-Host]

    [9/9] Executing 'cuda_gpu_mem_size_sum' stats report

    Total (MB)  Count  Avg (MB)   Med (MB)   Min (MB)   Max (MB)   StdDev (MB)           Operation          
    ----------  -----  ---------  ---------  ---------  ---------  -----------  ----------------------------
    50,000.000    100    500.000    500.000    500.000    500.000        0.000  [CUDA memcpy Host-to-Device]
    1,677.722      1  1,677.722  1,677.722  1,677.722  1,677.722        0.000  [CUDA memcpy Device-to-Host]

cuDART performance will largely be dependent on the sizes of files rendered, and the quality of the I/O environment rather than the GPU architecture itself. 
As I/O can represent a large fraction of the runtime, cuDART is structured to perform as few reads as possible (see the :ref:`code structure <structure_header>` documentation), and render as many images as possible for 
each load.


When :code:`Scene.render` is called with :code:`verbose_cpp = True`, the C++ executable will print a series of timestamps to the command line;
these times track the CPU clock, a more accurate wallclock duration is output as :code:`wallclock.txt` in the output directory.
