// standard library imports
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <initializer_list>
#include <ranges>
#include <vector>
#include <map>
#include <filesystem>

// custom external library imports
#include "npy.hpp"

// custom local library imports 
#include "vec3.hpp"
#include "ray.hpp"
#include "meshblock.hpp"
#include "tools.hpp"
#include "camera.hpp"
#include "mesh.hpp"

int main(int argc, char *argv[]) {

    // start general timer
    clock_t main_start = clock();

    // define space for user settings
    std::string cudart_version = "version 0.9 - April 2026";
    char *input_char = nullptr, *save_char = nullptr, *camera_char = nullptr, *mem_char = nullptr;
    char *doppler_char = nullptr, *power_law_char = nullptr;
    bool verbose = false, relativistic = false, append_mode = false, lookback = false, flexload = false;

    // process command line arguments
    for (int i = 1; i < argc; i++) {
        // check if argv[i] is a 2 character string of form "-X"
        if (*argv[i] == '-' && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {
            // check command line arguments
            char opt_letter = *(argv[i]+1);
            switch (opt_letter) { // parse options without arguments
                case 'h':
                    break;
                case 'v':
                    break;
                case 'r':
                    break;
                case 'l':
                    break;
                case 'f':
                    break;
                case 'a':
                    break;
                default:
                    if ((i+1 >= argc) || (*argv[i+1] == '-')) {
                        std::stringstream err_msg;
                        err_msg << "### FATAL ERROR in main ###\n";
                        err_msg << "-" << opt_letter << "must be followed by a valid argument\n";
                        CUDART_ERROR(err_msg);
                    }
            } // end cases
            switch (*(argv[i]+1)) { //
                case 'i':
                    input_char = argv[++i];
                    break;
                case 's':
                    save_char = argv[++i];
                    break;
                case 'm':
                    mem_char = argv[++i];
                    break;
                case 'd':
                    doppler_char = argv[++i];
                    break;
                case 'p':
                    power_law_char = argv[++i];
                    break;
                case 'v':
                    verbose = true;
                    break;
                case 'r':
                    relativistic = true;
                    break;
                case 'a':
                    append_mode = true;
                    break;
                 case 'l':
                    lookback = true;
                    break;
                case 'c':
                    camera_char = argv[++i];
                    break;
                case 'f':
                    flexload = true;
                    break;
                case 'h':
                default:
                    std::cout << "cuDART " << cudart_version << std::endl;
                    std::cout << "Usage: " << argv[0] << " [options]\n";
                    std::cout << "Options:\n";
                    std::cout << " -i <file>    specify input target (directory or .npy)\n";
                    std::cout << " -s <file>    specify save target (directory)\n";
                    std::cout << " -c <file>    specify camera data file\n";
                    std::cout << " -p <value>   power-law for rest-frame emission (default -0.6)\n";
                    std::cout << " -d <value>   Doppler index for boosting (deprecated for power-law)\n";
                    std::cout << " -m <value>   max VRAM in GB\n";
                    std::cout << " -l           lookback routine flag\n";
                    std::cout << " -f           flexible load flag\n";
                    std::cout << " -a           summation append flag\n";
                    std::cout << " -r           relativisitic boosting flag\n";
                    std::cout << " -v           verbosity flag\n";
                    std::cout << " -h           this help message\n"; 
                    return 0; 
            } // end cases
        } // end 2 char check
    }

    // handle fatal errors in input
    if (input_char == nullptr || save_char == nullptr) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main\n";
        err_msg << "No input file or output file specified.\n";
        CUDART_ERROR(err_msg);
    }
    std::string save_str_header(save_char);

    float power_law_index = -0.6; // default value for synchrotron emission
    float doppler_index = 2.0 - power_law_index;
    if (doppler_char != nullptr) {
        doppler_index = static_cast<float>(std::atof(doppler_char));
    }
    if (power_law_char != nullptr) { // priority over doppler specification
        power_law_index = static_cast<float>(std::atof(power_law_char));
        doppler_index = 2.0 - power_law_index;
    }
    
    // determine run mode (lookback)
    const std::string input_str(input_char);
    const std::filesystem::path input_path(input_char);
    if (lookback && !std::filesystem::is_directory(input_path)) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main\n";
        err_msg << "Lookback mode requires directory of data to function\n";
        CUDART_ERROR(err_msg);
    }

    // determine input mode (labelled/unlabelled)
    bool labelled_data = false;   
    if (!lookback) {
        if (std::filesystem::is_directory(input_path)) {
            labelled_data = true;
        } else {
            std::string npy_suffix = ".npy";
            if (input_path.extension() != npy_suffix) {
                std::stringstream err_msg;
                err_msg << "### FATAL ERROR in main\n";
                err_msg << "Input path must be .npy file (unlabelled data) or directory (labelled data)\n";
                CUDART_ERROR(err_msg);
            }
        } // end if input not dir
    } // end if !lookback

    // package trace info (TEMP, consider importing within lookback header)
    TraceArgs trace_args;
    trace_args.relativistic = relativistic;
    trace_args.doppler_index = doppler_index;
    trace_args.lookback = lookback;
    // the following are dummy values, overwritten in lookback routine
    trace_args.snapshot_dt = 1.0; 
    trace_args.inv_snapshot_dt = 1.0 / trace_args.snapshot_dt;
    trace_args.c = 1.0;
    trace_args.inv_c = 1.0 / trace_args.c; 
    trace_args.snapshot_index = 0; 
    trace_args.last_snapshot = 0;
    trace_args.last_time = 0;
    trace_args.num_snapshots = 0;
    trace_args.keep_edge = 0; // TODO: add option in header for this

    // print timing header
    if (verbose) {
        std::cout << "=============================================================\n";
        std::cout << "|      Activity        |    Location    |      Duration     |\n";
        std::cout << "=============================================================\n";
    }

    // load camera data and store in vector
    std::vector<Camera> cameras = load_cameras(camera_char, verbose);
    int num_images = cameras.size();
    size_t num_zero_pad = 5;

    // grab extremal camera properties for flexload (only important for lookback)
    float camera_r_min = std::numeric_limits<float>::max();
    float camera_r_max = std::numeric_limits<float>::min();
    float camera_t_min = camera_r_min;
    float camera_t_max = camera_r_max;
    for (auto &camera : cameras) {
        float camera_r = camera.origin.vector_mag();
        camera_r_min = (camera_r < camera_r_min) ? camera_r : camera_r_min;
        camera_r_max = (camera_r > camera_r_max) ? camera_r : camera_r_max;
        float camera_t_obs = camera.t_obs;
        camera_t_min = (camera_t_obs < camera_t_min) ? camera_t_obs : camera_t_min;
        camera_t_max = (camera_t_obs > camera_t_max) ? camera_t_obs : camera_t_max;
    } // end camera loop

    std::cout << "t_min = " << camera_t_min << ", t_max = " << camera_t_max << std::endl;
    std::cout << "r_min = " << camera_r_min << ", r_max = " << camera_r_max << std::endl;

    // inherit image dimensions from the first camera
    Camera standard_camera = cameras[0];
    int num_pixels = standard_camera.num_pixels;
    const size_t bytes_in_img = num_pixels * sizeof(float);

    // initialise image space on device 
    clock_t d_img_alloc_start = clock();
    float *d_img = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
    if (verbose) {
        float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image              (device)            %.6fs\n",d_img_alloc_dur);
    }

    // define render shape    
    int tx = 16, ty = 16; // must not exceed 1024 (max thread per block)
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)standard_camera.num_pixels_X / tx), 
                                std::ceil((float)standard_camera.num_pixels_Y / ty));

    // declare output container
    npy::npy_data_ptr<float> npy_img;
    npy_img.shape = {(unsigned long)standard_camera.num_pixels_X, (unsigned long)standard_camera.num_pixels_Y};

    // MAJOR CASE BREAK: w or w/o lookback
    if (lookback) {
        // run with lookback
        // 1. allocate space on device for data
        // 2. loop over snapshots, load data to host, copy to device
        // 3. loop over cameras, append to disc within loop

        // allocate image space on host
        // in lookback mode, each camera gets its own image space in host (device is reused)
        clock_t buffer_alloc_start = clock();
        size_t bytes_in_all_images = bytes_in_img * num_images;
        float *img_buffer = (float*) malloc(bytes_in_all_images);
        for (int i = 0; i < num_images * num_pixels; i++) {
            img_buffer[i] = 0.0; // init as zero, in prep for summation over m
        }
        if (verbose) {
            float buffer_alloc_dur = (float)(clock() - buffer_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc/init image buffer  (host)              %.6fs\n",buffer_alloc_dur);
        }

        // allocate scratch space for image sum on host
        clock_t scratch_alloc_start = clock();
        float *img_scratch = (float*) malloc(bytes_in_img);
        if (verbose) {
            float scratch_alloc_dur = (float)(clock() - scratch_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc buffer             (host)              %.6fs\n",scratch_alloc_dur);
        }

        // load header data from load dir
        // expect single line in form:
        // num_snapshots max_snapshot_size snapshot_dt L_domain
        std::string header_str = input_str + "/header.txt";
        std::ifstream header_file(header_str);
        int num_snapshots, max_snapshot_size;
        float snapshot_dt; // in units of Myr
        float L_domain; // if unlabelled data: length of longest side of domain in kpc
                        // if labelled data: length of unity in mb_info
        if (header_file.is_open()) {
            std::string line;
            int line_count = 0;
            while (std::getline(header_file, line)) {
                std::istringstream iss(line);
                if (!(iss >> num_snapshots >> max_snapshot_size >> snapshot_dt >> L_domain)) {
                    std::stringstream err_msg;
                    err_msg << "### FATAL ERROR in main ###\n";
                    err_msg << "Unable to parse line " << line_count << " of snapshot header file at " << header_str << std::endl;
                    CUDART_ERROR(err_msg);
                } else{
                    break; // read only first line
                }
            } // end while line
        } // end file open
        
        // set trace args
        trace_args.snapshot_dt = snapshot_dt;                           // in Myr
        trace_args.inv_snapshot_dt = 1.0 / trace_args.snapshot_dt;      // in Myr^{-1}
        float kpc_to_m = 3.086e+19;                                     // in m
        float Myr_to_s = 1e6 * 365 * 24 * 60 * 60;                      // in s
        float c_light = 3e8;                                            // in m/s
        float velocity_code_units = L_domain * kpc_to_m / Myr_to_s;     
        float c_in_code_units = c_light / velocity_code_units;
        float c_in_kpc_per_Myr = c_light * Myr_to_s / kpc_to_m;
        trace_args.c = c_in_code_units;
        trace_args.inv_c = 1.0 / trace_args.c; 
        trace_args.num_snapshots = num_snapshots;
        trace_args.last_snapshot = num_snapshots - 1;
        trace_args.last_time = trace_args.last_snapshot * trace_args.snapshot_dt;

        // float calc flexload limits
        int m_lower = 0, m_upper = num_snapshots - 1; // if no flexload, use full time range
        if (flexload) { // TODO: this can also be performed more simply in code units
            float domain_r_max = std::sqrt(3.0); // TODO: load this as part of header (via mesh_xr, mesh_xl)
            float d_min_in_kpc = (camera_r_min - domain_r_max) * L_domain;          // minimum camera-domain seperation
            float d_max_in_kpc = (camera_r_max - domain_r_max) * L_domain;          // maixmum camera-domain seperation
            std::cout << "d_min_in_kpc" << d_min_in_kpc << ", d_max_in_kpc = " << d_max_in_kpc << std::endl;
            float t_min_in_Myr = camera_t_min - d_max_in_kpc / c_in_kpc_per_Myr;    // earliest contributing snapshot time
            float t_max_in_Myr = camera_t_max - d_min_in_kpc / c_in_kpc_per_Myr;    // latest contributing snapshot time
            std::cout << "t_min_in_Myr" << t_min_in_Myr << ", t_max_in_Myr = " << t_max_in_Myr << std::endl;
            int m_min = std::floor(t_min_in_Myr * trace_args.inv_snapshot_dt);          // earliest contributing snapshot index
            int m_max = std::ceil(t_max_in_Myr * trace_args.inv_snapshot_dt);           // latest contributing snapshot index 
            std::cout << "m_min = " << m_min << ", m_max = " << m_max << std::endl;
            m_lower = (m_min > m_lower) ? m_min : m_lower;                          // start loop at earliest
            m_upper = (m_max < m_upper) ? m_max : m_upper;                          // end loop at latest
            std::cout << "m_lower = " << m_lower << ", m_upper = " << m_upper << std::endl;
        }

        // allocate data on host
        clock_t h_alloc_start = clock();
        size_t h_bytes = max_snapshot_size * sizeof(float);
        float *h_all_data = (float*) malloc(h_bytes);
        if (verbose) { 
            float h_alloc_dur = (float)(clock() - h_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc data               (host)              %.6fs\n",h_alloc_dur);
        }

        // determine VRAM limitations and handle excess
        float tolerance = 0.95; // use this fraction of available vram
        size_t d_bytes = calc_vram_limit(mem_char, tolerance, h_bytes);
        if (h_bytes > d_bytes) {
            std::stringstream err_msg;
            err_msg << "### FATAL ERROR in main\n";
            err_msg << "Requested memory in excess of space on device\n";
            CUDART_ERROR(err_msg);
        }

        // allocate space on device
        clock_t d_data_alloc_start = clock();
        float *d_data = nullptr;
        checkCudaErrors(cudaMalloc(&d_data, d_bytes));
        if (verbose) {
            float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc data               (device)            %.6fs\n",d_data_alloc_dur);
        }

        // loop over snapshots
        if (verbose) {
            std::cout << "=============================================================\n";
            if (num_snapshots == 1) {
                std::cout << "Starting render for single snapshot...\n";
            } else {
                std::cout << "Starting render cycle for " << num_snapshots << " snapshots...\n";
            }
            std::cout << "-------------------------------------------------------------\n";
        }
        for (int m = m_lower; m <= m_upper; m++) {

            clock_t snapshot_start = clock();

            // update trace_args
            trace_args.snapshot_index = m;

            // import npy data to host
            std::vector<MeshBlockInfo> all_mb_info;
            bool host_malloc = false;
            
            // prep empty containers
            MeshBlock **mb_list;
            Mesh **mesh;
            int num_meshblocks = 0;

            if (labelled_data) { // TODO: add suppoort for labelled lookback
                std::string snapshot_str = input_str + "/snapshot" + zero_pad_str(m, num_zero_pad);
                all_mb_info = load_labelled_meshblocks(snapshot_str, h_all_data, h_bytes, trace_args.relativistic, verbose, host_malloc);
            } else {
                std::string snapshot_str = input_str + "/snapshot" + zero_pad_str(m, num_zero_pad) + ".npy";
                all_mb_info = load_unlabelled_meshblock(snapshot_str, h_all_data, h_bytes, trace_args.relativistic, verbose, host_malloc);
            }
            num_meshblocks = all_mb_info.size();
        
            // copy all data from host into device
            clock_t data_copy_start = clock();
            checkCudaErrors(cudaMemcpy(d_data, h_all_data, d_bytes, cudaMemcpyHostToDevice)); 
            checkCudaErrors(cudaPeekAtLastError());
            if (verbose) {
                float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
                printf("memcpy data               (host->device)      %.6fs\n",data_copy_dur);
            }

            // initialise MeshBlock list on device
            build_containers(all_mb_info, d_data, mb_list, mesh, verbose);

            // loop over cameras 
            int img_count = 0;
            for (auto &camera : cameras) {
                
                // perform flexload check against domain size
                if (flexload) {
                    float d_min_mesh = std::numeric_limits<float>::max();
                    float d_max_mesh = std::numeric_limits<float>::min();
                    for (int mb_id = 0; mb_id < num_meshblocks; mb_id++) {
                        // define sphere which contains subdomain (TODO:offload to mbinfo)
                        vec3 mb_center = 0.5 * (all_mb_info[mb_id].xl + all_mb_info[mb_id].xr);
                        float mb_radius = (all_mb_info[mb_id].xl - mb_center).vector_mag();

                        // define radius which contains camera
                        float camera_radius = (camera.origin - camera.lower_left).vector_mag();

                        // calculate extremal camera-domain seperations
                        float center_sep = (mb_center - camera.origin).vector_mag();
                        float d_min_mb = center_sep - camera_radius - mb_radius;
                        float d_max_mb = center_sep + camera_radius + mb_radius;

                        // store mesh extrema
                        d_min_mesh = (d_min_mb < d_min_mesh) ? d_min_mb : d_min_mesh;
                        d_max_mesh = (d_max_mb > d_max_mesh) ? d_max_mb : d_max_mesh;
                    } // end mb loop

                    std::cout << "d_min_mesh = " << d_min_mesh << std::endl;
                    std::cout << "d_max_mesh = " << d_max_mesh << std::endl;

                    // find extremal snapshots in time
                    float t_min_in_Myr = camera.t_obs - d_max_mesh * L_domain / c_in_kpc_per_Myr;
                    float t_max_in_Myr = camera.t_obs - d_min_mesh * L_domain / c_in_kpc_per_Myr;
                
                    int m_min = std::floor(t_min_in_Myr * trace_args.inv_snapshot_dt);
                    int m_max = std::ceil(t_max_in_Myr * trace_args.inv_snapshot_dt);

                    // check if this camera can recieve contributions from this snapshot
                    if ((m < m_min) || (m > m_max)) {
                        if (verbose) {
                            std::cout << ".............................................................\n";
                            std::cout << "no overlap between snapshot " << m << " and camera " << img_count << ", skippping.";
                            std::cout << ".............................................................\n";
                        }
                        continue;
                    }
                } // end flexload
                

                if (verbose) {
                    std::cout << ".............................................................\n";
                }

                clock_t this_img_start = clock();

                // call render
                clock_t render_start = clock();
                render_from_mesh<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mesh, trace_args);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                if (verbose) {
                    float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
                    printf("render kernel             (device)            %.6fs\n",render_dur);
                }

                // copy image data to scratch space, and sum into main buffer
                clock_t img_copy_start = clock();
                checkCudaErrors(cudaMemcpy(img_scratch, d_img, bytes_in_img, cudaMemcpyDeviceToHost));
                for (int i = 0; i < num_pixels; i++) {
                    img_buffer[i + img_count * num_pixels] += img_scratch[i];
                }
                if (verbose) {
                    float img_copy_dur = (float)(clock() - img_copy_start)/CLOCKS_PER_SEC;
                    printf("memcpy/sum image          (device->host)      %.6fs\n",img_copy_dur);
                }

                // clear d_img as prep for next render call
                wipe_img<<<blocks_per_grid,threads_per_block>>>(standard_camera, d_img);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                img_count++;
            } // end camera loop

            if (verbose) {
                float snapshot_dur = (float)(clock() - snapshot_start)/CLOCKS_PER_SEC;
                std::cout << ".............................................................\n";
                printf("snapshot total            (host/device)       %.6fs\n",snapshot_dur);
                std::cout << "=============================================================\n";
            }

            // prepare for next snapshot
            free_mesh<<<1,1>>>(mesh, num_meshblocks);
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaFree(mesh));
            checkCudaErrors(cudaFree(mb_list));
        } // end snapshot loop

        // image buffer populated, save render data as npy
        clock_t npy_write_start = clock();
        for (int n = 0; n < num_images; n++) {
            std::string save_str = save_str_header + "/raw" + zero_pad_str(n, num_zero_pad) + ".npy";
            if (append_mode) {
                // attempt to add values to existing file (if it exists)
                bool file_exists = std::filesystem::is_regular_file(save_str);
                if (file_exists) { 
                    npy::npy_data existing_npy_data = npy::read_npy<float>(save_str);
                    std::vector<unsigned long> existing_npy_shape = existing_npy_data.shape;
                    if (existing_npy_shape != npy_img.shape) {
                        std::stringstream err_msg;
                        err_msg << "### FATAL ERROR in main\n";
                        err_msg << "Dimensions of existing npy data at " << save_str << " does not match standard camera.\n";
                        CUDART_ERROR(err_msg);
                    }
                    std::vector<float> img_vec = existing_npy_data.data;
                    for (int i = 0; i < standard_camera.num_pixels; i++) {
                        img_buffer[i + n * num_pixels] += img_vec[i]; // add existing data to img buffer
                    }
                }
            }
            // point npy container to img sub-buffer
            npy_img.data_ptr = img_buffer + n * num_pixels;
            npy::write_npy(save_str, npy_img);
        }

        if (verbose) {
            float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
            printf("write all raw image       (host->npy)         %.6fs\n",npy_write_dur);
        }

        // perform cleanup of device/host data
        clock_t free_start = clock();
        checkCudaErrors(cudaFree(d_img));
        checkCudaErrors(cudaFree(d_data));
        free(h_all_data);
        free(img_buffer);
        free(img_scratch);
        cudaDeviceReset();
        if (verbose) {
            float free_dur = (float)(clock() - free_start)/CLOCKS_PER_SEC;
            printf("free all                  (device/host)       %.6fs\n",free_dur);
        }
    } else {
        // run without lookback
        // 1. load data to host, allocate space on device, copy to device
        // 2. build containers on device
        // 3. loop over cameras, save to disc within loop

        // allocate image space on host
        clock_t img_alloc_start = clock();
        float *img = (float*) malloc(bytes_in_img);
        if (verbose) {
            float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc image              (host)              %.6fs\n",img_alloc_dur);
        }

        // import npy data to host
        std::vector<MeshBlockInfo> all_mb_info;
        float *h_all_data = nullptr;
        size_t h_bytes = 0;
        bool host_malloc = true;
        if (labelled_data) {
            all_mb_info = load_labelled_meshblocks(input_str, h_all_data, h_bytes, trace_args.relativistic, verbose, host_malloc);
        } else {
            all_mb_info = load_unlabelled_meshblock(input_str, h_all_data, h_bytes, trace_args.relativistic, verbose, host_malloc);
        }
        int num_meshblocks = all_mb_info.size();

        // determine VRAM limitations and handle excess
        float tolerance = 0.95; // use this fraction of available vram
        size_t d_bytes = calc_vram_limit(mem_char, tolerance, h_bytes);
        if (h_bytes > d_bytes) {
            std::stringstream err_msg;
            err_msg << "### FATAL ERROR in main\n";
            err_msg << "Requested memory in excess of space on device\n";
            CUDART_ERROR(err_msg);
        }

        // allocate space on device
        clock_t d_data_alloc_start = clock();
        float *d_data = nullptr;
        checkCudaErrors(cudaMalloc(&d_data, d_bytes));
        if (verbose) {
            float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc data               (device)            %.6fs\n",d_data_alloc_dur);
        }

        // copy ALL data from host into device
        clock_t data_copy_start = clock();
        checkCudaErrors(cudaMemcpy(d_data, h_all_data, d_bytes, cudaMemcpyHostToDevice)); 
        checkCudaErrors(cudaPeekAtLastError());
        if (verbose) {
            float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
            printf("memcpy data               (host->device)      %.6fs\n",data_copy_dur);
        }

        // initialise MeshBlock list on device
        MeshBlock **mb_list;
        Mesh **mesh;
        build_containers(all_mb_info, d_data, mb_list, mesh, verbose);

        // iterate over cameras
        int img_count = 0;
        if (verbose) {
            std::cout << "=============================================================\n";
            if (num_images == 1) {
                std::cout << "Starting render for single image...\n";
            } else {
                std::cout << "Starting render cycle for " << num_images << " images...\n";
            }
            std::cout << "-------------------------------------------------------------\n";
        }
        for (auto &camera : cameras) {
            
            clock_t this_img_start = clock();

            // call render
            clock_t render_start = clock();
            render_from_mesh<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mesh, trace_args);
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            if (verbose) {
                float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
                printf("render kernel             (device)            %.6fs\n",render_dur);
            }

            // copy image data to host
            clock_t img_copy_start = clock();
            checkCudaErrors(cudaMemcpy(img, d_img, bytes_in_img, cudaMemcpyDeviceToHost));
            if (verbose) {
                float img_copy_dur = (float)(clock() - img_copy_start)/CLOCKS_PER_SEC;
                printf("memcpy image              (device->host)      %.6fs\n",img_copy_dur);
            }

            // save data
            clock_t npy_write_start = clock();
            std::string save_str = save_str_header + "/raw" + zero_pad_str(img_count, num_zero_pad) + ".npy";
            if (append_mode) {
                // attempt to add values to existing file (if it exists)
                bool file_exists = std::filesystem::is_regular_file(save_str);
                if (file_exists) { 
                    npy::npy_data existing_npy_data = npy::read_npy<float>(save_str);
                    std::vector<unsigned long> existing_npy_shape = existing_npy_data.shape;
                    if (existing_npy_shape != npy_img.shape) {
                        std::stringstream err_msg;
                        err_msg << "### FATAL ERROR in main\n";
                        err_msg << "Dimensions of existing npy data at " << save_str << " does not match standard camera.\n";
                        CUDART_ERROR(err_msg);
                    }
                    std::vector<float> img_vec = existing_npy_data.data;
                    for (int i = 0; i < standard_camera.num_pixels; i++) {
                        img[i] += img_vec[i];
                    }
                }
            }
            npy_img.data_ptr = img;
            npy::write_npy(save_str, npy_img);
            if (verbose) {
                float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
                printf("write raw image           (host->npy)         %.6fs\n",npy_write_dur);
                float this_img_dur = (float)(clock() - this_img_start)/CLOCKS_PER_SEC;
                printf("img total                 (host/device)       %.6fs\n",this_img_dur);
                if (img_count == num_images - 1) {
                    std::cout << "=============================================================\n";
                } else {
                    std::cout << "-------------------------------------------------------------\n";
                }
            }

            // prepare for next image
            if (num_images != 1) { // skip wipe if single image output
                wipe_img<<<blocks_per_grid,threads_per_block>>>(standard_camera, d_img);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                img_count++;
            } 
        } // end camera loop

        // perform cleanup of device/host data
        clock_t free_start = clock();
        free_mesh<<<1,1>>>(mesh, num_meshblocks);
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(d_img));
        checkCudaErrors(cudaFree(mesh));
        checkCudaErrors(cudaFree(d_data));
        checkCudaErrors(cudaFree(mb_list));
        free(h_all_data);
        free(img);
        cudaDeviceReset();
        if (verbose) {
            float free_dur = (float)(clock() - free_start)/CLOCKS_PER_SEC;
            printf("free all                  (device/host)       %.6fs\n",free_dur);
        }
    } // end if lookback

    // terminate
    if (verbose) {
        float main_dur = (float)(clock() - main_start)/CLOCKS_PER_SEC;
        printf("total runtime                                 %.6fs\n",main_dur);
        std::cout << "=============================================================\n";
        printf("cuDART terminated.\n");
    }

    return 0;
}