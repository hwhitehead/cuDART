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
#include <chrono>
#include <ctime>

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

    // print start time
    auto start_time_clock = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start_time_clock);
    std::cout << "Starting cuDART backend at " << std::ctime(&start_time) << std::endl;

    // define space for user settings
    std::string cudart_version = "version 0.9 - August 2026";
    char *input_char = nullptr, *save_char = nullptr, *camera_char = nullptr, *mem_char = nullptr;
    char *doppler_char = nullptr, *power_law_char = nullptr;
    bool verbose = false, relativistic = false, append_mode = false, lookback = false, flexload = false, keep_edge = false;

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
                case 'k':
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
                case 'k':
                    keep_edge = true;
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
    } // end cl parse

    // handle fatal errors in command line arguments
    if (input_char == nullptr || save_char == nullptr) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main ###\n";
        err_msg << "No input path (-i) or output path (-s) specified.\n";
        CUDART_ERROR(err_msg);
    }
    std::string save_str_header(save_char);             // cast save_char into string

    // process power law arguments
    float power_law_index = -0.6;                       // default value for synchrotron emission
    float doppler_index = 2.0 - power_law_index;        // two factors from Lorentz transform
    if (doppler_char != nullptr) {                      // cast doppler_char into float
        doppler_index = static_cast<float>(std::atof(doppler_char));
    }
    if (power_law_char != nullptr) {                    // cast power_law_char into float
        power_law_index = static_cast<float>(std::atof(power_law_char));
        doppler_index = 2.0 - power_law_index;          // power_law_char overrules doppler_char
    }
    
    // check existence for input path
    const std::string input_str(input_char);
    const std::filesystem::path input_path(input_char);
    if (!std::filesystem::exists(input_path)) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main ###\n";
        err_msg << "Unable to locate input path " << input_str << std::endl;
        CUDART_ERROR(err_msg);
    }

    // check compatibility of lookback usage with input path
    if (lookback && !std::filesystem::is_directory(input_path)) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main ###\n";
        err_msg << "Lookback mode requires directory of data to function\n";
        CUDART_ERROR(err_msg);
    }

    // package trace info 
    TraceArgs trace_args;
    // the following are user-defined values, set via command line arguments
    trace_args.relativistic = relativistic;
    trace_args.doppler_index = doppler_index;
    trace_args.lookback = lookback;
    // the following are dummy values, overwritten in lookback/no-lookback routines
    trace_args.t_obs = 0.0;
    trace_args.snapshot_dt = 1.0; 
    trace_args.inv_snapshot_dt = 1.0 / trace_args.snapshot_dt;
    trace_args.c = 1.0;
    trace_args.inv_c = 1.0 / trace_args.c; 
    trace_args.snapshot_index = 0; 
    trace_args.last_snapshot = 0;
    trace_args.last_time = 0;
    trace_args.num_snapshots = 0;
    trace_args.keep_edge = keep_edge; 

    // print timing header
    if (verbose) {
        std::cout << "=============================================================\n";
        std::cout << "|      Activity        |    Location    |      Duration     |\n";
        std::cout << "=============================================================\n";
    }

    // load camera data from .txt file into vector
    std::vector<Camera> cameras = load_cameras(camera_char, verbose);
    int num_images = cameras.size();

    // inherit image dimensions from the first camera
    Camera standard_camera = cameras[0];
    int num_pixels = standard_camera.num_pixels;
    const size_t bytes_in_img = num_pixels * sizeof(float);

    // define render shape    
    int tx = 16, ty = 16; 
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)standard_camera.num_pixels_X / tx), 
                                std::ceil((float)standard_camera.num_pixels_Y / ty));

    // declare container for libnpy write
    npy::npy_data_ptr<float> npy_img;
    npy_img.shape = {(unsigned long)standard_camera.num_pixels_X, (unsigned long)standard_camera.num_pixels_Y};

    // MAJOR CASE BREAK: w or w/o lookback
    if (lookback) {
        // run with lookback (finite light delay)
        // 1. allocate space for: image (host), data (host), data (device) and image (device)
        // 2. loop over snapshots, load data to host, copy to device
        // 3. loop over cameras, save images to communal buffer on device
        // 4. return image buffer to host

        // allocate image space on host for ALL images
        clock_t buffer_alloc_start = clock();
        size_t bytes_in_all_images = bytes_in_img * num_images;
        float *h_img_buffer = (float*) malloc(bytes_in_all_images);
        for (int i = 0; i < num_images * num_pixels; i++) {
            h_img_buffer[i] = 0.0; // init as zero, in prep for summation over snapshots (index m)
        }
        if (verbose) {
            float buffer_alloc_dur = (float)(clock() - buffer_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc/init image buffer  (host)              %.6fs\n",buffer_alloc_dur);
        }

        // load header data for all snapshots from load dir
        // expect single line in form:
        // num_snapshots max_snapshot_size snapshot_dt L_domain
        std::string header_str = input_str + "/header.txt";
        std::ifstream header_file(header_str);
        int num_snapshots, max_snapshot_size;
        float snapshot_dt;  // code time in units of Myr
        float L_domain;     // code length in units of kpc (if unlabelled longest domain size automatically unity)
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
        
        // save unit data in trace_args
        trace_args.snapshot_dt = snapshot_dt;                           // in Myr
        trace_args.inv_snapshot_dt = 1.0 / trace_args.snapshot_dt;      // in Myr^{-1}
        float velocity_code_units = L_domain * kpc_to_m / Myr_to_s;     
        float c_in_code_units = c_light / velocity_code_units;
        trace_args.c = c_in_code_units;
        trace_args.inv_c = 1.0 / trace_args.c; 
        trace_args.num_snapshots = num_snapshots;
        trace_args.last_snapshot = num_snapshots - 1;
        trace_args.last_time = trace_args.last_snapshot * trace_args.snapshot_dt;

        // allocate data space on host
        clock_t h_alloc_start = clock();
        size_t h_bytes = max_snapshot_size * sizeof(float); // ensure space for largest snapshot
        float *h_data_buffer = (float*) malloc(h_bytes);
        if (verbose) { 
            float h_alloc_dur = (float)(clock() - h_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc data               (host)              %.6fs\n",h_alloc_dur);
        }

        // determine VRAM limitations and handle excess
        float tolerance = 0.95; // use this fraction of available vram
        size_t d_bytes = calc_vram_limit(mem_char, tolerance, h_bytes);
        if (h_bytes > d_bytes) {
            std::stringstream err_msg;
            err_msg << "### FATAL ERROR in main ###\n";
            err_msg << "Requested memory in excess of space on device\n";
            CUDART_ERROR(err_msg);
        }

        // allocate data space on device
        clock_t d_data_alloc_start = clock();
        float *d_data_buffer = nullptr;
        checkCudaErrors(cudaMalloc(&d_data_buffer, d_bytes));
        if (verbose) {
            float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc data               (device)            %.6fs\n",d_data_alloc_dur);
        }

        // allocate image space on device (communal buffer, all images)
        clock_t d_img_buffer_alloc_start = clock();
        float *d_img_buffer = nullptr;
        checkCudaErrors(cudaMalloc((void **)&d_img_buffer, bytes_in_img * num_images));
        if (verbose) {
            float d_img_buffer_alloc_dur = (float)(clock() - d_img_buffer_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc image buffer       (device)            %.6fs\n",d_img_buffer_alloc_dur);
        }

        // device image buffer is treated as additive in lookback mode, require init before first render
        wipe_img<<<blocks_per_grid,threads_per_block>>>(standard_camera, d_img_buffer);
        checkCudaErrors(cudaPeekAtLastError());

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
        bool host_malloc = false;           // in lookback mode, host buffer is already allocated
        int num_snapshots_loaded = 0;       // flexload metadata (verbose only)
        int flexload_renders_skipped = 0;   // flexload metadata (verbose only)
        for (int m = 0; m < num_snapshots; m++) {

            // start timer for this snapshot
            clock_t snapshot_start = clock();

            // stash snapshot index in trace_args
            trace_args.snapshot_index = m;

            // prep containers for snapshot
            std::vector<MeshBlockInfo> all_mb_info;
            MeshBlock **mb_list;
            Mesh **mesh;
            int num_meshblocks = 0;

            // load snapshot, auto detect labelled state
            const std::string snapshot_dir_str = input_str + "/snapshot" + zero_pad_str(m, num_zero_pad);
            const std::filesystem::path snapshot_dir_path(snapshot_dir_str);
            if (std::filesystem::is_directory(snapshot_dir_path)) { // snapshot path is directory, load as labelled data
                all_mb_info = load_labelled_meshblocks(snapshot_dir_str, h_data_buffer, h_bytes, trace_args.relativistic, verbose, host_malloc);
            } else { // snapshot path is file, load as unlabelled data
                const std::string snapshot_npy_str = input_str + "/snapshot" + zero_pad_str(m, num_zero_pad) + ".npy";
                if (std::filesystem::exists(snapshot_npy_str)) {
                    all_mb_info = load_unlabelled_meshblock(snapshot_npy_str, h_data_buffer, h_bytes, trace_args.relativistic, verbose, host_malloc);
                } else {
                    std::stringstream err_msg;
                    err_msg << "### FATAL ERROR in main ###\n";
                    err_msg << "Unable to locate snapshot " << m << " in " << input_str << std::endl;
                    CUDART_ERROR(err_msg);
                }
            } // end mb load
            num_meshblocks = all_mb_info.size();

            // flexload: given camera and meshblock info, determine if snapshot can contribute to ANY camera
            // if no temporal overlap, skip load
            if (flexload) {
                bool snapshot_contributes = false;
                for (auto &camera : cameras) {
                    float d_min_mesh = std::numeric_limits<float>::max();
                    float d_max_mesh = std::numeric_limits<float>::min();
                    float camera_radius = (camera.origin - camera.lower_left).vector_mag();     // size of binding sphere for camera plane
                    for (auto &mb_info : all_mb_info) {                                         // loop over all MeshBlocks in snapshot
                        // calculate extremal camera-domain seperations
                        float center_sep = (mb_info.mb_origin - camera.origin).vector_mag();    // camera-domain origin seperation
                        float d_min_mb = center_sep - camera_radius - mb_info.mb_radius;        // minimum camera-domain seperation
                        float d_max_mb = center_sep + camera_radius + mb_info.mb_radius;        // maximum camera-domain seperation
                        // store mesh extrema
                        d_min_mesh = (d_min_mb < d_min_mesh) ? d_min_mb : d_min_mesh;
                        d_max_mesh = (d_max_mb > d_max_mesh) ? d_max_mb : d_max_mesh;
                    } // end mb loop

                    // check if this camera has temporal overlap with this snapshot
                    float t_min = camera.t_obs - d_max_mesh * trace_args.inv_c;                 // earliest contributing time for THIS camera
                    float t_max = camera.t_obs - d_min_mesh * trace_args.inv_c;                 // latest contributing time for THIS camera
                    int m_min = std::floor(t_min * trace_args.inv_snapshot_dt);                 // earliest contributing snapshot index for THIS camera
                    int m_max = std::ceil(t_max * trace_args.inv_snapshot_dt);                  // latest contributing snapshot index for THIS camera
                    if ((m >= m_min) && (m <= m_max)) {
                        snapshot_contributes = true;
                        camera.skip_render = false;
                    } else {
                        camera.skip_render = true;
                    } 
                } // end camera loop
                if (!snapshot_contributes) {
                    if (verbose) {
                        std::cout << ".............................................................\n";
                        std::cout << "no overlap of snapshot " << m << " and any camera, skippping load.";
                        std::cout << ".............................................................\n";
                    }
                    continue; // skip loading this snapshot
                }
            } // end flexload
            num_snapshots_loaded++;

            // copy all data from host into device
            clock_t data_copy_start = clock();
            checkCudaErrors(cudaMemcpy(d_data_buffer, h_data_buffer, d_bytes, cudaMemcpyHostToDevice)); 
            checkCudaErrors(cudaPeekAtLastError());
            if (verbose) {
                float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
                printf("memcpy data               (host->device)      %.6fs\n",data_copy_dur);
            }

            // initialise MeshBlock list on device
            build_containers(all_mb_info, d_data_buffer, mb_list, mesh, verbose);

            // loop over cameras 
            int img_count = 0;
            for (auto &camera : cameras) {
                
                // perform flexload check, given previous determination of temporal overlap between snapshot and camera
                if (flexload && camera.skip_render) {
                    flexload_renders_skipped++;
                    if (verbose) {
                        std::cout << ".............................................................\n";
                        std::cout << "no overlap of snapshot " << m << " and camera " << img_count << ", skippping render.";
                        std::cout << ".............................................................\n";
                    }
                    continue;
                } // end flexload check
                
                // stash camera properties
                trace_args.t_obs = camera.t_obs;
                trace_args.camera_index = img_count;

                if (verbose) {
                    std::cout << ".............................................................\n";
                }

                clock_t this_img_start = clock();

                // call render
                clock_t render_start = clock();
                render_from_mesh<<<blocks_per_grid,threads_per_block>>>(camera, d_img_buffer, mesh, trace_args);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                if (verbose) {
                    float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
                    printf("render kernel             (device)            %.6fs\n",render_dur);
                }

                // clear d_img as prep for next render call
                wipe_img<<<blocks_per_grid,threads_per_block>>>(standard_camera, d_img_buffer);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
                img_count++;
            } // end camera loop

            // report snapshot total duration
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
            num_snapshots_loaded++;
        } // end snapshot loop

        // copy image data from device buffer to image buffer
        clock_t img_copy_start = clock();
        checkCudaErrors(cudaMemcpy(h_img_buffer, d_img_buffer, num_images * bytes_in_img, cudaMemcpyDeviceToHost));
        if (verbose) {
            float img_copy_dur = (float)(clock() - img_copy_start)/CLOCKS_PER_SEC;
            printf("memcpy all images         (device->host)      %.6fs\n",img_copy_dur);
        }

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
                        err_msg << "### FATAL ERROR in main ###\n";
                        err_msg << "Dimensions of existing npy data at " << save_str << " does not match standard camera.\n";
                        CUDART_ERROR(err_msg);
                    }
                    std::vector<float> img_vec = existing_npy_data.data;
                    for (int i = 0; i < standard_camera.num_pixels; i++) {
                        h_img_buffer[i + n * num_pixels] += img_vec[i]; // add existing data to img buffer
                    }
                }
            }
            // point npy container to img sub-buffer
            npy_img.data_ptr = h_img_buffer + n * num_pixels;
            npy::write_npy(save_str, npy_img);
        }

        if (verbose) {
            float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
            printf("write all raw image       (host->npy)         %.6fs\n",npy_write_dur);
        }

        // perform cleanup of device/host data
        clock_t free_start = clock();
        checkCudaErrors(cudaFree(d_data_buffer));   // device data buffer
        checkCudaErrors(cudaFree(d_img_buffer));    // device image buffer
        free(h_data_buffer);                        // host data buffer
        free(h_img_buffer);                         // host image buffer
        cudaDeviceReset();
        if (verbose) {
            float free_dur = (float)(clock() - free_start)/CLOCKS_PER_SEC;
            printf("free all                  (device/host)       %.6fs\n",free_dur);
            if (flexload) {
                int nominal_total_renders = num_snapshots * num_images;                                 // total renders without flexload
                int true_total_renders = num_snapshots_loaded * num_images - flexload_renders_skipped;  // true render count
                int total_skipped_renders = nominal_total_renders - true_total_renders;                 // render delta
                int total_skipped_snapshots = num_snapshots - num_snapshots_loaded;                     // snapshot delta
                float perc_snapshot_reduction = 100.0 * total_skipped_snapshots / num_snapshots;   
                float perc_render_reduction = 100.0 * total_skipped_renders / nominal_total_renders;
                std::cout << "=============================================================\n";
                printf("Reporting flexload speedup....\n");
                printf("total snapshots skipped                       %d\n",total_skipped_snapshots);
                printf("total renders skipped                         %d\n",total_skipped_renders);
                printf("perc snapshot reduction                       %.3f%\n",perc_snapshot_reduction);
                printf("perc render reduction                         %.3f%\n",perc_render_reduction);
                std::cout << "=============================================================\n";
            }
        }
    } else {
        // run without lookback
        // 1. load data to host, allocate space on device, copy to device
        // 2. build containers on device
        // 3. loop over cameras, save to disc within loop

        // allocate space for single image on device 
        clock_t d_img_alloc_start = clock();
        float *d_img = nullptr;
        checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
        if (verbose) {
            float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc image              (device)            %.6fs\n",d_img_alloc_dur);
        }

        // allocate image space on host
        clock_t img_alloc_start = clock();
        float *img = (float*) malloc(bytes_in_img);
        if (verbose) {
            float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
            printf("malloc image              (host)              %.6fs\n",img_alloc_dur);
        }

        // import npy data to host, auto detect labelled state
        std::vector<MeshBlockInfo> all_mb_info;
        float *h_all_data = nullptr;
        size_t h_bytes = 0;
        bool host_malloc = true;
        if (std::filesystem::is_directory(input_path)) {
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
            err_msg << "### FATAL ERROR in main ###\n";
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
                        err_msg << "### FATAL ERROR in main ###\n";
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
    float main_dur = (float)(clock() - main_start)/CLOCKS_PER_SEC;
    if (verbose) {
        printf("total runtime                                 %.6fs\n",main_dur);
        std::cout << "=============================================================\n";
        printf("cuDART terminated.\n");
    }

    // write total duration to text file
    std::string wallclock_file_str = save_str_header + "/wallclock.txt";
    std::ofstream wallclock_file(wallclock_file_str);
    wallclock_file << main_dur;
    wallclock_file.close();

    // print start time
    auto end_time_clock = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end_time_clock);
    std::cout << "Ended cuDART backend at " << std::ctime(&end_time) << "." << std::endl;

    return 0;
}