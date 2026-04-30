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
    std::string cudart_version = "version 0.8 - April 2026";
    char *input_char = nullptr, *save_char = nullptr, *camera_char = nullptr, *mem_char = nullptr;
    char *doppler_char = nullptr, *power_law_char = nullptr;
    bool verbose = false, relativistic = false, append_mode = false;

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
                case 'c':
                    camera_char = argv[++i];
                    break;
                case 'h':
                default:
                    std::cout << "cuDART " << cudart_version << std::endl;
                    std::cout << "Usage: " << argv[0] << " [options]\n";
                    std::cout << "Options:\n";
                    std::cout << " -i <file>    specify input target\n";
                    std::cout << " -s <file>    specify save target\n";
                    std::cout << " -c <file>    specify camera data file\n";
                    std::cout << " -p <value>   power-law for rest-frame emission (default -0.6)\n";
                    std::cout << " -d <value>   Doppler index for boosting (deprecated for power-law)\n";
                    std::cout << " -m <value>   max VRAM in GB\n";
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
    
    // determine run mode type (labelled or unlabelled)
    bool labelled_data = false;
    const std::string input_str(input_char);
    const std::filesystem::path input_path(input_char);
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
    }

    // print timing header
    if (verbose) {
        std::cout << "=============================================================\n";
        std::cout << "|      Activity        |    Location    |      Duration     |\n";
        std::cout << "=============================================================\n";
    }

    // load camera data and store in vector
    std::vector<Camera> cameras = load_cameras(camera_char, verbose);
    
    // load image dimensions from the first camera
    Camera standard_camera = cameras[0];
    const size_t bytes_in_img = standard_camera.num_pixels * sizeof(float);

    // allocate image space on host
    clock_t img_alloc_start = clock();
    float *img = (float*) malloc(bytes_in_img);
    if (verbose) {
        float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image              (host)              %.6fs\n",img_alloc_dur);
    }

    // initialise image space on device 
    clock_t d_img_alloc_start = clock();
    float *d_img = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
    if (verbose) {
        float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image              (device)            %.6fs\n",d_img_alloc_dur);
    }

    // import npy data to host
    std::vector<MeshBlockInfo> all_mb_info;
    float *h_all_data = nullptr;
    size_t h_bytes = 0;
    if (labelled_data) {
        all_mb_info = load_labelled_meshblocks(input_str, h_all_data, h_bytes, relativistic, verbose);
    } else {
        all_mb_info = load_unlabelled_meshblock(input_str, h_all_data, h_bytes, relativistic, verbose);
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

    // define render shape    
    int tx = 16, ty = 16; // must not exceed 1024 (max thread per block)
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)standard_camera.num_pixels_X / tx), 
                                std::ceil((float)standard_camera.num_pixels_Y / ty));

    // declare output container
    npy::npy_data_ptr<float> npy_img;
    npy_img.shape = {(unsigned long)standard_camera.num_pixels_X, (unsigned long)standard_camera.num_pixels_Y};

    // iterate over cameras
    int img_count = 0;
    int total_images = cameras.size();
    size_t num_zero_pad = 5;
    if (verbose) {
        std::cout << "=============================================================\n";
        if (total_images == 1) {
            std::cout << "Starting render for single image...\n";
        } else {
            std::cout << "Starting render cycle for " << total_images << " images...\n";
        }
        std::cout << "-------------------------------------------------------------\n";
    }
    for (auto &camera : cameras) {
        
        clock_t this_img_start = clock();

        // call render
        clock_t render_start = clock();
        render_from_mesh<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mesh, relativistic, doppler_index);
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
        std::string save_str = save_str_header + zero_pad_str(img_count, num_zero_pad) + ".npy";
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
                // given matching shape, add new data to existing TODO: shape information appears corrupted, chase
                std::vector<float> img_vec = existing_npy_data.data;
                std::vector<float> new_img_vec;
                new_img_vec.insert(new_img_vec.end(), img, img + standard_camera.num_pixels );
                // std::vector<float> new_img_vec(img, img + sizeof(img) / sizeof(float));
                //new_img_vec.assign(img, img + standard_camera.num_pixels); 
                std::transform(img_vec.begin(), img_vec.end(), new_img_vec.begin(), img_vec.begin(), std::plus<float>());
                npy_img.data_ptr = img_vec.data();
            } else { // no existing file, direct write
                npy_img.data_ptr = img;
            }
        } else {
            // generate new data
            npy_img.data_ptr = img;
        }
        npy::write_npy(save_str, npy_img);
        if (verbose) {
            float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
            printf("write raw image           (host->npy)         %.6fs\n",npy_write_dur);
            float this_img_dur = (float)(clock() - this_img_start)/CLOCKS_PER_SEC;
            printf("img total                 (host/device)       %.6fs\n",this_img_dur);
            if (img_count == total_images - 1) {
                std::cout << "=============================================================\n";
            } else {
                std::cout << "-------------------------------------------------------------\n";
            }
        }

        // prepare for next image
        if (total_images != 1) { // skip wipe if single image output
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

    // terminate
    if (verbose) {
        float main_dur = (float)(clock() - main_start)/CLOCKS_PER_SEC;
        printf("total runtime                                 %.6fs\n",main_dur);
        std::cout << "=============================================================\n";
        printf("cuDART terminated.\n");
    }

    return 0;
}