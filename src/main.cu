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

__global__ void wipe_img(Camera camera, float *img) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; 
    img[pixel_index] = 0; // reset image
}

__global__ void render_from_mesh(Camera camera, float *img, Mesh **mesh) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; 

    // initialise ray
    vec3 pixel_origin = camera.calc_pixel_origin(i, j);
    Ray pixel_ray(pixel_origin, camera.normal);
    
    // calculate pixel value from MeshBlock data
    img[pixel_index] += (*mesh)->calc_trace(pixel_ray);
}

int main(int argc, char *argv[]) {

    // start general timer
    clock_t main_start = clock();

    // define space for user settings
    std::string cudart_version = "version 0.5 - January 2026";
    char *input_char = nullptr, *save_char = nullptr, *camera_char = nullptr, *mem_char = nullptr;
    bool verbose = false;

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
                default:
                    if ((i+1 >= argc) || (*argv[i+1] == '-')) {
                        std::stringstream err_msg;
                        err_msg << "### FATAL ERROR in main ###\n";
                        err_msg << "-" << opt_letter << "must be follower by a valid argument\n";
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
                case 'v':
                    verbose = true;
                    break;
                case 'c':
                    camera_char = argv[++i];
                    break;
                case 'h':
                default:
                    std::cout << "cuDART " << cudart_version << std::endl;
                    std::cout << "Usage: " << argv[0] << " [options]\n";
                    std::cout << "Options:\n";
                    std::cout << " -i <file>    specify input file [.npy]\n";
                    std::cout << " -s <file>    specify render save file [.ppm]\n";
                    std::cout << " -c <file>    specify camera data file [.txt]\n";
                    std::cout << " -m <value>   max VRAM in GB\n";
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
    std::vector<Camera> cameras = {};
    load_cameras(cameras, camera_char, verbose);
    
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
    std::vector<MeshBlockInfo> all_mb_info = {};
    float *h_all_data = nullptr;
    size_t h_bytes = 0;
    if (labelled_data) {
        load_labelled_meshblock(input_str, all_mb_info, h_all_data, h_bytes, verbose);
    } else {
        load_unlabelled_meshblocks(input_str, all_mb_info, h_all_data, h_bytes, verbose);
    }
    int num_meshblocks = all_mb_info.size();

    // determine VRAM limitations
    float vram_limit_f = 1e12;
    if (mem_char != nullptr) {
        vram_limit_f = static_cast<float>(std::atof(mem_char)) * 1e9;
    }
    size_t free_t, total_t;
    float vram_tolerance = 0.9; // undercut free by this tolerance
    checkCudaErrors(cudaMemGetInfo(&free_t,&total_t));
    float vram_free_f = static_cast<float>(free_t) * vram_tolerance;
    float vram_avail_f = std::min(vram_free_f, vram_limit_f);
    size_t d_bytes_avail = static_cast<size_t>(vram_avail_f);
            
    // handle memory request excess
    bool d_mem_excess = (d_bytes_avail < h_bytes);
    size_t d_bytes;
    if (d_mem_excess) {
        std::stringstream err_msg;
        err_msg << "Total input memory exceeds VRAM, partitioning currently unsupported\n";
        CUDART_ERROR(err_msg);
    } else {
        d_bytes = h_bytes; // allocate entire dataset to device
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
    build_containers(all_mb_info, mb_list, mesh, verbose);

    // clock_t mb_alloc_start = clock();
    // MeshBlock **mb_list;
    // checkCudaErrors(cudaMalloc((void **)&mb_list, num_meshblocks * sizeof(MeshBlock *)));

    // // initialise MeshBlocks on device
    // int mem_start = 0;
    // for (int n = 0; n < num_meshblocks; n++) {
    //     vec3 xl = all_mb_info[n].xl;
    //     vec3 xr = all_mb_info[n].xr;
    //     vec3 mb_dims = all_mb_info[n].mb_dims;
    //     init_meshblock<<<1,1>>>(mb_list, n, xl, xr, mb_dims, d_data, mem_start);
    //     checkCudaErrors(cudaPeekAtLastError());
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     mem_start += all_mb_info[n].mb_size;
    // }

    // // initialise Mesh on device
    // Mesh **mesh;
    // checkCudaErrors(cudaMalloc((void **)&mesh, sizeof(Mesh * ))); 
    // init_mesh<<<1,1>>>(mesh, mb_list, num_meshblocks);
    // checkCudaErrors(cudaPeekAtLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    
    // if (verbose) {
    //     float mb_alloc_dur = (float)(clock() - mb_alloc_start)/CLOCKS_PER_SEC;
    //     printf("malloc/init containers    (device)            %.6fs\n",mb_alloc_dur);
    // }

    // define render shape    
    int tx = 32, ty = 32; // must not exceed 1024 (max thread per block)
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)standard_camera.num_pixels_X / tx), 
                                std::ceil((float)standard_camera.num_pixels_Y / ty));

    // iterate over cameras
    int img_count = 0;
    int total_images = cameras.size();
    size_t num_zero_pad = 3;
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
        render_from_mesh<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mesh);
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
        std::string save_str(save_char);
        std::string num_str = std::to_string(img_count);
        auto padded_num_str = std::string(num_zero_pad - std::min(num_zero_pad, num_str.length()), '0') + num_str;
        save_str = save_str + padded_num_str + ".npy";
        npy::npy_data_ptr<float> npy_img;
        npy_img.data_ptr = img;
        npy_img.shape = {(unsigned long)standard_camera.num_pixels_X, (unsigned long)standard_camera.num_pixels_Y};
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