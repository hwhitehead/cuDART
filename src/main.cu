// standard library imports
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>

// custom external library imports
#include "npy.hpp"

// custom local library imports 
#include "vec3.hpp"
#include "ray.hpp"
#include "meshblock.hpp"
#include "tools.hpp"

int main(int argc, char *argv[]) {
    
    // start general timer
    clock_t main_start = clock();

    // parse user input
    std::string cudart_version = "version 0.1 - January 2026";
    
    // filenames
    char *input_char = nullptr, *save_char = nullptr;
    
    // default camera settings
    int num_pixels_X = 100, num_pixels_Y = 100; 
    float R_pos = 1.0, theta_pos = 0.5 * M_PI, phi_pos = 0.0;
    float length_X = 1.0, length_Y = 1.0, tilt = 0.0;
    ve3 bias(0.0, 0.0, 1.0);

    // flags
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        // check if argv[i] is a 2 character string of form "-X"
        if (*argv[i] == "-" && *(argv[i]+1) != "\0" && *(argv[i]+2) == "\0") {
            // check command line arguments
            char opt_letter = *(argv[i]+1);
            switch (opt_letter) { // parse options without arguments
                case "h":
                case "v":
                case "t":
                    break;
                default:
                    if ((i+1 >= argc) || (*argv[i+1] == "-")) { 
                        std::cout << "### FATAL ERROR in main" << std::endl
                                << "-" << opt_letter << "must be follower by a valid argument\n";
                    }
            } // end cases
            switch (*(argv[i]+1)) { //
                case "i":
                    input_filename = argv[++i];
                    break;
                case "s":
                    save_filename = argv[++i];
                    break;
                case "v":
                    verbose = true;
                case "h":
                default:
                    std::cout << "cuDART " << cudart_version << std::endl;
                    std::cout << "Usage: " << argv[0] << " [options]\n";
                    std::cout << "Options:\n";
                    std::cout << " -i <file>    specify input file [.npy]\n";
                    std::cout << " -s <file>    specify render save file [.ppm]\n";
                    std::cout << " -v           verbosity flag\n";
                    std::cout << " -h           this help message\n";   
            } // end cases
        } // end 2 char check
    }

    if (input_filename == nullptr && save_filename == nullptr) {
        std::cout << "### FATAL ERROR in main" << std::endl;
                << "No input file or output file specified." << std::endl;
    }

    if (verbose) std::cout << "Starting cuDART...\n";

    // load npy data as specified by user
    //const std::string npy_path {"simdata/sn_low.npy"}; // old
    clock_t npy_read_start = clock();
    const std::string input_str(input_char);
    npy::npy_data d = npy::read_npy<float>(npy_path);
    std::vector<float> npy_data = d.data; // TODO: check speedup with cudaMallocHost pre-trasnfer
    std::vector<unsigned long> npy_shape = d.shape;
    vec3 mb_dims((float)npy_shape[0], (float)npy_shape[1], (float)npy_shape[2]);
    int data_size = npy_data.size();
    float *data = npy_data.data();
    size_t bytes_in_data = data_size * sizeof(float);
    if (verbose) {
        float npy_read_dur = (float)(clock() - npy_read_start)/CLOCKS_PER_SEC;
        printf("Loaded npy data to host in %.6fs\n",npy_read_dur);
    }

    // allocate device memory
    float *d_data = nullptr;
    clock_t d_data_alloc_start = clock();
    checkCudaErrors(cudaMalloc(&d_data, bytes_in_data));
    if (verbose) {
        float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
        printf("Allocated memory for data on device in %.6fs\n",d_data_alloc_dur);
    }
    
    // copy data into device memory
    clock_t data_copy_start = clock();
    checkCudaErrors(cudaMemcpy(d_data, data, bytes_in_data, cudaMemcpyHostToDevice));
    if (verbose) {
        float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
        printf("Copied data to device in %.6fs\n",data_copy_dur);
    }

    // initialise MeshBlock
    vec3 xl(0.0, 0.0, 0.0); // TODO: add shape-sentive domain definition
    vec3 xr(1.0, 1.0, 1.0);
    MeshBlock **mb = nullptr;
    clock_t mb_alloc_start = clock();
    checkCudaErrors(cudaMalloc(&mb, sizeof(MeshBlock *))); // locator of MeshBlock memory position
    init_meshblock<<<1,1>>>(mb, xl, xr, mb_dims, d_data);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    if (verbose) {
        float mb_alloc_dur = (float)(clock() - mb_alloc_start)/CLOCKS_PER_SEC;
        printf("Allocated space for MeshBlock on device in %.6fs\n",mb_alloc_dur);
    }

    // initialise camera settings as specified by user
    Camera camera;
    // decide on format for user input 
    camera.update_camera();
    if (verbose) std::cout << "Positioned camera.\n";

    // initialise image space on device
    clock_t d_img_alloc_start = clock();
    const size_t bytes_in_img = camera.num_pixels * sizeof(float);
    float *d_img = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
    if (verbose) {
        float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
        printf("Allocated space for (%d,%d) image on device in %.6fs\n",camera.num_pixels_X, camera.num_pixels_Y, d_img_alloc_dur);
    }

    // call render
    clock_t render_start = clock();
    const dim3 threads_per_block(32,32); // must not exceed 1024 (max thread per block)
    const dim3 blocks_per_grid(std::ceil(camera.num_pixels_X/32), 
                                std::ceil(camera.num_pixels_Y/32));
    if (verbose) std::cout << "Starting render...\n";
    camera.render_img<<<blocks_per_grid,threads_per_block>>>(d_img, mb); // how to run this as kernel?
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    if (verbose) {
        float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
        printf("Render kernel completed in %.6fs\n",render_dur);
    }

    // allocate image space on host
    clock_t img_alloc_start = clock();
    float *img = (float*) malloc(bytes_in_img);
    if (verbose) {
        float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
        printf("Allocated space for image on host in %.6fs\n",img_alloc_dur);
    }

    // copy image data to host
    clock_t img_copy_start = clock();
    checkCudaErrors(cudaMemcpy(img, d_img, bytes_in_img, cudaMemcpyDeviceToHost));
    if (verbose) {
        float img_copy_dur = (float)(clock() - img_copy_start)/CLOCKS_PER_SEC;
        printf("Copied image to host in %.6fs\n",img_copy_dur);
    }

    // save data
    clock_t npy_write_start = clock();
    const std::string save_str(save_char);
    npy::npy_data_ptr<float> npy_img;
    npy_img.data_ptr = img;
    npy_img.shape = {camera.num_pixels_Y, camera.num_pixels_X};
    npy::write_npy(save_str, npy_img);
    if (verbose) {
        float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
        printf("Saved image data to npy in %.6fs\n",npy_write_dur);
    }

    // perform cleanup of device/host data
    checkCudaErrors(cudaFree(img));
    checkCudaErrors(cudaFree(mb));
    checkCudaErrors(cudaFree(data));
    free(img);
    free(data);

    // terminate
    if (verbose) {
        float main_dur = (float)(clock() - main_start)/CLOCKS_PER_SEC;
        printf("cuDART terminating after %.6fs\n",main_dur);
    }

    return 0;
}