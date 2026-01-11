// standard library imports
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>

// custom external library imports
#include "npy.hpp"

// custom local library imports 
#include "vec3.hpp"
#include "ray.hpp"
#include "meshblock.hpp"
#include "tools.hpp"

int main(int argc, char *argv[]) {
    // parse user input
    std::string cudart_version = "version 0.1 - January 2026";
    // filenames
    char *input_filename = nullptr, *save_filename = nullptr;
    // default camera settings
    int num_pixels_x = 100, num_pixels_y = 100; 
    float R_pos = 1.0, theta_pos = 0.5 * M_PI, phi_pos = 0.0;
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

    // initialise camera settings as specified by user


    // initialise image space
    const size_t img_size = cam.num_pixels * sizeof(float);
    float *img;
    checkCudaErrors(cudaMallocManaged((void **)&img, img_size));
    if (verbose) std::cout << "initialised image space.\n";

    // load npy data as specified by user
    const std::string npy_path {"simdata/sn_low.npy"};
    npy::npy_data d = npy::read_npy<float>(npy_path);
    std::vector<float> npy_data = d.data;
    std::vector<unsigned long> npy_shape = d.shape;
    vec3 dims((float)npy_shape[0], (float)npy_shape[1], (float)npy_shape[2]);
    int data_size = npy_data.size();
    float *p_data = npy_data.data();
    size_t bytes = data_size * sizeof(float);
    if (verbose) std::cout << "finished load from npy input.\n";

    // allocate device memory
    float *data;
    checkCudaErrors(cudaMallocManaged(&data, bytes));
    if (verbose) std::cout << "finished data alloc on device.\n";
    
    // copy data into device memory
    checkCudaErrors(cudaMemcpy(data, p_data, bytes, cudaMemcpyHostToDevice));
    if (verbose) std::cout << "finished data copy to device.\n";

    // initialise MeshBlock
    int thr_per_blk = 32;
    int blk_in_grid = 64;
    vec3 xl(0.0, 0.0, 0.0);
    vec3 xr(1.0, 1.0, 1.0);
    MeshBlock **mb;
    checkCudaErrors(cudaMalloc(&mb, sizeof(MeshBlock *)));
    init_meshblock<<<thr_per_blk,blk_in_grid>>>(mb, xl, xr, dims, data); // run on single?
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    if (verbose) std::cout << "finished meshblock init on device.\n";
    
    // call render
    if (verbose) std::cout << "starting render...\n";
    cam.render_img(fb, mb)
    if (verbose) std::cout << "finished render.\n";

    // perform cleanup
    checkCudaErrors(cudaFree(img));
    checkCudaErrors(cudaFree(mb));
    checkCudaErrors(cudaFree(data));
    if (verbose) std::cout << "finished cleanup, terminating.\n";


    return 0;
}