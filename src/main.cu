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
#include "camera.hpp"

__global__ void render_img(Camera camera, float *img, MeshBlock **mb) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; // well behaved

    // initialise ray
    vec3 pixel_origin = camera.calc_pixel_origin(i, j);
    Ray pixel_ray(pixel_origin, camera.normal);
    
    // calculate pixel value from MeshBlock data
    img[pixel_index] = (*mb)->calc_trace(pixel_ray);
}

int main(int argc, char *argv[]) {

    // start general timer
    clock_t main_start = clock();

    // define space for user settings
    std::string cudart_version = "version 0.2 - January 2026";
    char *input_char = nullptr, *save_char = nullptr, char *camera_char = nullptr;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        // check if argv[i] is a 2 character string of form "-X"
        if (*argv[i] == '-' && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {
            // check command line arguments
            char opt_letter = *(argv[i]+1);
            switch (opt_letter) { // parse options without arguments
                case 'h':
                case 'v':
                default:
                    if ((i+1 >= argc) || (*argv[i+1] == '-')) { 
                        std::cout << "### FATAL ERROR in main" << std::endl
                                << "-" << opt_letter << "must be follower by a valid argument\n";
                    }
            } // end cases
            switch (*(argv[i]+1)) { //
                case 'i':
                    input_char = argv[++i];
                    break;
                case 's':
                    save_char = argv[++i];
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
                    std::cout << " -v           verbosity flag\n";
                    std::cout << " -h           this help message\n"; 
                    return 0; 
            } // end cases
        } // end 2 char check
    }

    if (input_char == nullptr && save_char == nullptr) {
        std::cout << "### FATAL ERROR in main\n";
        std::cout << "No input file or output file specified.\n";
        return 0;
    }

    if (camera_char == nullptr && verbose) {
        std::cout << "No user specified camera input, falling back to default.\n";
    } else { // determine number of camera locations
        std::string camera_str(camera_char);
        std::ifstream camera_file(camera_str);
        camera_file.unsetf(std::ios_base::skipws);
        int line_count = std::count(std::istream_iterator<char>(camera_file),
                                        std::istream_iterator<char>(),'\n');
        std::cout << "lines in file = " << line_count << std::endl;
    }

    if (verbose) {
        std::cout << "Starting cuDART (verbose)...\n";
        std::cout << "----------------------------------------------------------\n";
        std::cout << "|      Activity        |    Location    |    Duration    |\n";
        std::cout << "----------------------------------------------------------\n";
    }
    // load npy data as specified by user
    //const std::string npy_path {"simdata/sn_low.npy"}; // old
    clock_t npy_read_start = clock();
    const std::string input_str(input_char);
    npy::npy_data d = npy::read_npy<float>(input_str);
    std::vector<float> npy_data = d.data; // TODO: check speedup with cudaMallocHost pre-trasnfer
    std::vector<unsigned long> npy_shape = d.shape;
    vec3 mb_dims((float)npy_shape[0], (float)npy_shape[1], (float)npy_shape[2]);
    int data_size = npy_data.size();
    float *data = npy_data.data();
    size_t bytes_in_data = data_size * sizeof(float);
    if (verbose) {
        float npy_read_dur = (float)(clock() - npy_read_start)/CLOCKS_PER_SEC;
        printf("read/malloc data        (npy->host)         %.6fs\n",npy_read_dur);
    }

    // allocate device memory
    float *d_data = nullptr;
    clock_t d_data_alloc_start = clock();
    checkCudaErrors(cudaMalloc(&d_data, bytes_in_data));
    if (verbose) {
        float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc data             (device)            %.6fs\n",d_data_alloc_dur); // align these prints?
    }
    
    // copy data into device memory
    clock_t data_copy_start = clock();
    checkCudaErrors(cudaMemcpy(d_data, data, bytes_in_data, cudaMemcpyHostToDevice));
    if (verbose) {
        float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
        printf("memcpy data             (host->device)      %.6fs\n",data_copy_dur);
    }

    // initialise MeshBlock
    vec3 xl(-0.5, -0.5, -0.5); // TODO: add shape-sentive domain definition
    vec3 xr(0.5, 0.5, 0.5);
    MeshBlock **mb = nullptr;
    clock_t mb_alloc_start = clock();
    checkCudaErrors(cudaMalloc(&mb, sizeof(MeshBlock *))); // locator of MeshBlock memory position
    init_meshblock<<<1,1>>>(mb, xl, xr, mb_dims, d_data);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    if (verbose) {
        float mb_alloc_dur = (float)(clock() - mb_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc/init MeshBlock   (device)            %.6fs\n",mb_alloc_dur);
    }

    // initialise camera settings as specified by user
    Camera camera;
    camera.num_pixels_X = 2048;
    camera.num_pixels_Y = 2048;
    camera.theta_pos = (75.0 / 180) * M_PI;
    camera.phi_pos = (180.0 / 180) * M_PI;
    camera.tilt = (-38.0 / 180) * M_PI;
    camera.update_camera();

    // initialise image space on device
    clock_t d_img_alloc_start = clock();
    const size_t bytes_in_img = camera.num_pixels * sizeof(float);
    float *d_img = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
    if (verbose) {
        float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image            (device)            %.6fs\n",d_img_alloc_dur);
    }

    // call render
    clock_t render_start = clock();
    int tx = 32, ty = 32; // must not exceed 1024 (max thread per block)
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)camera.num_pixels_X / tx), 
                                std::ceil((float)camera.num_pixels_Y / ty));
    render_img<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mb);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    if (verbose) {
        float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
        printf("render kernel           (device)            %.6fs\n",render_dur);
    }

    // allocate image space on host
    clock_t img_alloc_start = clock();
    float *img = (float*) malloc(bytes_in_img);
    if (verbose) {
        float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image            (host)              %.6fs\n",img_alloc_dur);
    }

    // copy image data to host
    clock_t img_copy_start = clock();
    checkCudaErrors(cudaMemcpy(img, d_img, bytes_in_img, cudaMemcpyDeviceToHost));
    if (verbose) {
        float img_copy_dur = (float)(clock() - img_copy_start)/CLOCKS_PER_SEC;
        printf("memcpy image            (device->host)      %.6fs\n",img_copy_dur);
    }

    // save data
    clock_t npy_write_start = clock();
    const std::string save_str(save_char);
    npy::npy_data_ptr<float> npy_img;
    npy_img.data_ptr = img;
    npy_img.shape = {(unsigned long)camera.num_pixels_X, (unsigned long)camera.num_pixels_Y};
    npy::write_npy(save_str, npy_img);
    if (verbose) {
        float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
        printf("write data              (host->npy)         %.6fs\n",npy_write_dur);
    }

    // perform cleanup of device/host data
    clock_t free_start = clock();
    free_meshblock<<<1,1>>>(mb);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(mb));
    checkCudaErrors(cudaFree(d_data));
    free(img);
    free(data);
    cudaDeviceReset();
    if (verbose) {
        float free_dur = (float)(clock() - free_start)/CLOCKS_PER_SEC;
        printf("free all                (device/host)       %.6fs\n",free_dur);
    }

    // terminate
    if (verbose) {
        float main_dur = (float)(clock() - main_start)/CLOCKS_PER_SEC;
        printf("total runtime                               %.6fs\n",main_dur);
        std::cout << "----------------------------------------------------------\n";
        printf("cuDART terminated.\n");
    }

    return 0;
}