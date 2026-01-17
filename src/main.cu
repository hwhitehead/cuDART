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

// custom external library imports
#include "npy.hpp"

// custom local library imports 
#include "vec3.hpp"
#include "ray.hpp"
#include "meshblock.hpp"
#include "tools.hpp"
#include "camera.hpp"
#include "dev_array.hpp"

__global__ void render_img(Camera camera, float *img, MeshBlock **mb) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; 

    // initialise ray
    vec3 pixel_origin = camera.calc_pixel_origin(i, j);
    Ray pixel_ray(pixel_origin, camera.normal);
    
    // calculate pixel value from MeshBlock data
    img[pixel_index] = (*mb)->calc_trace(pixel_ray);
}

__global__ void render_partitioned_img(Camera camera, float *img, MeshBlock **mb, int il, int ir) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; 

    // initialise ray
    vec3 pixel_origin = camera.calc_pixel_origin(i, j);
    Ray pixel_ray(pixel_origin, camera.normal);
    
    // calculate pixel value from MeshBlock data
    img[pixel_index] += (*mb)->calc_partitioned_trace(pixel_ray, il, ir);
}

__global__ void wipe_img(Camera camera, float *img) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; 
    img[pixel_index] = 0; // reset image
}

int main(int argc, char *argv[]) {

    // start general timer
    clock_t main_start = clock();

    // define space for user settings
    std::string cudart_version = "version 0.4 - January 2026";
    char *input_char = nullptr, *save_char = nullptr, *camera_char = nullptr, *mem_char = nullptr;
    bool verbose = false;

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
                        std::cout << "### FATAL ERROR in main ###" << std::endl
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

    if (input_char == nullptr && save_char == nullptr) {
        std::cout << "### FATAL ERROR in main\n";
        std::cout << "No input file or output file specified.\n";
        return 0;
    }

    std::vector<Camera> cameras = {};
    if (camera_char == nullptr && verbose) {
        std::cout << "No user specified camera input, falling back to default.\n";
        Camera default_camera;
        cameras.push_back(default_camera);
    } else { // determine number of camera locations
        std::string camera_str(camera_char);
        std::ifstream camera_file(camera_str);
        int line_count = 0, num_pixels_X, num_pixels_Y;
        if (camera_file.is_open()) {
            std::string line;
            float a, b, c, d, e, f;
            while (std::getline(camera_file, line)) {
                std::istringstream iss(line);
                if (!(iss >> a >> b >> c >> d >> e >> f)) {
                    std::cout << "### FATAL ERROR in main ###\n";
                    std::cout << "Unable to parse line " << line_count << "of camera file at " << camera_str << std::endl;
                    return 0;
                } else {
                    // read line by line
                    if (line_count == 0) { // read static header
                        num_pixels_X = a;
                        num_pixels_Y = b;
                    } else { // read dynamic camera data
                        Camera this_camera;
                        this_camera.num_pixels_X = num_pixels_X;
                        this_camera.num_pixels_Y = num_pixels_Y;
                        this_camera.R_pos = a;
                        this_camera.theta_pos = b;
                        this_camera.phi_pos = c;
                        this_camera.tilt = d;
                        this_camera.length_X = e;
                        this_camera.length_Y = f;
                        this_camera.update_camera();
                        cameras.push_back(this_camera);
                    }
                }
                line_count++;
            }
            camera_file.close();
        } else {
            std::cout << "### FATAL ERROR in main ###\n";
            std::cout << "Unable to open camera file at " << camera_str << std::endl;
            return 0;
        }
    }

    if (verbose) {
        std::cout << "Starting cuDART (verbose)...\n";
        std::cout << "==========================================================\n";
        std::cout << "|      Activity        |    Location    |    Duration    |\n";
        std::cout << "==========================================================\n";
    }
    // load npy data as specified by user
    clock_t npy_read_start = clock();
    const std::string input_str(input_char);
    npy::npy_data d = npy::read_npy<float>(input_str);
    std::vector<float> npy_data = d.data; // TODO: check speedup with cudaMallocHost pre-transfer (seems unhelpful)
    std::vector<unsigned long> npy_shape = d.shape;
    vec3 mb_dims((float)npy_shape[0], (float)npy_shape[1], (float)npy_shape[2]);
    int data_size = npy_data.size();
    float *data = npy_data.data();
    size_t bytes_in_data = data_size * sizeof(float);
    if (verbose) {
        float npy_read_dur = (float)(clock() - npy_read_start)/CLOCKS_PER_SEC;
        printf("read/malloc data        (npy->host)         %.6fs\n",npy_read_dur);
    }

    // load image dimensions from the first camera
    Camera camera = cameras[0];
    const size_t bytes_in_img = camera.num_pixels * sizeof(float);

    // allocate image space on host
    clock_t img_alloc_start = clock();
    float *img = (float*) malloc(bytes_in_img);
    if (verbose) {
        float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image            (host)              %.6fs\n",img_alloc_dur);
    }

    // initialise image space on device TODO: identify insane overhead?? O(5)s
    clock_t d_img_alloc_start = clock();
    float *d_img = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
    if (verbose) {
        float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc img              (device)            %.6fs\n",d_img_alloc_dur);
    }

    // check dimensions of GPU/user VRAM maximum
    size_t free_t, total_t;
    checkCudaErrors(cudaMemGetInfo(&free_t,&total_t));
    float free_f = static_cast<float>(free_t);
    float avail_mem = free_f;
    if (not (mem_char == nullptr)) {
        avail_mem = std::min(static_cast<float>(std::atof(mem_char) * 1e9), avail_mem); // convert GB to B
    }

    // define data memory on device
    int num_divisions = 1;
    float *d_data = nullptr;
    size_t bytes_on_device = bytes_in_data;
    if (bytes_in_data > avail_mem) { // data must be partioned to fit device
        float mem_ratio = bytes_in_data / avail_mem;
        num_divisions = std::ceil(mem_ratio);
        bytes_on_device = bytes_in_data / num_divisions;
    } 

    clock_t d_data_alloc_start = clock();
    checkCudaErrors(cudaMalloc(&d_data, bytes_on_device));
    if (verbose) {
        float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc data             (device)            %.6fs\n",d_data_alloc_dur);
    }

    // initialise MeshBlock on device
    vec3 xl(-0.5, -0.5, -0.5); // TODO: add shape-sensitive domain definition <-----
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

    // define render shape    
    int tx = 32, ty = 32; // must not exceed 1024 (max thread per block)
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)camera.num_pixels_X / tx), 
                                std::ceil((float)camera.num_pixels_Y / ty));
    int floats_on_device = bytes_on_device / sizeof(float);

    // iterate over cameras
    int img_count = 0;
    size_t num_zeros = 3;
    if (verbose) {
        std::cout << "----------------------------------------------------------\n";
        float f_avail = free_f / 1e9;
        float f_req = bytes_in_data / 1e9;
        if (not (mem_char == nullptr)) {
            float f_limit = static_cast<float>(std::atof(mem_char));
            printf("VRAM: Avail = %.2fGB, Limit = %.2fGB, Required = %.2fGB\n",f_avail,f_limit,f_req);
        } else {
            printf("VRAM: Avail = %.2fGB, Limit = None, Required = %.2fGB\n",f_avail,f_req);
        }
        
        if (num_divisions == 1) {
            std::cout << "Data fits within available VRAM, launching one partition.\n";
        } else {
            std::cout << "Data exceeds available VRAM, launching " << num_divisions << " partitions.\n";
        }
        if (verbose) std::cout << "==========================================================\n";
    }
    std::cout << "data size = " << data_size << std::endl;
    std::cout << "bytes_on_device = " << bytes_on_device << std::endl;
    std::cout << "floats_on_device = " << floats_on_device << std::endl;
    for (auto &camera : cameras) {
        
        clock_t this_img_start = clock();

        // TODO: escape single partition run from loop to bypass wipe overhead

        // iterate over partitions
        for (int n = 0; n < num_divisions; n++) {
            // wipe image
            wipe_img<<<blocks_per_grid,threads_per_block>>>(camera, d_img);
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            int il = n * floats_on_device;
            int ir = il + floats_on_device;
            if (ir > data_size) ir = data_size; // prevent overrun
            size_t bytes_in_sub = (ir - il) * sizeof(float);
            std::cout << "(il,ir) = (" << il << "," << ir << ")\n";
            size_t sublen_in_bytes = (ir - il) * sizeof(float);
            std::cout << "sublen_in_bytes = " << sublen_in_bytes << std::endl;

            // copy subsection of data into device memory
            clock_t data_copy_start = clock();
            float* data_addr = data + il * sizeof(float);
            checkCudaErrors(cudaMemcpy(d_data, &data[il], bytes_in_sub, cudaMemcpyHostToDevice)); // no fail, but missing domain?
            checkCudaErrors(cudaPeekAtLastError());
            if (verbose) {
                float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
                printf("memcpy data             (host->device)      %.6fs\n",data_copy_dur);
            }

            // call render
            clock_t render_start = clock();
            render_partitioned_img<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mb, il, ir);
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            if (verbose) {
                float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
                printf("render kernel           (device)            %.6fs\n",render_dur);
            }
            if (verbose) std::cout << "..........................................................\n";
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
        std::string save_str(save_char);
        std::string num_str = std::to_string(img_count);
        auto padded_num_str = std::string(num_zeros - std::min(num_zeros, num_str.length()), '0') + num_str;
        save_str = save_str + padded_num_str + ".npy";
        npy::npy_data_ptr<float> npy_img;
        npy_img.data_ptr = img;
        npy_img.shape = {(unsigned long)camera.num_pixels_X, (unsigned long)camera.num_pixels_Y};
        npy::write_npy(save_str, npy_img);
        if (verbose) {
            float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
            printf("write data              (host->npy)         %.6fs\n",npy_write_dur);
            float this_img_dur = (float)(clock() - this_img_start)/CLOCKS_PER_SEC;
            printf("img total               (host/device)       %.6fs\n",this_img_dur);
            if (verbose) std::cout << "==========================================================\n";
        }
        img_count++;
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
        std::cout << "==========================================================\n";
        printf("cuDART terminated.\n");
    }

    return 0;
}