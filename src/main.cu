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
    size_t num_zeros = 3;

    // define space for user settings
    std::string cudart_version = "version 0.5 - January 2026";
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
        std::cout << "=============================================================\n";
        std::cout << "|      Activity        |    Location    |      Duration     |\n";
        std::cout << "=============================================================\n";
    }

    // load image dimensions from the first camera
    Camera camera = cameras[0];
    const size_t bytes_in_img = camera.num_pixels * sizeof(float);

    // allocate image space on host
    clock_t img_alloc_start = clock();
    float *img = (float*) malloc(bytes_in_img);
    if (verbose) {
        float img_alloc_dur = (float)(clock() - img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image              (host)              %.6fs\n",img_alloc_dur);
    }

    // initialise image space on device TODO: identify insane overhead?? O(5)s
    clock_t d_img_alloc_start = clock();
    float *d_img = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_img, bytes_in_img));
    if (verbose) {
        float d_img_alloc_dur = (float)(clock() - d_img_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc image              (device)            %.6fs\n",d_img_alloc_dur);
    }

    // read memory info from header
    std::string data_dir(input_char);
    std::string header_str = data_dir + "/header.txt";
    std::ifstream header_file(header_str);
    std::cout << header_str << std::endl;
    std::vector<MeshBlockInfo> all_mb_info = {};
    int total_float_count = 0;
    if (header_file.is_open()) {
        std::string line;
        int line_count = 0;
        int mb_size, nx, ny, nz;
        float xl, yl, zl, xr, yr, zr;
        while (std::getline(header_file, line)) {
            std::istringstream iss(line);
            if (!(iss >> mb_size >> nx >> ny >> nz >> xl >> yl >> zl >> xr >> yr >> zr)) {
                std::stringstream err_msg;
                err_msg << "### FATAL ERROR in main###\n";
                err_msg << "Unable to parse line " << line_count << " of header file at " << header_str << std::endl;
                CUDART_ERROR(err_msg);
            } else {
                MeshBlockInfo mb_info;
                mb_info.mb_size = mb_size;
                mb_info.xl = vec3(xl,yl,zl);
                mb_info.xr = vec3(xr,yr,zr);
                mb_info.mb_dims = vec3(nx,ny,nz);
                all_mb_info.push_back(mb_info);
                total_float_count += mb_size;
            }
            line_count++;
        }
    }

    // determine total memory requirements
    size_t bytes_in_data = sizeof(float) * total_float_count;

    // check dimensions of GPU/user VRAM maximum
    size_t free_t, total_t;
    float tolerance = 0.9; // undercut avail by this tolerance
    checkCudaErrors(cudaMemGetInfo(&free_t,&total_t));
    float free_f = static_cast<float>(free_t) * tolerance;
    float avail_mem = free_f;
    float f_limit;
    if (not (mem_char == nullptr)) {
        f_limit = static_cast<float>(std::atof(mem_char)) * 1e9;
        avail_mem = std::min(f_limit, avail_mem); // convert GB to B
    }
    size_t bytes_avail = static_cast<size_t>(avail_mem); // check typing here
            
    // define clustering 
    bool run_clustering = (bytes_avail < bytes_in_data);
    size_t bytes_on_device;
    if (run_clustering) {
        bytes_on_device = bytes_avail; // allocate available data
    } else {
        bytes_on_device = bytes_in_data; // allocate entire dataset
    }

    // allocate space on device
    clock_t d_data_alloc_start = clock();
    float *d_data = nullptr;
    checkCudaErrors(cudaMalloc(&d_data, bytes_on_device));
    if (verbose) {
        float d_data_alloc_dur = (float)(clock() - d_data_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc data               (device)            %.6fs\n",d_data_alloc_dur);
    }

    if (run_clustering) {
        std::stringstream err_msg;
        err_msg << "clustering currently unsupported\n";
        CUDART_ERROR(err_msg);
    }

    // assert no clustering for current build

    // load ALL npy data into host memory
    clock_t npy_read_start = clock();
    int num_meshblocks = all_mb_info.size();
    float *h_all_data = (float*) malloc(bytes_in_data);
    int mem_offset = 0;
    for (int n = 0; n < num_meshblocks; n++) {
        // load npy data into host
        std::string num_str = std::to_string(n);
        auto padded_num_str = std::string(num_zeros - std::min(num_zeros, num_str.length()), '0') + num_str;
        std::string npy_str = data_dir + "/meshblock" + padded_num_str + ".npy";
        std::cout << "reading npy file at " << npy_str << std::endl;
        npy::npy_data npy_data = npy::read_npy<float>(npy_str);
        std::vector<float> npy_vector = npy_data.data; // ERR: populated with zeros?
        for (int i = 0; i < 1000; i++) {
            std::cout << "d[i] = " << npy_vector[i] << std::endl;
        }
        std::vector<unsigned long> npy_shape = npy_data.shape;
        size_t bytes_in_npy = npy_vector.size() * sizeof(float);
        // add check against header here?
        
        // copy data into host memory buffer. TODO readd offset
        std::memcpy(h_all_data, npy_vector.data(), bytes_in_npy);
    }

    for (int i = 0; i < 1000; i++) {
        std::cout << "i = " << h_all_data[i] << std::endl;
    }

    if (verbose) {
        float npy_read_dur = (float)(clock() - npy_read_start)/CLOCKS_PER_SEC;
        printf("read/malloc data          (npy->host)         %.6fs\n",npy_read_dur);
    }

    // copy data from host into device
    int il = 0; 
    clock_t data_copy_start = clock();
    checkCudaErrors(cudaMemcpy(d_data, &h_all_data[il], bytes_on_device, cudaMemcpyHostToDevice)); // no fail, but missing domain?
    checkCudaErrors(cudaPeekAtLastError());
    if (verbose) {
        float data_copy_dur = (float)(clock() - data_copy_start)/CLOCKS_PER_SEC;
        printf("memcpy data               (host->device)      %.6fs\n",data_copy_dur);
    }

    // initialise meshblock list on device
    clock_t mb_alloc_start = clock();
    MeshBlock **mb_list;
    checkCudaErrors(cudaMalloc((void **)&mb_list, num_meshblocks * sizeof(MeshBlock *)));
    
    // initialise meshblocks
    int mb_start = 0;
    for (int n = 0; n < num_meshblocks; n++) {
        vec3 xl = all_mb_info[n].xl;
        vec3 xr = all_mb_info[n].xr;
        vec3 mb_dims = all_mb_info[n].mb_dims;
        init_meshblock<<<1,1>>>(mb_list, n, xl, xr, mb_dims, d_data, mb_start);
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        mb_start += all_mb_info[n].mb_size;
    }

    // initialise mesh
    Mesh **mesh;
    checkCudaErrors(cudaMalloc((void **)&mesh, sizeof(Mesh * ))); 
    init_mesh<<<1,1>>>(mesh, mb_list, num_meshblocks);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    if (verbose) {
        float mb_alloc_dur = (float)(clock() - mb_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc/init MeshBlock     (device)            %.6fs\n",mb_alloc_dur);
    }

    // define render shape    
    int tx = 32, ty = 32; // must not exceed 1024 (max thread per block)
    const dim3 threads_per_block(tx,ty); 
    const dim3 blocks_per_grid(std::ceil((float)camera.num_pixels_X / tx), 
                                std::ceil((float)camera.num_pixels_Y / ty));
    int floats_on_device = bytes_on_device / sizeof(float);

    // iterate over cameras
    int img_count = 0;
    if (verbose) std::cout << "=============================================================\n";
    for (auto &camera : cameras) {
        
        clock_t this_img_start = clock();
        
        // add loop here to add new meshblocks

        // call render
        clock_t render_start = clock();
        render_from_mesh<<<blocks_per_grid,threads_per_block>>>(camera, d_img, mesh); // TODO: add partition support?
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) {
            float render_dur = (float)(clock() - render_start)/CLOCKS_PER_SEC;
            printf("render kernel             (device)            %.6fs\n",render_dur);
        }
        if (verbose) std::cout << ".............................................................\n";

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
        auto padded_num_str = std::string(num_zeros - std::min(num_zeros, num_str.length()), '0') + num_str;
        save_str = save_str + padded_num_str + ".npy";
        npy::npy_data_ptr<float> npy_img;
        npy_img.data_ptr = img;
        npy_img.shape = {(unsigned long)camera.num_pixels_X, (unsigned long)camera.num_pixels_Y};
        npy::write_npy(save_str, npy_img);
        if (verbose) {
            float npy_write_dur = (float)(clock() - npy_write_start)/CLOCKS_PER_SEC;
            printf("write data                (host->npy)         %.6fs\n",npy_write_dur);
            float this_img_dur = (float)(clock() - this_img_start)/CLOCKS_PER_SEC;
            printf("img total                 (host/device)       %.6fs\n",this_img_dur);
            if (verbose) std::cout << "=============================================================\n";
        }

        // prepare for next image
        wipe_img<<<blocks_per_grid,threads_per_block>>>(camera, d_img);
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        img_count++;
    }

    // perform cleanup of device/host data
    clock_t free_start = clock();
    free_mesh<<<1,1>>>(mesh, num_meshblocks);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(mesh));
    checkCudaErrors(cudaFree(d_data));
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