#ifndef MESH_HPP_
#define MESH_HPP_

#include "vec3.hpp"
#include "meshblock.hpp"
#include "tools.hpp"

class Mesh {
    // data container to support heterogenous domain resolutions
    public:
        // ctors
        __device__ Mesh() {}
        __device__ Mesh(MeshBlock **l, int num_mb) {mb_list = l, num_meshblocks = num_mb;}
        
        // routines
        __device__ float calc_trace(const Ray &r, TraceArgs trace_args);

        // members
        MeshBlock **mb_list;
        int num_meshblocks;
};

__global__ void render_from_mesh(Camera camera, float *img, Mesh **mesh, TraceArgs trace_args) {
    // idenitfy relevant pixel for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= camera.num_pixels_X) || (j >= camera.num_pixels_Y)) return; // skip oob
  	int pixel_index = i * camera.num_pixels_Y + j; 

    // initialise ray
    vec3 pixel_origin = camera.calc_pixel_origin(i, j);
    Ray pixel_ray(pixel_origin, camera.normal);
    
    // copy camera observer time into trace_args
    trace_args.t_obs = camera.t_obs;

    // determine stash address
    int mem_position = pixel_index;
    if (trace_args.save_to_buffer) {
        mem_position += camera.num_pixels * trace_args.camera_index;
    }

    // calculate pixel value from MeshBlock data
    img[mem_position] += (*mesh)->calc_trace(pixel_ray, trace_args);
    return;
}

__global__ void init_mesh(Mesh **mesh, MeshBlock **mb_list, int num_meshblocks) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i == 0 && j == 0) {
        *mesh = new Mesh(mb_list, num_meshblocks);
    }
    return;
}

__global__ void free_mesh(Mesh **mesh, int num_meshblocks) {
    // free mesh and linked meshblocks from memory
    for (int n = 0; n < num_meshblocks; n++) {
        delete (*mesh)->mb_list[n];
    }
    
    delete *mesh;
}

__device__ float Mesh::calc_trace(const Ray &r, TraceArgs trace_args) {
    // calculate weighted path of a given ray through the Mesh and linked MeshBlocks

    float total_trace = 0;
    for (int n = 0; n < num_meshblocks; n++) {
        float local_trace = mb_list[n]->calc_trace(r, trace_args);
        total_trace += local_trace;
    }
    return total_trace;
}

__host__ void build_containers(std::vector<MeshBlockInfo> all_mb_info, float* &d_data, MeshBlock** &mb_list, Mesh** &mesh, bool verbose) {
    // allocate and initalise data containers (meshblock, meshblock list, mesh)

    clock_t container_alloc_start = clock();

    // allocate memory on device for meshblock list
    int num_meshblocks = all_mb_info.size();
    checkCudaErrors(cudaMalloc((void **)&mb_list, num_meshblocks * sizeof(MeshBlock *)));

    // allocate and intialise meshblocks on device
    int mem_start = 0;
    for (int n = 0; n < num_meshblocks; n++) {
        init_meshblock<<<1,1>>>(all_mb_info[n], mb_list, d_data);
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        mem_start += all_mb_info[n].mb_size;
    }

    // allocate and initialise mesh on device
    checkCudaErrors(cudaMalloc((void **)&mesh, sizeof(Mesh * ))); 
    init_mesh<<<1,1>>>(mesh, mb_list, num_meshblocks);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (verbose) {
        float container_alloc_dur = (float)(clock() - container_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc/init containers    (device)            %.6fs\n",container_alloc_dur);
    }
    
    return;
}


#endif