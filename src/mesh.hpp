#ifndef MESH_HPP_
#define MESH_HPP_

#include "vec3.hpp"

// THIS CODE IS WIP AND NOT ACTIVE IN MAIN

class Mesh {
    // data container to support heterogenous domain resolutions
    public:
        // ctors
        __device__ Mesh() {}
        __device__ Mesh(MeshBlock **l, int num_mb) {mb_list = l, num_meshblocks = num_mb;}
        
        // routines
        __device__ float calc_trace(const Ray &r);

        // members
        MeshBlock **mb_list;
        int num_meshblocks;
};

__global__ void init_mesh(Mesh **mesh, MeshBlock **mb_list, int num_meshblocks) { // allow mb to added here?
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

__device__ float Mesh::calc_trace(const Ray &r) {
    // calculate weighted path of a given ray through the Mesh and linked MeshBlocks

    float total_trace = 0;
    for (int n = 0; n < num_meshblocks; n++) {
        float local_trace = mb_list[n]->calc_trace(r); // ERR: returns zero
        total_trace += local_trace;
    }
    return total_trace;
}

#endif