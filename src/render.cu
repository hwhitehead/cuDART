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

__global__ void PrintMeshBlockData(MeshBlock **mb) {
    printf("printing data from mb data...\n");
    printf("expect %.6d entries", (*mb)->Size());
    for (int i = 0; i < (*mb)->Size(); i++) {
        printf("%.6f\n", (*mb)->data[i]);
    }
    printf("finished print.");
}

__global__ void HelloCUDA(float f) {
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

__global__ void SumData(MeshBlock **mb) {
    printf("starting sum...\n");
    (*mb)->sum = 0;
    for (int i = 0; i < (*mb)->Size(); i++) {
        (*mb)->sum += (*mb)->data[i];
    }
    printf("finished sum.\n");
}

__global__ void InitMeshBlock(MeshBlock **mb, const vec3 xl, const vec3 xr, float *data, const int data_size) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        *mb = new MeshBlock(xl, xr, data, data_size);
    }
}

int main(void) {
    // testing MeshBlock build

    // initiliase MeshBlock space
    MeshBlock **mb;
    cudaMalloc((void **) &mb, sizeof(MeshBlock *));

    // load npy data
    const std::string npy_path {"simdata/data.npy"};
    npy::npy_data d = npy::read_npy<float>(npy_path);
    std::vector<float> npy_data = d.data;
    int data_size = npy_data.size();
    float *p_data = npy_data.data();
    size_t bytes = data_size * sizeof(float);
    std::cout << "finished load from npy" << std::endl;

    // allocate device memory
    float *data;
    checkCudaErrors(cudaMallocManaged((void **)&data, bytes));
    std::cout << "finished data init on device" << std::endl;

    // copy data into device memory
    checkCudaErrors(cudaMemcpy(data, p_data, bytes, cudaMemcpyHostToDevice));
    std::cout << "finished data copy to device" << std::endl;

    // initialsie MeshBlock
    int thr_per_blk = 1;
    int blk_in_grid = 1;
    vec3 xl(0.0, 0.0, 0.0);
    vec3 xr(1.0, 1.0, 1.0);
    InitMeshBlock<<<thr_per_blk,blk_in_grid>>>(mb, xl, xr, data, data_size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "finished meshblock init on device" << std::endl;

    // check MeshBlock data
    PrintMeshBlockData<<<thr_per_blk,blk_in_grid>>>(mb);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "finished." << std::endl;
    // SumData<<<thr_per_blk,blk_in_grid>>>(mb);
    // checkCudaErrors(cudaPeekAtLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // std::cout << "size = " << (*mb)->Size() << std::endl;
    // std::cout << "sum = " << (*mb)->sum << std::endl;

    return 0;
}