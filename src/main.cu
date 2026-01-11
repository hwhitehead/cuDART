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

int main(void) {
    // testing MeshBlock build with real dataset

    // initiliase MeshBlock space
    MeshBlock **mb;
    cudaMalloc(&mb, sizeof(MeshBlock *));

    // load npy data
    const std::string npy_path {"simdata/sn_low.npy"};
    npy::npy_data d = npy::read_npy<float>(npy_path);
    std::vector<float> npy_data = d.data;
    std::vector<unsigned long> npy_shape = d.shape;
    vec3 dims((float)npy_shape[0], (float)npy_shape[1], (float)npy_shape[2]);
    int data_size = npy_data.size();
    float *p_data = npy_data.data();
    size_t bytes = data_size * sizeof(float);
    std::cout << "finished load from npy" << std::endl;

    // allocate device memory
    float *data;
    checkCudaErrors(cudaMallocManaged(&data, bytes));
    std::cout << "finished data init on device" << std::endl;

    // copy data into device memory
    checkCudaErrors(cudaMemcpy(data, p_data, bytes, cudaMemcpyHostToDevice));
    std::cout << "finished data copy to device" << std::endl;

    // initialise MeshBlock
    int thr_per_blk = 32;
    int blk_in_grid = 64;
    vec3 xl(0.0, 0.0, 0.0);
    vec3 xr(1.0, 1.0, 1.0);
    init_meshblock<<<thr_per_blk,blk_in_grid>>>(mb, xl, xr, dims, data);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "finished meshblock init on device" << std::endl;

    // test MeshBlock properties
    print_meshblock_properties<<<thr_per_blk,blk_in_grid>>>(mb);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "finished printing properties." << std::endl;

    // check MeshBlock data
    // PrintMeshBlockData<<<thr_per_blk,blk_in_grid>>>(mb);
    // checkCudaErrors(cudaPeekAtLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // std::cout << "finished." << std::endl;
    // SumData<<<thr_per_blk,blk_in_grid>>>(mb);
    // checkCudaErrors(cudaPeekAtLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // std::cout << "size = " << (*mb)->Size() << std::endl;
    // std::cout << "sum = " << (*mb)->sum << std::endl;

    cudaFree(mb);
    cudaFree(data);

    return 0;
}