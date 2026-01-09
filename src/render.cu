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

// __global__ void PrintMeshBlockData(MeshBlock **mb) {
//     int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (thr_idx == 0) {
//         printf("printing data from mb data...\n");
//         printf("expect %.6d entries\n", (*mb)->mb_size);
//         printf("data at 0x%p\n", (*mb)->mb_data); // null here!
//         for (int i = 0; i < (*mb)->mb_size; i++) {
//             printf("%.6f\n", (*mb)->mb_data[i]);
//         }
//         //(*mb)->PrintData();
//         printf("finished print.");
//     }  
// }

// __global__ void HelloCUDA(float f) {
//     printf("Hello thread %d, f=%f\n", threadIdx.x, f);
// }

// __global__ void render(Camera *pcam, MeshBlock **mb) { // untested
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int j = threadIdx.y + blockIdx.y * blockDim.y;
//     if ((i >= max_x) || (j >= max_y)) return; // ignore oob
//   	int pixel_idx = j * max_x + i;

//     vec3 pixel_origin = pcam->get_pixel_origin(i, j);
//     Ray pixel_ray(pixel_origin, pcam->normal);
     
//     pcam->fb[pixel_idx] = pcam->trace(pixel_ray, mb);
// }


int main(void) {
    // testing MeshBlock build with real dataset

    // initiliase MeshBlock space
    MeshBlock **mb;
    cudaMalloc(&mb, sizeof(MeshBlock));

    // load npy data
    const std::string npy_path {"simdata/sn.npy"};
    npy::npy_data d = npy::read_npy<float>(npy_path);
    std::vector<float> npy_data = d.data;
    std::vector<unsigned long> npy_shape = d.shape;
    int mb_shape[3] = {npy_shape[0], npy_shape[1], npy_shape[2]};
    int data_size = npy_data.size();
    float *p_data = npy_data.data();
    size_t bytes = data_size * sizeof(float);
    std::cout << "finished load from npy" << std::endl;

    // allocate device memory
    float *data;
    checkCudaErrors(cudaMallocManaged(&data, bytes));
    std::cout << "finished data init on device" << std::endl;
    std::cout << "data at " << data << std::endl;

    // copy data into device memory
    checkCudaErrors(cudaMemcpy(data, p_data, bytes, cudaMemcpyHostToDevice));
    std::cout << "finished data copy to device" << std::endl;

    // initialise MeshBlock
    int thr_per_blk = 32;
    int blk_in_grid = 64;
    vec3 xl(0.0, 0.0, 0.0);
    vec3 xr(1.0, 1.0, 1.0);
    InitMeshBlock<<<thr_per_blk,blk_in_grid>>>(mb, xl, xr, mb_shape, data);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "finished meshblock init on device" << std::endl;

    // test MeshBlock properties
    PrintMeshBlockProperties<<<thr_per_blk,blk_in_grid>>>(mb);
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