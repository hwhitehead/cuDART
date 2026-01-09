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

__global__ void PrintData(MeshBlock &mb) {
    printf("printing data from mb data...\n");
    for (int i = 0; i < mb.Size(); i++) {
        printf("%.6f\n", mb.data[i]);
    }
    printf("finished print.");
}

__global__ void HelloCUDA(float f) {
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

__global__ void SumData(MeshBlock &mb) {
    printf("starting sum...\n");
    mb.sum = 0;
    for (int i = 0; i < mb.Size(); i++) {
        mb.sum += mb.data[i];
    }
    printf("finished sum.\n");
}

int main(void) {
    // testing MeshBlock build

    const vec3 xl(0, 0, 0);
    const vec3 xr(1, 1, 1);
    const std::vector<int> dims{10, 10, 10};

    MeshBlock mb(xl, xr, dims);

    // load npy data
    const std::string npy_path {"simdata/data.npy"};
    mb.ImportNumpyData(npy_path);

    std::cout << "import finished." << std::endl;

    int thr_per_blk = 1;
    int blk_in_grid = 1;

    SumData<<<thr_per_blk,blk_in_grid>>>(mb);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "size = " << mb.Size() << std::endl;
    std::cout << "sum = " << mb.sum << std::endl;

    PrintData<<<1, 1>>>(mb);
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}