// standard library imports
#include <iostream>
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

__global__ void PrintData(MeshBlock mb) {
    for (int i = 0; i < mb.Size(); i++) {
        printf("%.6f\n", mb.data[i]);
    }
}

int main(void) {
    // testing MeshBlock build

    const vec3 xl(0, 0, 0);
    const vec3 xr(1, 1, 1);
    const std::vector<int> dims{10, 10, 10};

    MeshBlock mb(xl, xr, dims);

    std::cout << "xl = " << mb.Edge(false) << std::endl;
    std::cout << "xr = " << mb.Edge(true) << std::endl;

    // load npy data
    const std::string npy_path {"simdata/data.npy"};
    mb.ImportNumpyData(npy_path);

    int thr_per_blk = 16;
    int blk_in_grid = 16;

    PrintData<<<1, 1>>>(mb);

    return 0;
}