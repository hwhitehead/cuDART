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


int main(void) {
    // testing MeshBlock build

    const vec3 xl(0, 0, 0);
    const vec3 xr(1, 1, 1);
    const std::vector<int> dims{10, 10, 10};

    MeshBlock mb(xl, xr, dims);

    std::cout << "xl = " << mb.Edge(false) << std::endl;
    std::cout << "xr = " << mb.Edge(true) << std::endl;

    return 0;
}