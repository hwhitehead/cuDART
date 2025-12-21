// standard library imports
#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>

// custom external library imports
#include "npy.hpp"


// custom local library imports 

int main(void) {
    // perform basic test of libnpy functionality
    const std::string path {"simdata/data.npy"};
    npy::npy_data d = npy::read_npy<double>(path);
    std::vector<double> data = d.data;
    std::vector<unsigned long> shape = d.shape;
    std::cout << shape << std::endl;
    return 0;
}