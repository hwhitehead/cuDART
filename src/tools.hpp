#ifndef TOOLS_HPP_
#define TOOLS_HPP_

#define CUDART_ERROR(x) std::cout << x.str(); std::exit(EXIT_FAILURE);

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

size_t calc_vram_limit(char *mem_char, float tolerance, size_t h_bytes) {
    // calculate available vram with user ceil
    
    float vram_limit_f = 1e12;
    if (mem_char != nullptr) {
        vram_limit_f = static_cast<float>(std::atof(mem_char)) * 1e9;
    }
    size_t free_t, total_t;
    checkCudaErrors(cudaMemGetInfo(&free_t,&total_t));
    float vram_free_f = static_cast<float>(free_t) * tolerance;
    float vram_avail_f = std::min(vram_free_f, vram_limit_f);
    size_t d_bytes_avail = static_cast<size_t>(vram_avail_f);
            
    // handle memory request excess
    bool d_mem_excess = (d_bytes_avail < h_bytes);
    size_t d_bytes;
    if (d_mem_excess) {
        std::stringstream err_msg;
        err_msg << "Total input memory exceeds VRAM, partitioning currently unsupported\n";
        CUDART_ERROR(err_msg);
    } else {
        d_bytes = h_bytes; // allocate entire dataset to device
    }
    return d_bytes;
}

__host__ std::string zero_pad_str(int value, size_t num_zero_pad) {
    std::string num_str = std::to_string(value);
    return std::string(num_zero_pad - std::min(num_zero_pad, num_str.length()), '0') + num_str;
}

__device__ float calc_doppler_fac(vec3 beta_vec, vec3 view_vec) {
    // calculate doppler boosting factor for a given bulk velocity and view
    float beta = beta_vec.vector_mag();
    float gamma = 1.0 / sqrt(1 - beta * beta);
    float cos_theta = beta_vec.dot_prod(view_vec) / beta; // view_vec assumed unit vec
    return 1.0 / (gamma * (1 - beta * cos_theta));
}

#endif