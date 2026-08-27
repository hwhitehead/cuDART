#ifndef TOOLS_HPP_
#define TOOLS_HPP_

#define CUDART_ERROR(x) std::cout << x.str(); std::exit(EXIT_FAILURE);

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
__host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// statics
const size_t num_zero_pad = 5;
const float kpc_to_m = 3.086e+19;                                     // in m
const float Myr_to_s = 1e6 * 365 * 24 * 60 * 60;                      // in s
const float c_light = 3e8;                                            // in m/s

struct TraceArgs {
    // relativistic settings
    bool relativistic; 
    float doppler_index;
    // lookback settings 
    bool lookback, keep_edge;
    float t_obs, snapshot_dt, c, last_time; 
    float inv_snapshot_dt, inv_c;
    int snapshot_index, num_snapshots, last_snapshot;
    // image buffer settings
    int camera_index;
};

__host__ size_t calc_vram_limit(char *mem_char, float tolerance, size_t h_bytes) {
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

__device__ float calc_boost_factor(vec3 beta_vec, vec3 view_vec, float doppler_index) {
    // calculate D, the doppler boosting factor for a given bulk velocity and view
    // emissivity is boosted as D^(2-alpha) := D^doppler_index
    float beta = beta_vec.vector_mag();
    float inv_gamma = sqrt(1 - beta * beta);
    float one_plus_beta_cos_theta = 1 + beta_vec.dot_prod(view_vec); // view_vec assumed unit vec
    float doppler_fac = inv_gamma / one_plus_beta_cos_theta;
    return pow(doppler_fac, doppler_index);
}

__device__ float calc_lookback_factor(float s, TraceArgs trace_args) {
    // caculate the lerp weighting factor between temporal states
    float t_bar = trace_args.t_obs - s * trace_args.inv_c; // lookback time at distance s along line-of-sight

    // handle edge cases
    if (t_bar < 0) { // lookback time occurs before simulation data
        return ((trace_args.snapshot_index == 0) && trace_args.keep_edge); // use first snapshot
    } else if (t_bar > trace_args.last_time) { // lookback time occurs after simulation data
        return ((trace_args.snapshot_index == trace_args.last_snapshot) && trace_args.keep_edge); // use last snapshot
    }

    // given membership in simulation duration, identify neighbours
    int m_bar = floor(t_bar * trace_args.inv_snapshot_dt); // leading snapshot s.t. t_bar \in [m_bar, m_bar+1]
    if ((trace_args.snapshot_index >= m_bar) && (trace_args.snapshot_index <= m_bar + 1)) {
        float lerp_factor = abs(t_bar - trace_args.snapshot_index * trace_args.snapshot_dt) * trace_args.inv_snapshot_dt;
        return 1.0 - lerp_factor; // snapshot is adjacent, lerp contribution
    } 

    return 0; // snapshot is not adjacent, no contribution
}

__host__ void host_to_npy(const std::string &filename, float* host_addr, std::vector<unsigned long int> data_shape) {

    std::ofstream file_stream(filename, std::ofstream::binary);
    if (!file_stream) {
        throw std::runtime_error("I/O error, unable to save file at " + filename);
    }

    const npy::dtype_t dtype = npy::dtype_map.at(std::type_index(typeid(float)));

    npy::header_t header{dtype, false, data_shape}; // always save with C-order indexing
    npy::write_header(file_stream, header);

    auto data_size = static_cast<size_t>(npy::comp_size(data_shape));

    file_stream.write(reinterpret_cast<const char *>(host_addr), sizeof(float) * data_size);
    return;
}

__host__ std::vector<unsigned long int> npy_to_host(const std::string &file_str, float* &host_addr, size_t &h_bytes, bool verbose, bool host_malloc) {
    
    // cast to stream
    std::ifstream file_stream(file_str, std::ifstream::binary);
    if (!file_stream) {
        throw std::runtime_error("I/O error: unable to load file at " + file_str);
    }

    // load header data
    std::string header_str = npy::read_header(file_stream);
    npy::header_t header = npy::parse_header(header_str);

    // check if the typestring matches float32
    const npy::dtype_t dtype = npy::dtype_map.at(std::type_index(typeid(float)));
    if (header.dtype.tie() != dtype.tie()) {
        throw std::runtime_error("formatting error: input data must be float32 dtype");
    }

    // compute data size based on shape
    auto data_size = static_cast<size_t>(npy::comp_size(header.shape));

    // if flagged, allocate data
    if (host_malloc) {
        h_bytes = data_size * sizeof(float);
        auto h_alloc_start = std::chrono::steady_clock::now();
        host_addr = (float*) malloc(h_bytes);
        if (verbose) { 
            std::chrono::duration<float> h_alloc_dur = (std::chrono::steady_clock::now() - h_alloc_start);
            printf("malloc data               (host)              %.6fs\n",h_alloc_dur.count());
        } // end verbose
    } // end host_malloc

    // read data into host_addr
    file_stream.read(reinterpret_cast<char *>(host_addr), sizeof(float) * data_size);

    // return shape
    return header.shape;
}

#endif