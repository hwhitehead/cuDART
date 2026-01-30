#ifndef MESHBLOCK_HPP_
#define MESHBLOCK_HPP_

#include <stdexcept>

#include "vec3.hpp"
#include "ray.hpp"
#include "tools.hpp"

struct MeshBlockInfo {
    int mb_size;
    vec3 mb_dims, xl, xr;
};

class MeshBlock {
    // a MeshBlock provides a wrapper for data that allows for tracing by rays
    // MesHBlocks are purely device objects, including a pointer to traceable data device-allocated externally 
    public:
        // ctors
        __device__ MeshBlock() {}
        __device__ MeshBlock(const vec3 xleft, const vec3 xright, const vec3 dims, float *all_data, int mem_start);

        // routines
        __device__ bool calc_mb_intercept(const Ray &r, float &tl, float &tr);
        __device__ int int_clamp(float f, float l, float r);
        __device__ vec3 get_edge(bool sign) {return (sign) ? xr : xl;}
        __device__ void print_data();
        __device__ float calc_trace(const Ray &r);

        // public properties
        const int axes_bitmap[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        float *all_data;
        float sum;
        int mb_size, mem_start;
        vec3 xl, xr, dx, mb_dims;
};

__host__ std::vector<MeshBlockInfo> load_unlabelled_meshblock(std::string input_str, float* &h_all_data, size_t &h_bytes, bool verbose) {
    // load single homgenous meshblock info, allocate host memory and load data

    clock_t npy_read_start = clock();
    std::vector<MeshBlockInfo> all_mb_info = {};
    npy::npy_data npy_data = npy::read_npy<float>(input_str);
    std::vector<float> npy_vector = npy_data.data; 
    std::vector<unsigned long> npy_shape = npy_data.shape;
    vec3 mb_dims((float)npy_shape[0], (float)npy_shape[1], (float)npy_shape[2]);
    int mb_size = npy_vector.size();
    if (verbose) {
        float npy_read_dur = (float)(clock() - npy_read_start)/CLOCKS_PER_SEC;
        printf("npy read                  (host)              %.6fs\n",npy_read_dur);
    }
    
    //assume equal spacing in x, y, z and centering at origin
    float longest_side = static_cast<float>(*std::max_element(npy_shape.begin(), npy_shape.end()));
    vec3 mb_extent = mb_dims / longest_side;
    vec3 xl = -0.5 * mb_extent;
    vec3 xr = 0.5 * mb_extent;

    // stash info
    MeshBlockInfo mb_info;
    mb_info.mb_size = mb_size;
    mb_info.xl = xl;
    mb_info.xr = xr;
    mb_info.mb_dims = mb_dims;
    all_mb_info.push_back(mb_info);

    // allocate space on host
    h_bytes = mb_size * sizeof(float);
    clock_t h_alloc_start = clock();
    h_all_data = (float*) malloc(h_bytes);
    if (verbose) { 
        float h_alloc_dur = (float)(clock() - h_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc data               (host)              %.6fs\n",h_alloc_dur);
    }

    // load mb data into host memory
    clock_t memcpy_start = clock();
    std::memcpy(h_all_data, npy_vector.data(), h_bytes);
    if (verbose) { 
        float memcpy_dur = (float)(clock() - memcpy_start)/CLOCKS_PER_SEC;
        printf("memcpy data               (host)              %.6fs\n",memcpy_dur);
    }

    return all_mb_info;
}

__host__ std::vector<MeshBlockInfo> load_labelled_meshblocks(std::string input_str, float* &h_all_data, size_t &h_bytes, bool verbose) {
    // load heterogeneous meshblock info, allocate host memory and load data

    // read header data
    clock_t header_init_start = clock();
    std::vector<MeshBlockInfo> all_mb_info = {};
    std::string header_str = input_str + "/header.txt";
    std::ifstream header_file(header_str);
    int npy_floats = 0;
    if (header_file.is_open()) {
        std::string line;
        int line_count = 0;
        int mb_size, nx, ny, nz;
        float xl, yl, zl, xr, yr, zr;
        while (std::getline(header_file, line)) {
            std::istringstream iss(line);
            if (!(iss >> mb_size >> nx >> ny >> nz >> xl >> yl >> zl >> xr >> yr >> zr)) {
                std::stringstream err_msg;
                err_msg << "### FATAL ERROR in main ###\n";
                err_msg << "Unable to parse line " << line_count << " of header file at " << header_str << std::endl;
                CUDART_ERROR(err_msg);
            } else {
                MeshBlockInfo mb_info;
                mb_info.mb_size = mb_size;
                mb_info.xl = vec3(xl,yl,zl);
                mb_info.xr = vec3(xr,yr,zr);
                mb_info.mb_dims = vec3(nx,ny,nz);
                all_mb_info.push_back(mb_info);
                npy_floats += mb_size;
            }
            line_count++;
        }
        h_bytes = npy_floats * sizeof(float);
    } else {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main ####\n";
        err_msg << "Unable to open header file at " << header_str << std::endl;
        CUDART_ERROR(err_msg);
    } // end header read

    if (verbose) { 
        float header_init_dur = (float)(clock() - header_init_start)/CLOCKS_PER_SEC;
        printf("parsed header             (device)            %.6fs\n",header_init_dur);
    }

    // allocate space on host
    clock_t h_alloc_start = clock();
    h_all_data = (float*) malloc(h_bytes);
    if (verbose) { 
        float h_alloc_dur = (float)(clock() - h_alloc_start)/CLOCKS_PER_SEC;
        printf("malloc data               (host)              %.6fs\n",h_alloc_dur);
    }

    // load mb data into host memory
    clock_t npy_read_start = clock();
    int mem_offset = 0;
    size_t num_zero_pad = 3;
    for (int n = 0; n < all_mb_info.size(); n++) {
        std::string num_str = std::to_string(n);
        auto padded_num_str = std::string(num_zero_pad - std::min(num_zero_pad, num_str.length()), '0') + num_str;
        std::string npy_str = input_str + "/meshblock" + padded_num_str + ".npy";
        npy::npy_data npy_data = npy::read_npy<float>(npy_str);
        std::vector<float> npy_vector = npy_data.data; // populated
        std::vector<unsigned long> npy_shape = npy_data.shape;
        int floats_in_mb  = npy_vector.size();
        size_t bytes_in_mb = floats_in_mb * sizeof(float);
        
        // copy data into host memory buffer
        std::memcpy(h_all_data + mem_offset, npy_vector.data(), bytes_in_mb);
        mem_offset += floats_in_mb;
    } // end mb loop

    if (verbose) {
        float npy_read_dur = (float)(clock() - npy_read_start)/CLOCKS_PER_SEC;
        printf("npy read/memcpy           (host)              %.6fs\n",npy_read_dur);
    }

    return all_mb_info;
}

__device__ float MeshBlock::calc_trace(const Ray &r) {
    // calculate the weighted path of a given ray through the MeshBlock
    float tl, tr, trace = 0;
    bool hit = calc_mb_intercept(r, tl, tr);
    if (hit) { // valid intercept found
        // prep arrays for orientation
        int cell[3] = {0, 0, 0}; // convert to vec3? typesafe?
        float dt[3] = {0.0, 0.0, 0.0};
        float next_t_cross[3] = {0.0, 0.0, 0.0};
        int exit_cond[3] = {0, 0, 0};
        int step_dir[3] = {0, 0, 0};
        vec3 mb_entrance = r.march(tl);

        // orientate trace
        for (int i = 0; i <= 2; i++) {
            float ray_mb_orgin = mb_entrance[i] - xl[i];
            cell[i] = int_clamp(ray_mb_orgin / dx[i], 0, (int)mb_dims[i] - 1);
            if (r.sign[i]) { 
                step_dir[i] = -1; // traverse backwards
                exit_cond[i] = -1; // stop walk when leading edge reached
                dt[i] = - dx[i] * r.inv_normal[i];
                next_t_cross[i] = tl + (cell[i] * dx[i] - ray_mb_orgin) * r.inv_normal[i];
            } else {
                step_dir[i] = 1; // traverse forwards
                exit_cond[i] = (int)mb_dims[i]; // stop walk when tailing edge reached
                dt[i] = dx[i] * r.inv_normal[i];
                next_t_cross[i] = tl + ((cell[i]+1) * dx[i] - ray_mb_orgin) * r.inv_normal[i];
            } // end if
        } // end for

        // perform traversal
        float t_current = tl;
        while (t_current < tr) { // terminate on mb exit
            // identify next step direction
            int k = (((next_t_cross[0] < next_t_cross[1]) << 2) +
                    ((next_t_cross[0] < next_t_cross[2]) << 1) +
                    ((next_t_cross[1] < next_t_cross[2])));
            int axis = axes_bitmap[k];

            // determine dwell
            float dwell = next_t_cross[axis] - t_current; 

            // add local cell to trace 
            int cell_index = cell[0] * (int)mb_dims[1] * (int)mb_dims[2]
                            + cell[1] * (int)mb_dims[2] + cell[2];
            int data_index = mem_start + cell_index;
            trace += dwell * all_data[data_index];

            // update position of ray head
            t_current = next_t_cross[axis]; // += dwell
            cell[axis] += step_dir[axis];
            next_t_cross[axis] += dt[axis];

            // check for termination (necessary?)
            if (cell[axis] == exit_cond[axis]) break;
        } // end while     
    } // end if
    return trace;
}

__global__ void init_meshblock(MeshBlock **mb_list, int mb_index, const vec3 xl, const vec3 xr, vec3 dims, float *data, int mem_start) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        mb_list[mb_index] = new MeshBlock(xl, xr, dims, data, mem_start);
    }
    return;
}

__global__ void free_meshblock(MeshBlock **mb) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        delete *mb;
    }
}

__global__ void print_meshblock_properties(MeshBlock **mb) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        printf("printing data from mb...\n");
        printf("xl = (%.3f, %.3f, %.3f)\n", (*mb)->xl[0], (*mb)->xl[1], (*mb)->xl[2]);
        printf("xr = (%.3f, %.3f, %.3f)\n", (*mb)->xr[0], (*mb)->xr[1], (*mb)->xr[2]);
        printf("data_size = %d\n", (*mb)->mb_size);
    }  
}

__device__ int MeshBlock::int_clamp(float f, float l, float r) {
    return max(l, min(std::floor(f), r));
}

__device__ MeshBlock::MeshBlock(const vec3 xleft, const vec3 xright, const vec3 dims, float *data, int m_start) {
    xl = xleft;
    xr = xright;
    all_data = data;
    mem_start = m_start;
    mb_dims = dims;
    mb_size = (int)mb_dims[0] * (int)mb_dims[1] * (int)mb_dims[2];
    dx = (xr - xl) / mb_dims;
}

__device__ bool MeshBlock::calc_mb_intercept(const Ray &r, float &tl, float &tr) {
    tl = 0.0, tr = 0.0;
    float tcmin, tcmax, tmin, tmax;
    for (int i = 0; i <= 2; i++) {
        tcmin = (get_edge(r.sign[i])[i] - r.origin[i]) * r.inv_normal[i];
        tcmax = (get_edge(1 - r.sign[i])[i] - r.origin[i]) * r.inv_normal[i];
        if (i == 0) {
            tmin = tcmin;
            tmax = tcmax;
            continue;
        }

        if ((tmin > tcmax) or (tcmin > tmax)) {
            return false;
        }

        if (tcmin > tmin) {
            tmin = tcmin;
        }

        if (tcmax < tmax) {
            tmax = tcmax;
        }
    }
    tl = tmin;
    tr = tmax;
    return true;
}

#endif