#ifndef MESHBLOCK_HPP_
#define MESHBLOCK_HPP_

#include <stdexcept>

#include "vec3.hpp"
#include "ray.hpp"
#include "tools.hpp"

class MeshBlock {
    // a MeshBlock provides a wrapper for data that allows for tracing by rays
    // MesHBlocks are purely device objects, including a pointer to traceable data device-allocated externally 
    public:
        // ctors
        __device__ MeshBlock() {}
        __device__ MeshBlock(const vec3 xleft, const vec3 xright, const vec3 dims, float *data);

        // routines
        __device__ bool calc_mb_intercept(Ray r, float &tl, float &tr);
        __device__ int int_clamp(float f, float l, float r);
        __device__ vec3 get_edge(bool sign) {return (sign) ? xr : xl;}
        __device__ void print_data();
        __device__ float calc_trace(Ray &r);

        // public properties
        const int axes_bitmap[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        float *mb_data;
        float sum;
        int mb_size;
        vec3 xl, xr, dx, mb_dims;    
};

__device__ float MeshBlock::calc_trace(Ray &r) {
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
        
        if (mb_entrance[1] > 0 && mb_entrance[2] > 0) {
            return 1.0 / mb_dims[0];
        } else if (mb_entrance[1] <= 0 && mb_entrance[2] > 0) {
            return (2.0 / 3) / mb_dims[0];
        } else if (mb_entrance[1] <= 0 && mb_entrance[2] <= 0)
            return (-2.0 / 3) / mb_dims[0];
        } else {
            return -1.0 / mb_dims[0];
        }

        // orientate trace
        for (int i = 0; i <= 2; i++) {
            float ray_mb_orgin = mb_entrance[i] - xl[i];
            cell[i] = int_clamp(ray_mb_orgin / dx[i], 0, (int)mb_dims[i] - 1); // awkward typing, template vec3?
            if (r.sign[i]) { 
                step_dir[i] = -1; // traverse backwards
                exit_cond[i] = -1; // stop walk when leading edge reached
                dt[i] = - dx[i] * r.inv_normal[i];
                next_t_cross[i] = tl + (cell[i] * dx[i] - ray_mb_orgin) * r.inv_normal[i];
            } else {
                step_dir[i] = 1; // traverse forwards
                exit_cond[i] = mb_dims[i]; // stop walk when tailing edge reached
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
            float dwell = next_t_cross[axis] - t_current; // INVALID

            // add local cell to trace
            int cell_index = cell[0] * (int)mb_dims[1] * (int)mb_dims[2]
                            + cell[1] * (int)mb_dims[2] + cell[2];
            trace += dwell * mb_data[cell_index];

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

__global__ void init_meshblock(MeshBlock **mb, const vec3 xl, const vec3 xr, vec3 dims, float *data) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        *mb = new MeshBlock(xl, xr, dims, data);
    }
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

__device__ void MeshBlock::print_data() {
    for (int i = 0; i < mb_size; i++) {
        printf("%.6f\n", mb_data[i]);
    }
}

__device__ MeshBlock::MeshBlock(const vec3 xleft, const vec3 xright, vec3 dims, float *data) {
    xl = xleft;
    xr = xright;
    mb_data = data;
    mb_dims = dims;
    mb_size = mb_dims[0] * mb_dims[1] * mb_dims[2];
    dx = (xr - xl) / mb_dims;
}

__device__ bool MeshBlock::calc_mb_intercept(Ray r, float &tl, float &tr) {
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