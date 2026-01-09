#ifndef MESHBLOCK_HPP_
#define MESHBLOCK_HPP_

#include <stdexcept>

#include "vec3.hpp"
#include "ray.hpp"
#include "tools.hpp"

class MeshBlock {
    public:
        // ctors
        __device__ MeshBlock() {}
        __device__ MeshBlock(const vec3 xl, const vec3 xr, float *data, const int data_size);

        // routines
        __device__ bool CalcIntercept(Ray r, float &tl, float &tr);
        __device__ void CalcPath(Ray &r, float tl, float tr);
        __device__ float PathSum(Ray &r);
        __device__ int IntClamp(float f, float l, float r);
        __device__ vec3 Edge(bool sign) {return (sign) ? xr_ : xl_;}
        __host__ __device__ int Size() {return data_size_;}
        __device__ void PrintData();

        // dtor
        ~MeshBlock();

        // public properties
        int axes_bitmap[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        float *mb_data;
        float sum;

    private:
        int data_size_;
        // std::vector<int> dims_;
        vec3 xl_, xr_, dx_;
        void InitMeshBlock();        
};

__device__ int MeshBlock::IntClamp(float f, float l, float r) {
    return max(l, min(std::floor(f), r));
}

__device__ void MeshBlock::PrintData() {
    for (int i = 0; i < data_size_; i++) {
        printf("%.6f\n", data[i]);
    }
}

__device__ MeshBlock::MeshBlock(const vec3 xl, const vec3 xr, float *data, const int data_size) {
    xl_ = xl;
    xr_ = xr;
    printf("passed data as 0x%p\n", data);
    mb_data = data;
    data_size_ = data_size;
    // dx_ = vec3();
    // for (int i = 0; i <= 2; i++) {
    //     dx_[i] = (xr_[i] - xl_[i]) / dims_[i];
    // }
}

__device__ bool MeshBlock::CalcIntercept(Ray r, float &tl, float &tr) {
    tl = 0.0, tr = 0.0;
    float tcmin, tcmax, tmin, tmax;
    for (int i = 0; i <= 2; i++) {
        tcmin = (Edge(r.sign[i])[i] - r.origin[i]) * r.inv_normal[i];
        tcmax = (Edge(1 - r.sign[i])[i] - r.origin[i]) * r.inv_normal[i];
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