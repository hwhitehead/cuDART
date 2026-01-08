#ifndef MESHBLOCK_HPP_
#define MESHBLOCK_HPP_

#include "vec3.hpp"
#include "ray.hpp"

class MeshBlock {
    public:
        // ctors
        __host__ __device__ MeshBlock() {}
        __host__ __device__ MeshBlock(vec3 xl, vec3 xr, std::vector<int> dims) {xl_ = xl; xr_ = xr; dims_ = dims; InitMeshBlock();}

        // routines
        __host__ __device__ bool CalcIntercept(Ray r, float &tl, float &tr);
        __host__ __device__ void CalcPath(Ray &r, float tl, float tr);
        __host__ __device__ float PathSum(Ray &r);
        __host__ __device__ int IntClamp(float f, float l, float r);
        __host__ __device__ vec3 Edge(bool sign) {return (sign) ? xr_ : xl_;}
        __host__ __device__ void ImportNumpyData(const std::string path);
        __host__ __device__ int Size() {return dims_[0] * dims_[1] * dims_[2];}

        // dtor
        ~MeshBlock();

        // public properties
        int axes_bitmap[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        float *data;

    private:
        std::vector<int> dims_;
        vec3 xl_, xr_, dx_;
        void InitMeshBlock();        
};

MeshBlock::~MeshBlock() {
    cudaFree(data);
}

__host__ __device__ void MeshBlock::ImportNumpyData(const std::string path) {
    // import numpy 
    npy::npy_data d = npy::read_npy<float>(path);
    std::vector<float> npy_data = d.data;
    float *p_data = npy_data.data();
    std::vector<unsigned long> shape = d.shape;
    size_t bytes = npy_data.size() * sizeof(float);

    // allocate device memory
    cudaMalloc(&data, bytes);

    // copy data into device memory
    cudaMemcpy(data, p_data, bytes, cudaMemcpyHostToDevice);

}

__host__ __device__ int MeshBlock::IntClamp(float f, float l, float r) {
    return std::max(l, std::min(std::floor(f), r));
}

__host__ __device__ void MeshBlock::InitMeshBlock() {
    dx_ = vec3();
    for (int i = 0; i <= 2; i++) {
        dx_[i] = (xr_[i] - xl_[i]) / dims_[i];
    }
    // initialise trace field
}

__host__ __device__ bool MeshBlock::CalcIntercept(Ray r, float &tl, float &tr) {
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