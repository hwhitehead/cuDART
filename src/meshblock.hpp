#ifndef MESHBLOCK_HPP_
#define MESHBLOCK_HPP_

#include "vec3.hpp"

class MeshBlock {
    public:
        // ctors
        __host__ __device__ MeshBlock() {}
        __host__ __device__ MeshBlock(vec3 xl, vec3 xr, std::vector<int> dims) {xl_ = xl; xr_ = xr; dims_ = dims; InitMeshBlock();}

        // routines
        bool CalcIntercept(Ray r, float &tl, float &tr);
        void CalcPath(Ray &r, float tl, float tr);
        float PathSum(Ray &r);
        int IntClamp(float f, float l, float r);
        vec3 Edge(bool sign) {return (sign) ? xr_ : xl_;}

        // public properties
        int axes_bitmap[8] = {2, 1, 2, 1, 2, 2, 0, 0};

    private:
        std::vector<int> dims_;
        vec3 xl_, xr_, dx_;
        void InitMeshBlock();

}

int MeshBlock::IntClamp(float f, float l, float r) {
    return std::max(l, std::min(std::floor(f), r));
}

void InitMeshBlock() {
    dx_ = vec3();
    for (int i = 0; i <= 2; i++) {
        dx_[i] = (xr_[i] - xl_[i]) / dims_[i];
    }
    // initialise trace field
}

bool MeshBlock::CalcIntercept(Ray r, float &tl, float &tr) {
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
            std::cout << "false!" << std::endl;
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