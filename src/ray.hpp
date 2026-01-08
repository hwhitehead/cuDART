#ifndef RAY_HPP_
#define RAY_HPP_

#include <vector>

#include "vec3.hpp"

class Ray {
    public:
        // ctors
        __device__ Ray() {}
        __device__ Ray(Ray& r) {origin = r.origin; 
                                normal = r.normal;, 
                                inv_normal = 1.0 / r.normal;
                                sign = vec3(inv_normal.x() < 0, 
                                            inv_normal.y() < 0,
                                            inv_normal.z() < 0);}
        __device__ Ray(const vec3& o, const vec3& n) {origin = o; normal = n;}
        
        // methods
        __device__ vec3 O() const {return origin;}
        __device__ vec3 N() const {return normal;}
        __device__ vec3 march(float t) const {return origin + t * normal;}

        // internals
        vec3 origin;
        vec3 normal;
        vec3 inv_normal;
        vec3 sign;

};

#endif