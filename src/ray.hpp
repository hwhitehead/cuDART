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
                                inverse_normal = 1.0 / r.normal;
                                sign = vec3(inverse_normal.x() < 0, 
                                            inverse_normal.y() < 0,
                                            inverse_normal.z() < 0);}
        __device__ Ray(const vec3& o, const vec3& n) {origin = o; normal = n;}
        
        // methods
        __device__ vec3 O() const {return origin;}
        __device__ vec3 N() const {return normal;}
        __device__ vec3 march(float t) const {return origin + t * normal;}

        // internals
        vec3 origin;
        vec3 normal;
        vec3 inverse_normal;
        vec3 sign;

}

#endif