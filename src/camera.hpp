#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "vec3.hpp"

class Camera {
    public:
        // ctors
        __host__ __device__ Camera() {origin = vec3(1,0,0); 
                                        target = vec3(0,0,0); 
                                        normal = target - origin;
                                        num_pixels_X = 100; 
                                        num_pixels_Y = 100;
                                        length_X = 1.0;
                                        length_Y = 1.0;
                                        vertical = vec3(0,0,1);
                                        tilt = 0.0;
                                        upper_left = vec3(1,-0.5,0.5);
                                    }
        __host__ __device__ position(vec3 orig) {origin=orig;}
        __host__ __device__ aim(vec3 targ) {target=targ;}
        __host__ __device__ orient(vec3 vert, float t) {vertical=vert;, tilt=t}
        __host__ __device__ set_dims(int nX, int nY, float lX, float lY) {num_pixels_X = nX;
                                                                            num_pixels_Y = nY;
                                                                            length_X = lX;
                                                                            length_Y = lY;}
        __host__ __device__ init();

        // methods
        __host__ __device__ const get_pixel_origin(int i, int j) const;

        // internals
        vec3 origin, target, normal;
        int num_pixels_X, num_pixels_Y;
        float length_X, length_Y;
        float tilt;
        vec3 vertical, Xhat, Yhat;
        vec3 upper_left;
}

__host__ __device__ Camera::init() {
    // define axis unit vectors
    vec3 Xvec = cross(vertical, normal);
    Xhat = Xvec.norm();
    vec3 Yvec = cross(normal, Xhat);
    Yhat = Yvec.norm();
    if (tilt != 0) {
        Xhat = rotate_about(Xhat, normal, tilt);
        Yhat = rotate_about(Yhat, normal, tilt);
    }
    upper_left = origin - 0.5 * Xhat * length_X + 0.5 * Yhat * length_Y; 
}

__host__ __device__ get_pixel_origin(const int i, const int j) const {
    vec3 dY = -(i / num_pixels_Y) * length_Y;
    vec3 dX = (j / num_pixels_X) * length_X;
    return upper_left + dX * Xhat + dY * Yhat;
}

#endif