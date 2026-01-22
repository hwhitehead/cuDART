#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "vec3.hpp"
#include <math.h>

class Camera {
    public:
        // ctors
        __host__ Camera() {origin = vec3(1.0, 0.0, 0.0),
                                normal = vec3(-1.0, 0.0, 0.0),
                                bias = vec3(0.0, 0.0, 1.0);
                                length_X = 1.0;
                                length_Y = 1.0;
                                num_pixels_X = 10;
                                num_pixels_Y = 10;
                                tilt = 0.0;
                                build_camera();}
        
        // methods
        __host__ void build_camera();
        __device__ vec3 calc_pixel_origin(int i, int j) const;

        // attributes
        vec3 origin, normal, bias;                  // orientation generators
        vec3 unit_X, unit_Y, lower_left;            // orientation dependents
        int num_pixels_X, num_pixels_Y, num_pixels; // pixel dimensions 
        float length_X, length_Y, tilt;             // spatial dimensions and tilt
};

__host__ void Camera::build_camera() {
    // initialise dependent properties for Camera object
    
    // enforce normalisation
    bias = bias.vector_norm();
    normal = normal.vector_norm();
    if (bias == normal) {
        std::stringstream << err_msg;
        err_msg << "Unable to define unique camera orientation with bias == normal\n";
        CUDART_ERROR(err_msg);
    }

    // define spanning vectors
    unit_Y = (bias - bias.dot_prod(normal)).vector_norm(); // remove projected component
    unit_X = normal.cross_prod(unit_Y);

    // apply rotation
    unit_X = unit_X.rotate_about(normal, tilt);
    unit_Y = unit_Y.rotate_about(normal, tilt);

    // define spatial/pixel extent
    lower_left = origin - 0.5 * length_X * unit_X - 0.5 * length_Y * unit_Y;
    num_pixels = num_pixels_X * num_pixels_Y;
    return;
}

__device__ vec3 Camera::calc_pixel_origin(const int i, const int j) const {
    float dY = (((float)j + 0.5) / num_pixels_Y) * length_Y;
    float dX = (((float)i + 0.5) / num_pixels_X) * length_X;
    return lower_left + dX * unit_X + dY * unit_Y;
}

#endif