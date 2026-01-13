#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "vec3.hpp"
#include <math.h>

class Camera {
    public:
        // ctors
        __host__ Camera()  {R_pos = 2.0;
                            theta_pos = M_PI / 2.0 + 0.00001;
                            phi_pos = 0.0 + 0.0000001;
                            length_X = 1.0;
                            length_Y = 1.0;
                            num_pixels_X = 10;
                            num_pixels_Y = 10;
                            num_pixels = num_pixels_X * num_pixels_Y;
                            bias = vec3(0,0,1);
                            tilt = 0.0;
                            update_camera();}
        __host__ void update_camera();
        __host__ void print_camera();

        // routines
        __device__ vec3 calc_pixel_origin(int i, int j) const;                           

        // internals
        float R_pos, theta_pos, phi_pos;
        vec3 origin, normal;
        int num_pixels_X, num_pixels_Y, num_pixels;
        float length_X, length_Y;
        float tilt;
        vec3 bias, unit_X, unit_Y;
        vec3 upper_left;
        vec3 lower_left;
};

__host__ void Camera::print_camera() {
    std::cout << "Image size = (" << num_pixels_X << "," << num_pixels_Y << ")\n";
    std::cout << num_pixels << " pixels total\n";
    std::cout << "origin =" << origin << std::endl;
    std::cout << "normal = " << normal << std::endl;
    std::cout << "unit_X = " << unit_X << std::endl;
    std::cout << "unit_Y = " << unit_Y << std::endl;
    std::cout << "upper left = " << upper_left << std::endl;
}

__host__ void Camera::update_camera() {
    // calculate position
    origin = R_pos * vec3(sin(theta_pos) * cos(phi_pos),
                            sin(theta_pos) * sin(phi_pos),
                            cos(theta_pos));
    
    // calculate orientation
    unit_X = (bias.cross_prod(origin)).vector_norm();
    unit_Y = (origin.cross_prod(unit_X)).vector_norm();
    normal = -origin.vector_norm();
    unit_X = unit_X.rotate_about(normal, tilt);
    unit_Y = unit_Y.rotate_about(normal, tilt);

    // define screen size
    upper_left = origin - 0.5 * length_X * unit_X + 0.5 * length_Y * unit_Y;
    lower_left = origin - 0.5 * length_X * unit_X - 0.5 * length_Y * unit_Y;
    num_pixels = num_pixels_X * num_pixels_Y;
}

__device__ vec3 Camera::calc_pixel_origin(const int i, const int j) const {
    // float dY = -((float)i / num_pixels_Y) * length_Y;
    // float dX = ((float)j / num_pixels_X) * length_X;
    // return upper_left + dX * unit_X + dY * unit_Y;
    float dY = ((float)j / num_pixels_Y) * length_Y;
    float dX = ((float)i / num_pixels_X) * length_X;
    return lower_left + dX * unit_X + dY * unit_Y;
}

#endif