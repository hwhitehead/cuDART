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

__host__ void load_cameras(std::vector<Camera> cameras, char *camera_char, bool verbose) {
    // load camera data into a vector from camera .txt file

    clock_t camera_read_start = clock();

    // handle null char
    if (camera_char == nullptr) {
        if (verbose) {
            std::cout << "No user specified camera input, falling back to default.\n";
        }
        Camera default_camera;
        cameras.push_back(default_camera);
        return;
    }

    // given valid char, load from .txt file
    std::string camera_str(camera_char);
    std::ifstream camera_file(camera_str);
    int line_count = 0, num_pixels_X, num_pixels_Y;
    if (camera_file.is_open()) {
        std::string line;
        while (std::getline(camera_file, line)) {
            float inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9, inp10, inp11;
            std::istringstream iss(line);
            if (!(iss >> inp0 >> inp1 >> inp2 >> inp3 >> inp4 >> inp5 >> inp6 >> inp7 >> inp8 >> inp9 >> inp10 >> inp11)) {
                std::stringstream err_msg;
                err_msg << "### FATAL ERROR in main ###\n";
                err_msg << "Unable to parse line " << line_count << "of camera file at " << camera_str << std::endl;
                CUDART_ERROR(err_msg);
            } else {
                // read line by line
                if (line_count == 0) { // read static header
                    num_pixels_X = inp0;
                    num_pixels_Y = inp1;
                } else { // read dynamic camera data
                    Camera this_camera;
                    this_camera.num_pixels_X = num_pixels_X;
                    this_camera.num_pixels_Y = num_pixels_Y;
                    this_camera.origin = vec3(inp0, inp1, inp2);
                    this_camera.normal = vec3(inp3, inp4, inp5);
                    this_camera.bias = vec3(inp6, inp7, inp8);
                    this_camera.tilt = inp9;
                    this_camera.length_X = inp10;
                    this_camera.length_Y = inp11;
                    this_camera.build_camera();
                    cameras.push_back(this_camera);
                }
            }
            line_count++;
        }
        camera_file.close();
    } else {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in main ###\n";
        err_msg << "Unable to open camera file at " << camera_str << std::endl;
        CUDART_ERROR(err_msg);
    }
    if (verbose) {
        float camera_read_dur = (float)(clock() - camera_read_start)/CLOCKS_PER_SEC;
        printf("camera read               (host)              %.6fs\n",camera_read_dur);
    }
    return;
}

__host__ void Camera::build_camera() {
    // initialise dependent properties for Camera object
    
    // enforce normalisation
    bias = bias.vector_norm();
    normal = normal.vector_norm();
    if (bias == normal) {
        std::stringstream err_msg;
        err_msg << "Unable to define unique camera orientation with bias == normal\n";
        CUDART_ERROR(err_msg);
    }

    // define spanning vectors
    unit_Y = (bias - bias.dot_prod(normal) * normal).vector_norm(); // remove projected component
    unit_X = normal.cross_prod(unit_Y);

    // ensure unitary
    unit_Y = unit_Y.vector_norm();
    unit_X = unit_X.vector_norm();

    // apply rotation
    unit_Y = unit_Y.rotate_about(normal, tilt);
    unit_X = unit_X.rotate_about(normal, tilt);

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