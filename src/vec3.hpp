#ifndef VEC3_HPP_
#define VEC3_HPP_

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3  {
    // class definition for the 3D vector type vec3
    // supports basic vector arithmetic

    public:
        // constructors
        __host__ __device__ vec3() {}
        __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
        //__device__ vec3(curandState *rand_state) {e[0] = curand_uniform(rand_state); e[1] = curand_uniform(rand_state); e[2] = curand_uniform(rand_state); }
        // full definitions
        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }
        __host__ __device__ inline float r() const { return e[0]; }
        __host__ __device__ inline float g() const { return e[1]; }
        __host__ __device__ inline float b() const { return e[2]; }
        __host__ __device__ inline const vec3& operator+() const { return *this; }
        __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ inline float operator[](int i) const { return e[i]; }
        __host__ __device__ inline float& operator[](int i) { return e[i]; };
        __host__ __device__ inline float mag() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
        __host__ __device__ inline float magSqr() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
        // basic arithmetic
        __host__ __device__ inline vec3& operator+=(const vec3 &v2);
        __host__ __device__ inline vec3& operator-=(const vec3 &v2);
        __host__ __device__ inline vec3& operator*=(const vec3 &v2);
        __host__ __device__ inline vec3& operator/=(const vec3 &v2);
        __host__ __device__ inline vec3& operator*=(const float t);
        __host__ __device__ inline vec3& operator/=(const float t);
        // vector arithmetic 
        __host__ __device__ inline vec3 norm(vec3 v);
        //__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2);
        //__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2);
        //__host__ __device__ inline vec3 reflect(const vec3 &i, const vec3 &n);
        //__host__ __device__ inline vec3 refract(const vec3 &i, vec3 &N, float n);
        __host__ __device__ inline vec3 rotate_about(const vec3 k, const float theta);

        // internal values
        float e[3];
};

inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline bool operator==(const vec3 &v1, const vec3 &v2) {
    return (v1.e[0] == v2.e[0]) && (v1.e[1] == v2.e[1]) && (v1.e[2] == v2.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v) {
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3 norm(vec3 v) {
    return v / v.mag();
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}
// TODO: cleanup old funcs
__host__ __device__ inline vec3 reflect(const vec3 &i, const vec3 &n) {
    float perp = 2.0f * dot(i,n);
    return norm(i - 2 * perp * n);
}

__host__ __device__ inline vec3 refract(const vec3 &i, vec3 &N, float n) {
    if (n == 1) {
        return i; 
    }
    float c1 = dot(i,N);
    float eta;
    if (c1 < 0) { // entering object
        c1 = -c1;
        eta = 1 / n;
    } else { // exiting object
        N = -N;
        eta = n;
    }
    float c2 = std::sqrt(1 - eta * eta * (1 - c1 * c1));
    vec3 t = eta * i + (eta*c1 - c2) * N;
    return t;
}

__host__ __device__ inline vec3 vec3::rotate_about(const vec3 k, const float theta) {
    // use Rodrigues' rotation formula to rotate v about k by theta
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);
    return *this * cos_theta + cross(k,*this) * sin_theta + k * dot(k,*this) * (1-cos_theta);
}

#endif