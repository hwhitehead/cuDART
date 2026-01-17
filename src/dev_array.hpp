#ifndef DEV_ARRAY_HPP_
#define DEV_ARRAY_HPP_

class DevArray {
    public:
        // ctors
        DevArray() {start_ = 0; end_ = 0;}
        DevArray(size_t size) {
            allocate(size);
        }
        // dtor
        ~DevArray() {
            free();
        }
        // methods
        void resize(size_t size);
        size_t get_size() const;
        const float* data() const;
        void set_data(const float* src, size_t size);
        void pull_data(float *target, size_t size);

    private:
        void allocate(size_t size);
        void free();
        float *start_;
        float *end_;
};

// public routines

void DevArray::resize(size_t size) {
    free();
    allocate(size);
    return;
}

size_t DevArray::get_size() const {
    return end_ - start_;
}

const float* DevArray::data() const {
    return start_;
}

void DevArray::set_data(const float* src, size_t size) {
    size_t len = std::min(size, get_size());
    cudaError_t result = cudaMemcpy(start_, src, len * sizeof(float), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        throw std::runtime_error("failed to memcpy to device");
    }
    return;
}

void DevArray::pull_data(float *target, size_t size) {
    size_t len = std::min(size, get_size());
    cudaError_t result = cudaMemcpy(target, start_, len * sizeof(float), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        throw std::runtime_error("failed to memcpy to host");
    }


}

// private routines

void DevArray::allocate(size_t size) {
    cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(float));
    if (result != cudaSuccess) {
        start_ = end_ = 0;
        throw std::runtime_error("failed to alloc on device");
    }
    end_ = start_ + size;
}

void DevArray::free() {
    if (start_ != 0) {
        cudaFree(start_);
        start_ = end_ = 0;
    }
}

#endif