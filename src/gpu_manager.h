#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <iostream>
#include <cstddef>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class GPUManager {
private:
    // 声明一个不透明的结构体指针，具体实现在 .cu 中
    struct Impl;
    Impl* pimpl;

    GPUManager();
    ~GPUManager();

public:
    static GPUManager& getInstance();

    double* allocate_device_memory(size_t num_elements);
    void free_device_memory(double* d_ptr, size_t num_elements);
    void clear_pool();

    static void sync_to_gpu(const double* h_ptr, double* d_ptr, size_t num_elements);
    static void sync_to_cpu(double* h_ptr, const double* d_ptr, size_t num_elements);

    cudaStream_t get_stream();
    std::size_t fork_aux_streams(
        cudaStream_t parent, cudaStream_t* aux_streams, std::size_t capacity
    );
    void join_aux_streams(
        cudaStream_t parent, const cudaStream_t* aux_streams, std::size_t count
    );
    void synchronize_all();
    void synchronize_streams(const cudaStream_t* streams, std::size_t count);
};

#endif /* GPU_MANAGER_H */
