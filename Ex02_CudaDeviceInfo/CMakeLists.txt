cmake_minimum_required(VERSION 3.27)
project(Ex02_CudaDeviceInfo CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.6/include")
endif ()

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(Ex02_CudaDeviceInfo Ex02_CudaDeviceInfo.cu)

set_target_properties(
    Ex02_CudaDeviceInfo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
