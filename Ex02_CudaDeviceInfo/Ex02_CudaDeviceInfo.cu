#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace std::chrono;

// GPU 같은 외부 하드웨어를 사용할 때는 아래와 같은 매크로를 이용해서 
// 매번 제대로 실행되었는지를 확인하는 것이 좋습니다.
// 여기서는 학습 효율을 높이기 위해 대부분 생략하겠습니다.
// 오류는 Nsight를 통해서 확인할 수도 있습니다.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

void printCudaDeviceInfo()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << "\n-> " << cudaGetErrorString(error_id) << endl;
        return;
    }

    if (deviceCount == 0) {
        cout << "No CUDA devices found." << endl;
    }
    else {
        cout << "Found " << deviceCount << " CUDA devices." << endl;
    }

    int driverVersion = 0;
    int runtimeVersion = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        cout << "Device " << device << " - " << deviceProp.name << endl;
        cout << "  CUDA Driver Version / Runtime Version:         "
            << driverVersion / 1000 << "." << (driverVersion % 1000) / 10 << " / "
            << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << endl;
        cout << "  Total Global Memory:                           " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  GPU Clock Rate:                                " << deviceProp.clockRate * 1e-3f << " MHz" << endl;
        cout << "  Memory Clock Rate:                             " << deviceProp.memoryClockRate * 1e-3f << " MHz" << endl;
        cout << "  Memory Bus Width:                              " << deviceProp.memoryBusWidth << " bits" << endl;
        cout << "  L2 Cache Size:                                 " << deviceProp.l2CacheSize / 1024 << " KB" << endl;
        cout << "  Max Texture Dimension Size (x,y,z):            " << deviceProp.maxTexture1D << ", " << deviceProp.maxTexture2D[0] << ", " << deviceProp.maxTexture3D[0] << endl;
        cout << "  Max Layered Texture Size (dim) x layers:       " << deviceProp.maxTexture2DLayered[0] << " x " << deviceProp.maxTexture2DLayered[1] << endl;
        cout << "  Total Constant Memory:                         " << deviceProp.totalConstMem / 1024 << " KB" << endl;
        cout << "  Unified Addressing:                            " << (deviceProp.unifiedAddressing ? "Yes" : "No") << endl;
        cout << "  Concurrent Kernels:                            " << (deviceProp.concurrentKernels ? "Yes" : "No") << endl;
        cout << "  ECC Enabled:                                   " << (deviceProp.ECCEnabled ? "Yes" : "No") << endl;
        cout << "  Compute Capability:                            " << deviceProp.major << "." << deviceProp.minor << endl;
        
        /*
        asyncEngineCount is 1 when the device can concurrently copy memory between host and device while executing a kernel.
        It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.
        It is 0 if neither of these is supported.
        */

        cout << "  Async Engine Count:                            " << deviceProp.asyncEngineCount << endl;

        /*
        4090 laptop GPU 기준
        - Number of Multiprocessors: 76
        - Maximum Threads per MultiProcessor: 1536
        - Maximum number of threads in a GPU = 76 * 1536 = 116,736
        
        [참고] Maximum number of threads in a GPU https://forums.developer.nvidia.com/t/maximum-number-of-threads-in-a-gpu/237761?form=MG0AV3

        - Maximum Threads per Block: 1024
        - Warp size in threads: 32
        - Maximum Threads per Dimension (x, y, z): 1024, 1024, 64 (주의: xyz를 곱해서 최대 1024가 되는 조합)
        */

        cout << "  Number of Multiprocessors:                     " << deviceProp.multiProcessorCount << endl;
        cout << "  Maximum Threads per MultiProcessor:            " << deviceProp.maxThreadsPerMultiProcessor << endl;
        cout << "  Maximum Threads per Block:                     " << deviceProp.maxThreadsPerBlock << endl;
        cout << "  Maximum Blocks Per Multiprocessor:             " << deviceProp.maxBlocksPerMultiProcessor << endl;
        cout << "  Maximum Threads per Dimension (x, y, z):       " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << endl;
        cout << "  Warp size in threads:                          " << deviceProp.warpSize << endl;
        cout << "  Maximum Grid Size (x,y,z):                     " << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << endl;
    }
}

int main()
{
    printCudaDeviceInfo();
 
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

