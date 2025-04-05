#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",              \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

using namespace std;

template <typename T>
void printVector(const T *a, int size)
{
    for (int i = 0; i < size; i++)
        cout << setw(3) << a[i];
    cout << endl;
}

__global__ void addKernel(const int *a, const int *b, int *c, int size)
{
    // int i = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] + b[i];

    // printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    // 스트림 여러개 사용하기!!

    const int size = 1024 * 1024 * 512; // size가 너무 커서 한 번에 모두 계산할 수 없다고 가정
    // const int size = 40;
    const int numSplits = 8; // 여러 조각으로 나눠서 계산
    const int splitSize = size / numSplits;

    int *a = nullptr;
    int *b = nullptr;
    int *c = nullptr;

    cudaMallocHost(&a, sizeof(int) * size); // pinned-memory
    cudaMallocHost(&b, sizeof(int) * size);
    cudaMallocHost(&c, sizeof(int) * size);

    // CPU에서 할 일도 아래에서 분할 수행할 수 있습니다.
    for (int i = 0; i < size; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    cout << "Add vectors using CUDA" << endl;

    {
        // 스트림도 numSplits 개수만큼, 여러개 생성
        vector<cudaStream_t> streams(numSplits);
        for (int s = 0; s < streams.size(); s++)
            cudaStreamCreate(&streams[s]);

        vector<int *> dev_a(numSplits);
        vector<int *> dev_b(numSplits);
        vector<int *> dev_c(numSplits);

        // 각각의 스트림이 데이터를 따로따로 받으려면,
        // 메모리 공간도 따로 잡아줘야 함
        // s 의 값=idx 는 각 스트림 인덱스마다.. 메모리 할당
        // 따라서, 스트림을 여러개 사용하려면, 메모리가 더 필요함!
        // 즉, 속도를 올릴려면, 뭔가는 희생을 해야하는데, 여기서는, 메모리를 더 사용!
        for (int s = 0; s < numSplits; s++)
        {                                                            // GPU 메모리가 넉넉하다고 가정
            cudaMalloc((void **)&dev_a[s], splitSize * sizeof(int)); // size -> split_size
            cudaMalloc((void **)&dev_b[s], splitSize * sizeof(int)); // size -> split_size
            cudaMalloc((void **)&dev_c[s], splitSize * sizeof(int)); // size -> split_size
        }

        cudaEvent_t start, stop; // 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0); // 시작 시간 기록 (H2D -> Kernel -> D2H)

        // 여러 스트림에게, 복사만 몰아서 지시
        for (int s = 0; s < numSplits; s++)
        {
            // cudaStreamSynchronize(streams[s]); // 참고용이며 여기서는 사용하지 않습니다.

            // cudaMemcpyAsync(dev_a[s], &a[s * splitSize], splitSize * sizeof(int), cudaMemcpyHostToDevice, TODO ); // size -> split_size
            // cudaMemcpyAsync(dev_b[s], &b[s * splitSize], splitSize * sizeof(int), cudaMemcpyHostToDevice, TODO ); // size -> split_size
            cudaMemcpyAsync(dev_a[s], &a[s * splitSize], splitSize * sizeof(int), cudaMemcpyHostToDevice, streams[s]); // size -> split_size
            cudaMemcpyAsync(dev_b[s], &b[s * splitSize], splitSize * sizeof(int), cudaMemcpyHostToDevice, streams[s]); // size -> split_size
        }

        // 여러 스트림에게, 커널 실행만 몰아서 지시
        for (int s = 0; s < numSplits; s++)
        {
            int threadsPerBlock = 1024;                                 // 최대 deviceProp.maxThreadsPerBlock = 1024 까지 가능
            int blocks = int(ceil(float(splitSize) / threadsPerBlock)); // 블럭 여러 개 사용
            // addKernel << <blocks, threadsPerBlock, 0, TODO >> > (dev_a[s], dev_b[s], dev_c[s], splitSize);
            addKernel<<<blocks, threadsPerBlock, 0, streams[s]>>>(dev_a[s], dev_b[s], dev_c[s], splitSize);
        }

        // 여러 스트림에게, 복사만 몰아서 지시시
        for (int s = 0; s < numSplits; s++)
        {
            // cudaMemcpyAsync(&c[s * splitSize], dev_c[s], splitSize * sizeof(int), cudaMemcpyDeviceToHost, TODO );
            cudaMemcpyAsync(&c[s * splitSize], dev_c[s], splitSize * sizeof(int), cudaMemcpyDeviceToHost, streams[s]);
        }

        // for(int s = 0; s < num_splits; s++)  // 참고용이며 여기서는 사용하지 않습니다.
        //	cudaStreamSynchronize(streams[s]); // 밑에서 cudaDeviceSynchronize() 사용

        cudaEventRecord(stop, 0); // 끝나는 시간 기록
        cudaDeviceSynchronize();  // kernel이 끝날때까지 대기 (동기화)

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);          // 걸린 시간 계산
        cout << "Time elapsed: " << milliseconds << " ms" << endl; // 453ms

        // 안내: kernel 실행 후 cudaGetLastError() 생략

        // 결과 확인
        if (size <= 40)
        { // size가 작을 경우에는 출력해서 확인
            printVector(a, size);
            printVector(b, size);
            printVector(c, size);
        }

        for (int i = 0; i < size; i++)
            if (c[i] != a[i] + b[i])
            {
                cout << "Wrong result" << endl;
                return 1;
            }

        cout << "Correct" << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        for (int s = 0; s < numSplits; s++)
        {
            cudaFree(dev_c[s]);
            cudaFree(dev_a[s]);
            cudaFree(dev_b[s]);
        }

        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(c);

        cudaDeviceReset();
    }

    return 0;
}
