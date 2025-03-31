#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", __FILE__, __LINE__, __func__,       \
                    cudaGetErrorString(err));                                                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

using namespace std;

template <typename T> void printVector(const T *a, int size) {
    for (int i = 0; i < size; i++)
        cout << setw(3) << a[i];
    cout << endl;
}

__global__ void addKernel(const int *a, const int *b, int *c, int size) {
    // int i = threadIdx.x;

    for (int j = 0; j < 1000; j++) {

        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < size)
            c[i] = a[i] + b[i];
    }

    // printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    const int threadsPerBlock = 1024; // 최대 deviceProp.maxThreadsPerBlock = 1024 까지 가능

    const int size = 1024 * 1024 * 512; // size가 너무 커서 한 번에 모두 계산할 수 없다고 가정
    // const int size = 40;

    const int numSplits = 8; // 입출력 데이터를 몇 조각으로 나누는지
    const int numStreams = 2; // 스트림을 몇 개 사용할 지 (stream 하나가 한 번에 하나의 split을 담당
    // num_splits == num_streams 이면 앞의 예제와 같습니다.

    const int split_size = size / numSplits;
    const int blocks = int(ceil(float(split_size) / threadsPerBlock)); // 블럭 여러 개 사용

    int *a = nullptr;
    int *b = nullptr;
    int *c = nullptr;

    cudaMallocHost(&a, sizeof(int) * size); // pinned-memory
    cudaMallocHost(&b, sizeof(int) * size);
    cudaMallocHost(&c, sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        // a[i] = i;
        // b[i] = i;
    }

    cout << "Add vectors using CUDA" << endl;

    {
        vector<cudaStream_t> streams(numStreams); // num_streams 개 사용
        for (int s = 0; s < streams.size(); s++)
            cudaStreamCreate(&streams[s]);

        vector<int *> dev_a(numStreams); // num_splits -> num_streams
        vector<int *> dev_b(numStreams);
        vector<int *> dev_c(numStreams);

        for (int s = 0; s < numStreams;
             s++) { // GPU 메모리는 "전체 데이터 x num_streams / num_splits" 사용)
            cudaMalloc((void **)&dev_a[s], split_size * sizeof(int)); // size -> split_size
            cudaMalloc((void **)&dev_b[s], split_size * sizeof(int)); // size -> split_size
            cudaMalloc((void **)&dev_c[s], split_size * sizeof(int)); // size -> split_size
        }

        cudaEvent_t start,
            stop; // 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0); // 시작 시간 기록 (H2D -> Kernel -> D2H)

        int i = 0;

        // 안내:
        // - 이번 예제는 빈칸이 없습니다. 자신의 컴퓨터 사양에 맞춰서 전송/실행이 겹치는 패턴을 만들어 보세요.
        // - 아래 코드의 핵심은 데이터 전송과 커널 실행을 따로 묶었다는 점입니다.

        while (i <= numSplits) // i_split, i_stream 따로 사용
        {
            for (int i_stream = 0; i_stream < numStreams; i_stream++) {
                int i_split = i + i_stream;

                if (i > 0) // D2H 복사 (처음에는 받아올 데이터가 없음)
                {
                    // cout << "C " << i_split - numStreams << " " << i_stream << endl;
                    cudaMemcpyAsync(&c[(i_split - numStreams) * split_size], dev_c[i_stream],
                                    split_size * sizeof(int), cudaMemcpyDeviceToHost,
                                    streams[i_stream]);
                }

                if (i_split < numSplits) { // H2D 복사 (마지막에는 보낼 데이터가 없음)
                    cudaMemcpyAsync(dev_a[i_stream], &a[i_split * split_size],
                                    split_size * sizeof(int), cudaMemcpyHostToDevice,
                                    streams[i_stream]); // size -> split_size
                    cudaMemcpyAsync(dev_b[i_stream], &b[i_split * split_size],
                                    split_size * sizeof(int), cudaMemcpyHostToDevice,
                                    streams[i_stream]); // size -> split_size
                    // cout << "AB " << i_split << " " << i_stream << endl;
                }
            }

            if (i < numSplits)
                for (int i_stream = 0; i_stream < numStreams; i_stream++) {
                    addKernel<<<blocks, threadsPerBlock, 0, streams[i_stream]>>>(
                        dev_a[i_stream], dev_b[i_stream], dev_c[i_stream], split_size);

                    // cout << "Kernel " << i_stream << endl;
                }

            i += numStreams;
        }

        cudaEventRecord(stop, 0); // 끝나는 시간 기록
        cudaDeviceSynchronize();  // kernel이 끝날때까지 대기 (동기화)

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);          // 걸린 시간 계산
        cout << "Time elapsed: " << milliseconds << " ms" << endl; // 420ms 근처

        // 안내: kernel 실행 후 cudaGetLastError() 생략

        // 결과 확인
        if (size <= 40) { // size가 작을 경우에는 출력해서 확인
            printVector(a, size);
            printVector(b, size);
            printVector(c, size);
        }

        for (int i = 0; i < size; i++)
            if (c[i] != a[i] + b[i]) {
                cout << "Wrong result" << endl;
                return 1;
            }

        cout << "Correct" << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        for (int s = 0; s < numStreams; s++) {
            cudaFree(dev_a[s]);
            cudaFree(dev_b[s]);
            cudaFree(dev_c[s]);
        }

        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(c);

        cudaDeviceReset();
    }

    return 0;
}
