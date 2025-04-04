#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    const int threadsPerBlock = 256; // 최대 deviceProp.maxThreadsPerBlock = 1024 까지 가능

    const int size = 1024 * 1024 * 256; // 여기서는 블럭을 여러 개 사용해야 하는 큰 size
    // const int size = 37;

    // 생각해볼 점: 블럭이 몇 개가 필요할까?

    // vector<int> a(size);
    // vector<int> b(size);
    // vector<int> c_single(size);     // 결과 확인용
    // vector<int> c(size, -1); // CUDA에서 계산한 결과 저장

    int *a = nullptr;
    int *b = nullptr;
    int *c_single = nullptr;
    int *c = nullptr;

    // cudaMallocHost 사용해서 gpu 아니라, cpu 메모리를 동작으로 할당
    // 그런데, cpu 메모리인데, pinned-memory를 할당 받음
    // OS가 안쓰는 cpu 메모리는 하드 등의 저장장치로 옮겨두는데,
    // pinned-memory 는 못옮기게 고정시켜서, 항상 1차 저장장치, 메모리에 위치함
    // 그래서, pinned-memory로 잡을수 있는 최대 크기는,
    // 물리적으로 RAM 크기를 넘을수 없고, RAM 크기에서 OS 사용하는 메모리 크기를 뺀.. 크기임
    // 그래서, **GPU로 보낼 데이터를 저장해두는 변수는 보통, cpu의 pinned-memory 에 할당하는 것을 권장**
    cudaMallocHost(&a, sizeof(int) * size); // pinned-memory
    cudaMallocHost(&b, sizeof(int) * size);
    cudaMallocHost(&c_single, sizeof(int) * size); // GPU 통신에 사용되지 않기 때문에 꼭 pinned-memory를 사용할 필요는 없음
    cudaMallocHost(&c, sizeof(int) * size);

    for (int i = 0; i < size; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c_single[i] = a[i] + b[i];
    }

    cout << "Add vectors using CUDA" << endl;

    {
        // 참고: cudaStreamSynchronize()를 사용해서 개별 스트림만 따로 동기화 하고 싶은 경우
        //      여기서는 cudaDeviceSynchronize()를 사용하기 때문에 스트림만 따로 동기화하지는 않았습니다.
        //{
        //	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        //	unsigned int flags; cudaError_t err = cudaGetDeviceFlags(&flags);
        //	if (err != cudaSuccess) {
        //		cerr << "Failed to get device flags: " << cudaGetErrorString(err) << endl; return 1;
        //	}
        //	if (flags & cudaDeviceScheduleBlockingSync) {
        //		cout << "cudaDeviceScheduleBlockingSync: set" << endl;
        //	}
        //	else {
        //		cout << "cudaDeviceScheduleBlockingSync: NOT set" << endl;
        //	}
        //}

        // 스트림은 cudaStream_t 자료형으로 선언하고, cudaStreamCreate() 로 만듦
        // 스트림은 메모리카피(cudaMemcpyAsync), 커널 실행 등에 사용됨
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        int *dev_a = nullptr;
        int *dev_b = nullptr;
        int *dev_c = nullptr;

        cudaMalloc((void **)&dev_a, size * sizeof(int)); // input a
        cudaMalloc((void **)&dev_b, size * sizeof(int)); // input b
        cudaMalloc((void **)&dev_c, size * sizeof(int)); // output c

        cudaEvent_t start, stop; // 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0); // 시작 시간 기록 (H2D -> Kernel -> D2H)

        // TODO 완성해야 실행됩니다. 간단합니다.

        // 주의: 뒤에 Async가 붙은 cudaMemcpyAsync() 사용
        // cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, TODO ); // 비동기적으로 복사 복사
        // cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, TODO );

        // 이 stream을 이용해서, cpu에서 gpu로 메모리를 복사해라!
        // 비동기 함수이기때문에, gpu가 끝날때까지 cpu는 기다리지 않음
        cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream); // 비동기적으로 복사 복사
        cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream);

        int blocks = int(ceil(float(size) / threadsPerBlock)); // 블럭 여러 개 사용
        // addKernel << <blocks, threadsPerBlock, 0, TODO >> > (dev_a, dev_b, dev_c, size);

        // 커널을 실행시킬때도, stream을 사용해서 비동기적으로 실행시킬 수 있음
        //         블럭수, 쓰래드개수, 쉐어드메모리크기, 어떤스트림(흐름)으로실행시킬지
        // 비동기식이기때문에, cpu가 gpu한테 명령만 내리고, gpu가 안끝나도, 자기일 할 수 있음
        addKernel<<<blocks, threadsPerBlock, 0, stream>>>(dev_a, dev_b, dev_c, size);

        // 안내:
        // - 커널 호출할때 stream을 지정해주지 않으면 내부적으로 기본 스트림을 사용합니다.
        // - cudaMemcpy()와 달리 커널 호출은 항상 비동기적입니다. GPU에게 명령만 내리고 CPU는 바로 다음 명령을 수행합니다.
        // - CPU에게 GPU가 일을 다 끝날때까지 강제로 기다리게 하고 싶다면 아래의 cudaDeviceSynchronize()를 사용할 수 있습니다.
        // - 함수 이름에서 볼 수 있듯이, 이렇게 기다리는 것을 "동기화(synchronize)"라고 합니다.

        // 결과 복사 device -> host
        // cudaMemcpyAsync(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost, TODO );
        // 스트림을 이용해서 계산 결과를 받아옴
        cudaMemcpyAsync(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost, stream);

        // **스트림 사용하는 이유**
        // 1) GPU는 GPU대로 따로 일하고, CPU는 그 사이에 다른 일을 할 수 있음
        //      다른 성질의 디바이스, CPU와 GPU를 패러럴하게 비동기적으로 사용한다.
        // 2) 데이터 통신을 하는 동안, 커널을 실행 시킬 수 있음
        //      전체적으로 실행시간을 줄일 수 있음
        // 그런데, 스트림 1개는 직렬(serial)로 동작하므로,
        // 제대로 병렬로 동작시키려면, 여러 스트림을 사용해야 함

        cudaEventRecord(stop, 0); // 끝나는 시간 기록
        cudaDeviceSynchronize();  // kernel이 끝날때까지 대기 (동기화)

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
        cout << "Time elapsed: " << milliseconds << " ms" << endl;

        // 안내: kernel 실행 후 cudaGetLastError() 생략

        // 결과 확인
        if (size < 40)
        { // size가 작을 경우에는 출력해서 확인
            printVector(a, size);
            printVector(b, size);
            printVector(c_single, size);
            printVector(c, size);
        }

        for (int i = 0; i < size; i++)
            if (c_single[i] != c[i])
            {
                cout << "Wrong result" << endl;
                return 1;
            }

        cout << "Correct" << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(c_single);
        cudaFreeHost(c);

        cudaDeviceReset();
    }

    return 0;
}
