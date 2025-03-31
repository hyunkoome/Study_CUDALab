#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

template<typename T>
void printVector(const vector<T>& a)
{
    for (int v : a)
        cout << setw(3) << v;
    cout << endl;
}

__global__ void addKernel(const int* a, const int* b, int* c, int size)
{
    int i = threadIdx.x;

    c[i] = a[i] + b[i];

    // printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    int size = 1024; // 블럭(block) 하나만으로 계산할 수 있는 크기 = deviceProp.maxThreadsPerBlock = 1024

    vector<int> a(size);
    vector<int> b(size);
    vector<int> c_single(size); // 결과 확인용
    vector<int> c(size, -1);    // CUDA에서 계산한 결과 저장

    for (int i = 0; i < size; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c_single[i] = a[i] + b[i];
    }

    cout << "Add vectors using CUDA" << endl;

    {
        int* dev_a = nullptr;
        int* dev_b = nullptr;
        int* dev_c = nullptr;

        //.. 시간 측정: 변수 이벤트 세팅
        cudaEvent_t start, stop;// 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMalloc((void**)&dev_a, size * sizeof(int)); // input a
        cudaMalloc((void**)&dev_b, size * sizeof(int)); // input b
        cudaMalloc((void**)&dev_c, size * sizeof(int)); // output c

        cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

        //.. 시간 측정: 시작 시간 기록
        cudaEventRecord(start, 0); // 시작 시간 기록

        // 블럭 1개 x 쓰레드 size개
        addKernel << <1, size >> > (dev_a, dev_b, dev_c, size);

        //.. 시간 측정: 끝 시간 기록
        cudaEventRecord(stop, 0);  // 끝나는 시간 기록

        cudaDeviceSynchronize();       // kernel이 끝날때까지 대기 (동기화)
        // cudaEventSynchronize(stop); // 불필요 (동기화 중복)

        //.. 시간 측정 계산
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
        
        //.. 시간 측정 출력
        cout << "Time elapsed: " << milliseconds << " ms" << endl;

        // 안내: kernel 실행 후 cudaGetLastError() 생략

        // 결과 복사 device -> host
        cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        cudaDeviceReset();
    }

    if (size < 40) { // size가 작을 경우에는 출력해서 확인
        printVector(a);
        printVector(b);
        printVector(c_single);
        printVector(c);
    }

    for (int i = 0; i < size; i++)
        if (c_single[i] != c[i])
        {
            cout << "Wrong result" << endl;
            return 1;
        }

    cout << "Correct" << endl;

    return 0;
}

