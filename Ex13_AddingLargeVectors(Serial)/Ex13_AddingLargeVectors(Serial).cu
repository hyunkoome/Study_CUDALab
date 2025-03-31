#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

using namespace std;

template<typename T>
void printVector(const T* a, int size)
{
	for (int i = 0; i < size; i++)
		cout << setw(3) << a[i];
	cout << endl;
}

__global__ void addKernel(const int* a, const int* b, int* c, int size)
{
	// int i = threadIdx.x;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size)
		c[i] = a[i] + b[i];

	// printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
	const int threadsPerBlock = 1024; // 최대 deviceProp.maxThreadsPerBlock = 1024 까지 가능

	const int size = 1024 * 1024 * 512; // size가 너무 커서 한 번에 모두 계산할 수 없다고 가정
	//const int size = 40;
	const int numSplits = 8;            // 여러 조각으로 나눠서 계산 
	const int split_size = size / numSplits;
	
	int* a = nullptr;
	int* b = nullptr;
	int* c = nullptr;

	cudaMallocHost(&a, sizeof(int) * size); // pinned-memory
	cudaMallocHost(&b, sizeof(int) * size);
	cudaMallocHost(&c, sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	cout << "Add vectors using CUDA" << endl;

	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		cudaMalloc((void**)&dev_a, split_size * sizeof(int)); // size -> split_size (GPU 메모리 적게 사용)
		cudaMalloc((void**)&dev_b, split_size * sizeof(int)); // size -> split_size (GPU 메모리 적게 사용)
		cudaMalloc((void**)&dev_c, split_size * sizeof(int)); // size -> split_size (GPU 메모리 적게 사용)
		 
		cudaEvent_t start, stop;// 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0); // 시작 시간 기록 (H2D -> Kernel -> D2H)

		for (int s = 0; s < numSplits; s++) 
		{
			//cudaMemcpyAsync(dev_a, &a[s * TODO ], TODO * sizeof(int), cudaMemcpyHostToDevice, stream); // size -> split_size
			//cudaMemcpyAsync(dev_b, &b[s * TODO ], TODO * sizeof(int), cudaMemcpyHostToDevice, stream); // size -> split_size

			//int blocks = int(ceil(float( TODO ) / threadsPerBlock)); // 블럭 여러 개 사용
			//addKernel << <blocks, threadsPerBlock, 0, stream >> > (dev_a, dev_b, dev_c, TODO );
			
			//cudaMemcpyAsync(&c[s * TODO ], dev_c, TODO * sizeof(int), cudaMemcpyDeviceToHost, stream);
		}

		cudaEventRecord(stop, 0);  // 끝나는 시간 기록
		cudaDeviceSynchronize();   // kernel이 끝날때까지 대기 (동기화)

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
		cout << "Time elapsed: " << milliseconds << " ms" << endl; // 600ms

		// 안내: kernel 실행 후 cudaGetLastError() 생략

		// 결과 확인
		if (size <= 40) { // size가 작을 경우에는 출력해서 확인
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

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		cudaFreeHost(a);
		cudaFreeHost(b);
		cudaFreeHost(c);

		cudaDeviceReset();
	}


	return 0;
}

