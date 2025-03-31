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
	// int i = threadIdx.x;
	// blockDim: 블럭 디멘젼: 한 블럭안에 쓰래드가 몇개가 사용이 되냐?
	// blockIdx: 블럭 인덱스: 몇번째 블럭이냐?
	// threadIdx: 쓰래드 인덱스: 그 블럭안에서 몇번째 쓰레드 이냐?

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size)
		c[i] = a[i] + b[i];

	printf("%u %u %u %d\n", blockDim.x, blockIdx.x, threadIdx.x, i);
}

// 큰 문제에 대해서는
// addKernel<<<size/numThreadsPerBlock, numThreadsPerBlock>>>(...)
// addKernel<<<1, 나머지>>>(...)
// 이렇게 커널 호출 자체를 두 번 하는 것 보다 if(i < size) 같이 조건을 걸어주는 것이 (보통) 더 빠르고 편합니다.

int main()
{
	// 여러 쓰래드 뿐 아니라, 여러 블럭 사용하는 예제
	//const int size = 1024 * 1024 * 256; // 여기서는 블럭을 여러 개 사용해야 하는 큰 size
	const int size = 1024 * 256; // 여기서는 블럭을 여러 개 사용해야 하는 큰 size
	//const int size = 37;
	//const int size = 8; // addKernel <<<2, 4>> > (dev_a, dev_b, dev_c, size); // 블럭2개, 쓰래드4개 => 총 8개 쓰래드 사용 시..결과 Correct

	// 생각해볼 점: 블럭이 몇 개가 필요할까?

	vector<int> a(size);
	vector<int> b(size);
	vector<int> c_single(size);     // 결과 확인용
	vector<int> c(size, -1); // CUDA에서 계산한 결과 저장

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

		cudaMalloc((void**)&dev_a, size * sizeof(int)); // input a
		cudaMalloc((void**)&dev_b, size * sizeof(int)); // input b
		cudaMalloc((void**)&dev_c, size * sizeof(int)); // output c

		cudaEvent_t start, stop;// 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0); // 시작 시간 기록

		cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

		const int threadsPerBlock = 1024; // 최대 deviceProp.maxThreadsPerBlock = 1024 까지 가능
		// int blocks = TODO; // 블럭 여러 개 사용

		// ceil() 올림 연산: 딱 떨어지지 않을 경우 +1해서, 더 많은 블럭으로 돌림(일을 더 함), 
		// 단, 시행하는 커널 함수에서 if (i < size) 를 추가해서, 조건을 둠으로써, 불필요한 연산을 하지 않도록 막아줌
		int blocks = int(ceil(float(size)/threadsPerBlock)); // 블럭 여러 개 사용, 
		// addKernel <<<TODO, TODO>>> (dev_a, dev_b, dev_c, size);
		// addKernel <<<블럭개수, 쓰래드개수>>> (dev_a, dev_b, dev_c, size);		

		// addKernel <<<2, 4>> > (dev_a, dev_b, dev_c, size); // 블럭2개, 쓰래드4개 => 총 8개 쓰래드 사용
		addKernel <<<blocks, threadsPerBlock>>> (dev_a, dev_b, dev_c, size);

		// 안내: kernel 실행 후 cudaGetLastError() 생략

		// 결과 복사 device -> host
		cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		cudaEventRecord(stop, 0);  // 끝나는 시간 기록

		cudaDeviceSynchronize();       // kernel이 끝날때까지 대기 (동기화)
		// cudaEventSynchronize(stop); // 불필요 (동기화 중복)

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
		cout << "Time elapsed: " << milliseconds << " ms" << endl;

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

