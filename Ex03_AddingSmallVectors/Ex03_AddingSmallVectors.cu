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

// 앞에 붙는 __global__은 CPU(host)에서 실행시킬 수 있는 CUDA 커널(kernel) 함수(=GPU에서 연산 동작 수행)라는 의미입니다.
// __host__ : CPU에서 호출하고 CPU에서 실행되는 함수
// __device__ : GPU에서 호출하고 GPU에서 실행되는 함수
// __host__ __device__ : (함께 사용하면) CPU/GPU 모두에서 실행될 수 있는 함수, 주로 간단한 보조 함수
__global__ void addKernel(const int* a, const int* b, int* c)
{
	// threadIdx: 어떤 block 안에서 몇번째 thread 인지를 알려주는 변수
	int i = threadIdx.x; 

	// c[i] = TODO;
	c[i] = a[i] + b[i];

	// 안내: 쿠다에서도 printf()를 사용할 수 있습니다. 기본적인 디버깅에 활용하세요.
	// printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);

	// 처음에는 x위주로, 2d 이미지 사용시 x/y 사용, 뉴럴네트워크 확장시 x/y/z 까지 사용
	// 참고로, x * y * z 해서 최대값이 deviceProp.maxThreadsPerBlock = 1024 임
	printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
	/*
	1) 메인 메모리로부터 GPU 메모리로 데이터를 복사한다
	2) CPU가 GPU에게 일을 하라고 시킨다.
	3) 실제로 GPU가 일을 한다.
	4) 일이 끝나면, GPU에 저장되어 있는 결과를 메인 메모리로 복사한다.
	*/

	int size = 10; // 블럭(block) 하나만으로 계산할 수 있는 크기 = deviceProp.maxThreadsPerBlock = 1024 <= 블럭 하나가 실행할 수 있는 쓰래드 개수

	// 더할 벡터 vector 2개를 cpu 메모리에 만듦: a, b
	vector<int> a(size);
	vector<int> b(size);
	vector<int> cSingle(size); // 결과 확인용: 멀티 쓰레딩을 사용하지 않고 미리 정답을 계산할 변수
	vector<int> c(size, -1);    // CUDA에서 계산한 결과 저장

	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
		cSingle[i] = a[i] + b[i];  // 결과 확인용: 멀티 쓰레딩을 사용하지 않고 미리 정답을 계산
	}

	cout << "Add vectors using CUDA" << endl;
	{
		// CUDA에서는 cpu 포인터를 가지고, gpu 포인터 처럼 사용 가능
		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		// 그래서, cpu 메모리에 대한 포인터는 보통 c++의 vector 를 사용하고
		// gpu 메모리에 대한 포인터는 변수명 이름앞에 dev_ 를 붙이고, 사용함 

		// cudaMalloc 사용해서, GPU 메모리를 할당 받음
		// cudaMalloc(할당받은 메모리에 대한 포인터, 크기)
		cudaMalloc((void**)&dev_a, size * sizeof(int)); // input a
		cudaMalloc((void**)&dev_b, size * sizeof(int)); // input b
		cudaMalloc((void**)&dev_c, size * sizeof(int)); // output c

		// cudaMemcpy 사용해서, CPU 메모리에 있는 데이터를 GPU 메모리로 복사
		// 디바이스 즉 gpu 메모리->a메모리 데이터, 크기, 메모리 카피 방향
		// cudaMemcpyHostToDevice: 아주 중요! 메모리카피(host=cpu)2(device=gpu)
		cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

		// 블럭 1개 * 쓰레드 size개
		// addKernel <<<1, TODO >>> (dev_a, dev_b, dev_c);
		// addKernel<<<블럭이 몇 개 인지, 각 블럭당 쓰레드가 몇 개인지 >>>(dev_a, dev_b, dev_c);

		// addKernel 함수가, 만약 cpu 함수이면 => addKernel(dev_a, dev_b, dev_c); 이렇게 사용할텐데
		// gpu 커널 함수이므로, => addKernel <<<블럭개수, 블럭 당 쓰레드 개수 >>> (dev_a, dev_b, dev_c); 이렇게 사용

		// CUDA 커널은 block 단위로 실행 시켜야 함!
		addKernel <<< 1, size >>> (dev_a, dev_b, dev_c);
		// 총 스레드 개수: 블럭 개수 * 쓰래드 개수.. 따라서 블럭은 어떻게 디버깅? print? => ex04에서..

		// 안내:
		// - cudaMemcpy()와 달리 커널 호출은 항상 비동기적(asynchronous)입니다. 
		// - GPU에게 명령만 내리고 CPU는 바로 다음 명령을 수행한다는 의미입니다.
		// - CPU에게 GPU가 일을 다 끝날때까지 강제로 기다리게 하고 싶다면 아래의 
		// - cudaDeviceSynchronize()를 사용할 수 있습니다.
		// - 함수 이름에서 볼 수 있듯이, 이렇게 기다리는 것을 "동기화(synchronize)"라고 합니다.

		cudaDeviceSynchronize();       // kernel이 끝날때까지 대기 (동기화) <== 병렬처리에서는 Synchronize를 잘 다루는게 아주 중요함!

		// 안내: kernel 실행 후 cudaGetLastError() 생략하였습니다.

		// 결과 복사 device -> host
		// 계산이 끝난 결과를 CPU로 복사
		cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		// GPU 메모리 해제 free
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		cudaDeviceReset();
	}

	// 결과 화면에 출력
	if (size < 40) { // size가 작을 경우에는 출력해서 확인
		printVector(a);
		printVector(b);
		printVector(cSingle);
		printVector(c);
	}

	// 계산이 제대로 됬는지 확인
	for (int i = 0; i < size; i++)
		if (cSingle[i] != c[i])
		{
			cout << "Wrong result" << endl;
			return 1;
		}

	cout << "Correct" << endl;

	return 0;
}
