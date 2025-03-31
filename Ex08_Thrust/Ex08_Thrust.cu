#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

// https://developer.nvidia.com/thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

using namespace std;

void timedRun(const string name, const function<void()>& func) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	auto startCpu = chrono::high_resolution_clock::now(); // CPU 시간측정 시작
	cudaEventRecord(start, 0);                            // GPU 시간측정 시작

	func(); // 실행

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);                         // GPU 시간측정 종료
	auto endCpu = chrono::high_resolution_clock::now(); // CPU 시간측정 종료

	float elapsedGpu = 0;
	cudaEventElapsedTime(&elapsedGpu, start, stop);
	chrono::duration<float, milli> elapsedCpu = endCpu - startCpu;
	cout << name << ": CPU " << elapsedCpu.count() << " ms, GPU " << elapsedGpu << "ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// 참고 자료: https://github.com/rbga/CUDA-Merge-and-Bitonic-Sort/blob/master/BitonicMerge/kernel.cu

void printArray(vector<float>& arr) {
	for (int i = 0; i < std::min(int(arr.size()), 64); i++)
		cout << setw(3) << int(arr[i]);
	if (arr.size() > 64)
		cout << " ...";
	cout << endl;
}

int main() {

	const uint64_t size = 1024 * 1024 * 32;

	vector<float> arr(size);

	srand(uint32_t(time(nullptr)));
	for (int i = 0; i < size; ++i) {
		arr[i] = (float)(rand() % 100);
	}

	printArray(arr);

	// Reduce Sum 예제
	{
		thrust::host_vector<float> h_vec(arr);
		thrust::device_vector<float> d_vec = h_vec;

		float sum = 0.0f;

		timedRun("ReduceSum", [&]() {
			sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
			});  // 4 ms 근처

		std::cout << "Sum: " << sum << std::endl;
	}

	// 정렬 예제
	{
		auto temp = arr;

		timedRun("CPU Sorting", [&]() {
			thrust::sort(temp.begin(), temp.end());
			});  // 290 ms 근처

		thrust::host_vector<float> h_vec(arr);
		thrust::device_vector<float> d_vec = h_vec;

		timedRun("GPU Sorting", [&]() {
			thrust::sort(d_vec.begin(), d_vec.end());
			});  // 10 ms 근처

		thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
		std::copy(h_vec.begin(), h_vec.end(), arr.begin());

		printArray(arr); // 정렬된 결과 출력
	}

	return 0;
}
