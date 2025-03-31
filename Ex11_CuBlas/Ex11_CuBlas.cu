#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream> 
#include <iomanip>
#include <assert.h>
#include <cublas_v2.h>

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

// cublas.lib 추가
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64

// CUDA에서도 구조체를 사용할 수 있습니다. 메모리에 가로로 한 줄씩 나열되어 있는 구조입니다.
// Matrices are stored in row-major order: M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
	int height = 0; // Number of rows
	int width = 0;  // Number of columns
	int stride = 0; // 새로 추가!
	float* elements = nullptr;

	void Print()
	{
		assert(elements);

		// Column-Major Format
		for (int c = 0; c < width; c++)
		{
			for (int r = 0; r < height; r++)
				printf("%.1f ", elements[height * c + r]);
			printf("\n");
		}
		printf("\n");
	}
};

int main()
{
	// NumericalErrorTest();

	int M = 1024; // Num cols of C
	int N = 1024; // Num rows of C
	int K = 1024 * 8;

	//int M = 8; // Num cols of C
	//int N = 8; // Num rows of C
	//int K = 8;

	Matrix A{ M, K }; // M x K
	Matrix B{ K, N }; // K x N
	Matrix C{ M, N }; // AB = C, (M x K 행렬) x (K x N) 행렬 = M x N 행렬
	Matrix C_cpu{ M, N };

	A.stride = A.width; // 추가!
	A.elements = new float[A.width * A.height];
	for (int i = 0; i < A.width * A.height; i++)
		A.elements[i] = float(rand() % 10);

	B.stride = B.width; // 추가!
	B.elements = new float[B.width * B.height];
	for (int i = 0; i < B.width * B.height; i++)
			B.elements[i] = float(rand() % 10);

	C.stride = C.width; // 추가!
	C.elements = new float[C.width * C.height];
	for (int i = 0; i < C.width * C.height; i++)
		C.elements[i] = 0.0f; // 디버깅 편의

	// 주의: Column-Major Format
	C_cpu.elements = new float[C_cpu.width * C_cpu.height];
	for (int r = 0; r < M; r++)
		for (int c = 0; c < N; c++)
		{
			C_cpu.elements[c * C_cpu.height + r] = 0.0f;
			for (int k = 0; k < K; k++)
				C_cpu.elements[c * C_cpu.height + r] += A.elements[r + k * A.height] * B.elements[k + c * B.height];
			// C_cpu.elements[c + r * C_cpu.width] += A.elements[k + r * A.width] * B.elements[c + k * B.width]; // Row-Major format
		}

	{
		Matrix d_A{ M, K }; // M x K
		Matrix d_B{ K, N }; // K x N
		Matrix d_C{ M, N }; // AB = C, (M x K 행렬) x (K x N) 행렬 = M x N 행렬

		d_A.stride = d_A.width; // 추가!
		d_B.stride = d_B.width; // 추가!
		d_C.stride = d_C.width; // 추가!

		cudaMalloc(&d_A.elements, d_A.width * d_A.height * sizeof(float));
		cudaMalloc(&d_B.elements, d_B.width * d_B.height * sizeof(float));
		cudaMalloc(&d_C.elements, d_C.width * d_C.height * sizeof(float));
		cudaMemcpy(d_A.elements, A.elements, d_A.width * d_A.height * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B.elements, B.elements, d_B.width * d_B.height * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C.elements, C.elements, d_C.width * d_C.height * sizeof(float), cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;// 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0); // 시작 시간 기록

		//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1); // dimBlock.x * dimBlock.y * dimBlock.z <= 1024
		//dim3 dimGrid(d_C.width / dimBlock.x, d_C.height / dimBlock.y); // 나머지가 없다고 가정
		//MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

		cublasHandle_t handle;
		cublasCreate(&handle);

		const float alpha = 1.0f;
		const float beta = 0.0f;

		// Column-Major format
		// https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmex
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A.elements, M, d_B.elements, K, &beta, d_C.elements, M);

		// 안내: kernel 실행 후 cudaGetLastError() 생략

		// 결과 복사 device -> host
		cudaMemcpy(C.elements, d_C.elements, d_C.width * d_C.height * sizeof(float), cudaMemcpyDeviceToHost);

		cudaEventRecord(stop, 0);  // 끝나는 시간 기록

		cudaDeviceSynchronize();       // kernel이 끝날때까지 대기 (동기화)
		// cudaEventSynchronize(stop); // 불필요 (동기화 중복)

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
		cout << "Time elapsed: " << milliseconds << " ms" << endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(d_A.elements);
		cudaFree(d_B.elements);
		cudaFree(d_C.elements);

		cudaDeviceReset();
	}

	if (M <= 16 && N <= 16 && K <= 16) { // 행렬(matrix)이 작을 때는 출력해서 확인
		A.Print();
		B.Print();
		C_cpu.Print();
		C.Print();
	}

	for (int i = 0; i < C.width * C.height; i++)
		if (C.elements[i] != C_cpu.elements[i])
		{
			cout << "Wrong result" << endl;
			return 1;
		}

	cout << "Correct" << endl;

	delete[] A.elements;
	delete[] B.elements;
	delete[] C.elements;
	delete[] C_cpu.elements;

	return 0;
}
