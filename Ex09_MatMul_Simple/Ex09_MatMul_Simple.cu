#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <functional>
#include <chrono>
#include <string>

using namespace std;

void timedRun(const string name, const function<void()> &func) {
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

// CUDA C++ Programming Guide 3.2.4 Shared Memory
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=matrix%20multiply#shared-memory

// **CUDA에서도 구조체를 사용할 수 있습니다.** 메모리에 가로로 한 줄씩 나열되어 있는 구조입니다.
// Matrices are stored in row-major order: M(row, col) = *(M.elements + row * M.width + col)
struct Matrix {
    int height = 0; // Number of rows
    int width = 0;  // Number of columns
    float *elements = nullptr;

    void Print() {
        assert(elements);

        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++)
                printf("%.1f ", elements[c + width * r]);
            printf("\n");
        }
        printf("\n");
    }
};

void MatMulCpu(const Matrix &A, const Matrix &B, Matrix &C) {
    int M = A.height;
    int K = A.width;
    int N = B.width;
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++) {
            float Cvalue = 0.0f;
            for (int e = 0; e < K; e++)
                Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
            C.elements[row * C.width + col] = Cvalue;
        }
}

// C = A * B
// 하나의 쓰레드가 C에 저장될 값을 하나씩 맡아서 계산하는 방식
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    float Cvalue = 0;

    // TODO: CPU 버전 참고하세요.
    // A의 가로줄 하나 * B의 세로줄 하나 => 각 원소들 곲해서 더하면 => C의 point 하나 값
    // 따라서, A의 가로줄의 요소개수 == B의 세로줄의 요소 개수 임
    // C.elements[row * C.width + col] = Cvalue;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;

    printf("%u %u %u %u %u %u", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x,
           threadIdx.y);
}

int main() {
    int M = 1024 * 2; // Num cols of C
    int N = 1024;     // Num rows of C
    int K = 512 * 2;
    // int K = 1024 * 8;

    // 프로파일러 안에서 10.486s

    Matrix A{M, K}; // M x K
    Matrix B{K, N}; // K x N
    Matrix C{M, N}; // AB = C, (M x K 행렬) x (K x N) 행렬 = M x N 행렬
    Matrix C_cpu{M, N};

    A.elements = new float[A.width * A.height];
    for (int i = 0; i < A.width * A.height; i++)
        A.elements[i] = float(rand() % 10);

    B.elements = new float[B.width * B.height];
    for (int i = 0; i < B.width * B.height; i++)
        B.elements[i] = float(rand() % 10);

    C.elements = new float[C.width * C.height];
    for (int i = 0; i < C.width * C.height; i++)
        C.elements[i] = 0.0f; // 디버깅 편의

    C_cpu.elements = new float[C_cpu.width * C_cpu.height];
    {
        timedRun("MatMulCpu", [&]() {
            MatMulCpu(A, B, C_cpu); // 4500ms 근처
        });
    }

    // B.elements에 접근할때 k가 하나 커지면 B.width 만큼 멀어진 곳에서 가져와야 합니다.
    // C의 값 하나를 계산하는 것만 보면 A와 B의 값들을 한 번씩만 사용한 것으로 보이지만,
    // C의 모든 값들에 대해서 보면 A와 B의 값들을 여러 번 사용하게 됩니다.
    // - A의 한 값은 B의 width(num_cols)번 만큼 사용,
    // - B의 한 값은 A의 height(num_rows)번 만큼 사용

    {
        Matrix d_A{M, K}; // M x K
        Matrix d_B{K, N}; // K x N
        Matrix d_C{M, N}; // AB = C, (M x K 행렬) x (K x N) 행렬 = M x N 행렬

        cudaMalloc(&d_A.elements, d_A.width * d_A.height * sizeof(float));
        cudaMalloc(&d_B.elements, d_B.width * d_B.height * sizeof(float));
        cudaMalloc(&d_C.elements, d_C.width * d_C.height * sizeof(float));
        cudaMemcpy(d_A.elements, A.elements, d_A.width * d_A.height * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_B.elements, B.elements, d_B.width * d_B.height * sizeof(float),
                   cudaMemcpyHostToDevice);

        // 가로방향으로도 쓰래드를 32개 구동시키고, 세로방향으로도 쓰래드를 32개 구동시킴
        // dimBlock 이 아래에서.. 쓰래드 개수로 들어감
        dim3 dimBlock(32, 32, 1); // dimBlock.x * dimBlock.y * dimBlock.z <= 1024 (== 32 x 32)
        // 따라서, 결국, 1블럭당 1024개 쓰래드가 도는 것음

        dim3 dimGrid(d_C.width / dimBlock.x, d_C.height / dimBlock.y); // 나머지가 없다고 가정

        timedRun("MatMul(Simple)", [&]() {
            MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); // 2.9ms 근처
        });

        cudaDeviceSynchronize(); // kernel이 끝날때까지 대기 (동기화)

        // 안내: kernel 실행 후 cudaGetLastError() 생략

        // 결과 복사 device -> host
        cudaMemcpy(C.elements, d_C.elements, d_C.width * d_C.height * sizeof(float),
                   cudaMemcpyDeviceToHost);

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
        if (C.elements[i] != C_cpu.elements[i]) {
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
