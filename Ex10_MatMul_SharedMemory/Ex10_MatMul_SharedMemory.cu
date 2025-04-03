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

#define BLOCK_SIZE 32 // GPU에 맞게 블럭사이즈를 조절하기!
// 매크로를 바꿔가면서 셰이더를 여러 개 만들어서 사용하기도 합니다

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

// CUDA에서도 구조체를 사용할 수 있습니다. 메모리에 가로로 한 줄씩 나열되어 있는 구조입니다.
// Matrices are stored in row-major order: M(row, col) = *(M.elements + row * M.width + col)
struct Matrix {
    int height = 0; // Number of rows
    int width = 0;  // Number of columns
    int stride = 0; // 새로 추가!
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
                Cvalue += A.elements[e + row * A.width] * B.elements[col + e * B.width];
            C.elements[row * C.width + col] = Cvalue;
        }
}

// __device__는 GPU 커널에서만 호출가능
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix C, int row, int col, float value) {
    C.elements[row * C.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    // 강의 그림에서는 (A.width / BLOCK_SIZE) == 2인 경우
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m); // Get sub-matrix Asub of A
        Matrix Bsub = GetSubMatrix(B, m, blockCol); // Get sub-matrix Bsub of B

        // 그래서, shared 메모리가 2개이고.
        // Asub => As 2차원 배열로, Bsub => Bs 2차원 배열로

        // A와 B의 작은 부분을 임시로 저장하는 공유(__shared__) 메모리
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; // 컴파일할때 BLOCK_SIZE가 결정되어 있어야 하기
                                                     // 때문에 매크로를 사용했습니다.
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // As[row][col] = GetElement(TODO, TDOO, TODO)
        // <-- 코드상으로는 마치 배열 하나를 복사하는 것 처럼 보이지만,
        // 쓰래드 개수가 블럭사이즈 32 x 블럭사이즈 32 이기 때문에,
        // A의 블럭을 하나 통채로 복사하고, B의 블럭을 하나 통채로 복사하는 것 처럼 실행이 됨
        // As[row][col] = GetElement(TODO, TDOO, TODO);
        // Bs[row][col] = GetElement(TODO, TODO, TODO);
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        // 계산하는 블럭당 매트릭스 연산을 하기전에, 싱크스래드 함수로, 동기화를 하고
        __syncthreads();

        // Multiply Asub and Bsub together
        // 실제로 곱해서 더하는 연산을 하고, 이때 블럭사이즈 만큼 루프를 돌고.
        // As 의 row 와 Bs의 colum 을 dot product 후 더해서 결과를 c 에 저장
        // for (int e = 0; e < BLOCK_SIZE; ++e)
        //    각 서브 블럭당 나온 c 결과를 누적해서 c에 더하면 최종 합이 됨
        //    Cvalue += TODO * TODO;
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

int main() {
    int M = 1024 * 2; // Num cols of C
    int N = 1024;     // Num rows of C
    int K = 512 * 2;
    // int K = 1024 * 8;

    Matrix A{M, K}; // M x K
    Matrix B{K, N}; // K x N
    Matrix C{M, N}; // AB = C, (M x K 행렬) x (K x N) 행렬 = M x N 행렬
    Matrix C_cpu{M, N};

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

    C_cpu.elements = new float[C_cpu.width * C_cpu.height];
    {
        timedRun("MatMulCpu", [&]() {
            MatMulCpu(A, B, C_cpu); // 4500ms 근처 (실행시킬때마다 차이가 꽤 큽니다)
        });

        // B.elements에 접근할때 k가 하나 커지면 B.width 만큼 멀어진 곳에서 가져와야 합니다.
        // C의 값 하나를 계산하는 것만 보면 A와 B의 값들을 한 번씩만 사용한 것으로 보이지만,
        // C의 모든 값들에 대해서 보면 A와 B의 값들을 여러 번 사용하게 됩니다.
        // - A의 한 값은 B의 width(num_cols)번 만큼 사용,
        // - B의 한 값은 A의 height(num_rows)번 만큼 사용
    }

    {
        Matrix d_A{M, K}; // M x K
        Matrix d_B{K, N}; // K x N
        Matrix d_C{M, N}; // AB = C, (M x K 행렬) x (K x N) 행렬 = M x N 행렬

        d_A.stride = d_A.width; // 추가!
        d_B.stride = d_B.width; // 추가!
        d_C.stride = d_C.width; // 추가!

        cudaMalloc(&d_A.elements, d_A.width * d_A.height * sizeof(float));
        cudaMalloc(&d_B.elements, d_B.width * d_B.height * sizeof(float));
        cudaMalloc(&d_C.elements, d_C.width * d_C.height * sizeof(float));
        cudaMemcpy(d_A.elements, A.elements, d_A.width * d_A.height * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_B.elements, B.elements, d_B.width * d_B.height * sizeof(float),
                   cudaMemcpyHostToDevice);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1); // dimBlock.x * dimBlock.y * dimBlock.z <= 1024
        dim3 dimGrid(d_C.width / dimBlock.x, d_C.height / dimBlock.y); // 나머지가 없다고 가정

        timedRun("MatMul(SharedMemory)", [&]() {
            MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); // 2.3ms 근처
        });

        // 안내: kernel 실행 후 cudaGetLastError() 생략

        // 결과 복사 device -> host
        cudaMemcpy(C.elements, d_C.elements, d_C.width * d_C.height * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize(); // kernel이 끝날때까지 대기 (동기화)
        // cudaEventSynchronize(stop); // 불필요 (동기화 중복)

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
