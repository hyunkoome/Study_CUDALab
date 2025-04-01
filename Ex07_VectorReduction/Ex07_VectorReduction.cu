#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <functional>
#include <string>

using namespace std;

// 참고 자료
// - https://github.com/umfranzw/cuda-reduction-example/tree/master/reduce0
// - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

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
    // 보통 CPU 시간이 크게 나오는데, CPU가 GPU한테 일을 시키는 시간을 포함하기때문임

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 교재: "Programming Massively Parallel Processors: A Hands-on Approach" 4th
// 벡터 reduction 의 의미는 , 입력 행렬이 1개이든, 100개 이든, 1000만개든, 1개로 만들기에 reduction
// 이라고하며 여기서는, sum reduction 모두 더해서 1개로 만드는 문제..
// 최대값을 찾거나, 최소값을 찾거나 하는 것들도 reduction 문제임
//
__global__ void atomicSumReductionKernel(float *input, float *output) {

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // output[0] += input[i]; // <- 여러 쓰레드가 경쟁적으로 메모리에 접근하기 때문에 오류 발생

    /*
    * 두 개의 쓰레드가 한 메모리 공간을 두고 경쟁(racing) 하는 사례
    * 원래 값 1에다가 쓰레드 2개가 각각 1씩 더해서 3이 되어야 하는 경우

     1. 시리얼로 하나씩 더할 때
     - 처음에 저장되어 있는 값 output[0] = 1
     - thread 0이 output[0]의 1을 읽어옴
     - thread 0이 읽어온 1에다가 1을 더함
     - thread 0이 output[0]에다가 2를 저장
     - thread 1이 output[0]의 2를 읽어옴
     - thread 1이 읽어온 2에다가 1을 더함
     - thread 1이 output[0]에다가 3을 저장
     - 결과적으로 output[0]은 3

     2. 멀티쓰레딩으로 하나씩 더할 때 문제가 생기는 경우
     - output[0] = 1
     - thread 0: 1 read
     - thread 1: 1 read (thread 0이 write 하기 전에 thread 1이 읽어옴)
     - thread 0: output[0] <- 1 + 1 저장
     - thread 1: output[0] <- 1 + 1 저장 (앞서 thread 0이 저장한 output[0]은 2이지만 thread 1은 알지
    못함)

    멀티쓰레딩을 사용하면 경우에 따라서 문제가 생기지 않을 수도 있습니다. 그래서 정상작동하는 것으로
    착각하는 경우도 많습니다. 내가 구현하는 방식에 메모리 접근 경쟁이 생기는지 아닌지를 항상
    주의해야 합니다.
    */

    // TODO; // <- atomicAdd()로 정확하게 계산하지만 지나치게 느려집니다.
    //    여러개의 쓰레드들이 자기 차례를 기다려야 하기 때문입니다.
    atomicAdd(output, input[i]);
    // 정확하게 계산은 하지만, 모든 덧셈연산을 atomic 으로 하게되면 매우 느려짐
    // 그래서, 정확도를 유지하면서, 빠르게 처리하려면? => 블럭 단위로 처리함!!
}

// 같은 블럭의 쓰래드끼리는 서로 순서를 정할수 있다는 성질을 이용해보자!!
// 블럭이 하나일 때만 사용 가능
// 엔비디아 슬라이드 그림 속의 Values는 여기서 input 입니다.
// for문 안의 각 단계에서 입력 배열(input)에 덮어쓰면서 reduce 해 나가다가
// 마지막 하나만 output에 저장합니다.
__global__ void convergentSumReductionKernel(float *input,
                                             float *output) { // block 하나로 처리가능한 크기
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            // TODO: i를 사용해도 되고 threadIdx.x를 직접 사용해도 됩니다.
            input[i] += input[i + stride];
        }
        // 이 부분이 중요!
        // __syncthreads() 를 사용하면,
        // 한 블럭안에 있는 모든 쓰레드들이 일을 다 끝날때까지 멈췄다가 진행할 수 있음
        // 즉, 동기화 가능
        __syncthreads(); // <- 같은 블럭 안에 있는 쓰레드들 동기화
    }
    if (threadIdx.x == 0)
        *output = input[0];
}

__global__ void sharedMemorySumReductionKernel(float *input, float *output) {

    // 쉐어드 메모리(양이 작지만, 빠름)는 블럭단위로, 같은 블럭내에서 같은 쉐어드 메모리(L1 캐쉬)
    // 사용 가능 이때, 스레드가 총 1024개.
    // 그래서, 자주 사용하는 데이터는 쉐어드 메모리로 옮겨서..!!

    // inputShared 배열의 크기가 빠져있는 이유는, 함수 호출할때 넣어준 공유메모리
    // 크기만큼..공유메모리가 배정이 됨
    extern __shared__ float inputShared[]; // <- 블럭 안에서 여러 쓰레드들이 공유하는 빠른 메모리

    // 경우에 따라서는, 이렇게 수동으로 사이즈를 지정할 수도 있음
    // __shared__ float inputShared[1024];

    unsigned int t = threadIdx.x;

    // input[t], input[t + blockDim.x]: 글로벌 메모리
    // inputShared[t]: 쉐어드 메모리
    inputShared[t] = input[t] + input[t + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        // 바로 위에서 inputShared[t] = input[t] + input[t + blockDim.x]; 로
        // 공유메모리를 사용했기때문에, for 문 들어오자마자, 동기화(__syncthreads) 시켜 줌
        __syncthreads();

        if (threadIdx.x < stride) {
            // TODO:
            inputShared[t] += inputShared[t + stride];
        }
    }
    if (t == 0)
        *output = inputShared[0];
}

// 각각의 블럭단위로 더한 합들을, 다시 한번 합쳐주는 것
__global__ void segmentedSumReductionKernel(float *input, float *output) {
    extern __shared__ float inputShared[];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // TODO: 위의 두 개를 잘 합치면 됩니다.
    // [힌트] atomicSumReductionKernel(),  sharedMemorySumReductionKernel() 두 개를 잘 합치면
    // 됩니다.
    inputShared[t] = input[i] + input[i + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            inputShared[t] += inputShared[t + stride];
        }
    }
    if (t == 0)
        atomicAdd(output, inputShared[0]);
}

int main(int argc, char *argv[]) {
    // 벡터 리덕션(reduction): 배열에 들어있는 모든 숫자들의 합을 구하는 것임

    const int size = 1024 * 1024 * 32;

    // 배열 만들기
    vector<float> arr(size);
    srand(uint32_t(time(nullptr)));
    for (int i = 0; i < size; i++)
        arr[i] = (float)rand() / RAND_MAX;

    // CPU에서 합 구하기: 쓰래드 하나를 이용을 해서 합을 구하기기
    float sumCpu = 0.0f;
    // timedRun 함수는 위에서 구현, 시간 측정 함수
    timedRun("CPU Sum", [&]() {
        for (int i = 0; i < size; i++) {
            sumCpu += arr[i];
        }
    });

    // 수치 에러로 인해, CPU로 싱글 쓰래드로 계산한것과, GPU로 멀티 쓰레드로 계산한 것이 완벽이 같지
    // 않을수도 있음

    // GPU 준비
    float *dev_input;
    float *dev_output;

    int threadsPerBlock = 1024;

    // cudaMalloc 으로 GPU 메모리를 할당 받으면, global 메모리(양이 크지만, 느림)임..
    // 따라서, 모든 블럭, 모든 쓰래드에서 접근 가능
    cudaMalloc(&dev_input, size * sizeof(float));
    cudaMalloc(&dev_output, sizeof(float));
    cudaMemcpy(dev_input, arr.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    timedRun("Atomic", [&]() {
        int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        atomicSumReductionKernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);
    }); // 68 ms

    // 주의: size = threadsPerBlock * 2 라고 가정 (블럭 하나만 사용)
    timedRun("GPU Sum", [&]() {
        convergentSumReductionKernel<<<1, threadsPerBlock>>>(dev_input, dev_output);
        // 블럭이 하나일 때만 사용
    });

    timedRun("GPU Sum", [&]() {
        sharedMemorySumReductionKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            dev_input, dev_output); // 블럭이 하나일 때만 사용
    });
    // 커널 함수 실행시, 파라미터로, 블럭수, 스레드수, '블럭 당 사용하는 공유메모리크기' 도 넣을 수
    // 있음 threadsPerBlock * sizeof(float): 블럭 당 사용하는 공유메모리 크기
    //  kernel<<<블럭수, 쓰레드수, 공유메모리크기>>>(...);

    // 최종 goal
    // 각각의 블럭단위로 더한 합들을, 다시 한번 합쳐주는 것
    timedRun("Segmented", [&]() {
        // int numBlocks = TODO; // size 나누기 2 주의
        // 블럭 여러 개 사용,
        // int numBlocks = ((size / 2) + threadsPerBlock - 1) / threadsPerBlock;
        // 즉, input[i] + input[i + blockDim.x] 를 수행하려면 스레드 하나당 2개의 입력을 처리해야
        // 하므로, 전체 size를 2로 나누고, 블럭당 스레드 수만큼 나눠야 함
        // segmentedSumReductionKernel 내부에서:
        // inputShared[t] = input[i] + input[i + blockDim.x];
        // 즉, 한 스레드가 2개의 요소(input[i], input[i + blockDim.x])를 더해서 shared memory로 복사
        // 따라서 전체 size 개의 입력을 처리하려면 size / 2 개의 스레드만 있으면 충분
        // 각 블럭에는 threadsPerBlock 개의 스레드가 있으므로:
        // numBlocks = (size / 2) / threadsPerBlock = size / (2 * threadsPerBlock)
        // 여기에, 올림처리를 위해,
        // numBlocks = ((size / 2) + threadsPerBlock - 1) / threadsPerBlock;
        // 따라서, numBlocks = ceil(size / (2 * threadsPerBlock)) 과 동일
        int numBlocks = ceil(size / (2 * threadsPerBlock));
        segmentedSumReductionKernel<<<numBlocks, threadsPerBlock,
                                      threadsPerBlock * sizeof(float)>>>(dev_input, dev_output);
    }); // 1 ms 근처

    float sumGpu = 0.0f;
    cudaMemcpy(&sumGpu, dev_output, sizeof(float), cudaMemcpyDeviceToHost); // 숫자 하나만 복사

    cout << "sumCpu = " << sumCpu << ", sumGpu = " << sumGpu << endl;
    cout << "Avg Error = " << std::abs((sumCpu - sumGpu)) / size << endl;

    cudaFree(dev_input);
    cudaFree(dev_output);

    return EXIT_SUCCESS;
}
