#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

// https://developer.nvidia.com/thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

using namespace std;

void timedRun(const string name, const function<void()> &func)
{
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

void printArray(vector<float> &arr)
{
    for (int i = 0; i < std::min(int(arr.size()), 64); i++)
        cout << setw(3) << int(arr[i]);
    if (arr.size() > 64)
        cout << " ...";
    cout << endl;
}

void printArrayMiddle(const vector<float> &arr, int count = 32)
{
    int mid = arr.size() / 2;
    int start = max(0, mid - count / 2);
    int end = min(int(arr.size()), start + count);

    cout << "[중간 인덱스 출력]" << endl;
    for (int i = start; i < end; ++i)
        cout << setw(3) << int(arr[i]);
    cout << endl;
}

void printArrayHeadTail(const vector<float> &arr, int count = 32)
{
    cout << "[앞부분]" << endl;
    for (int i = 0; i < min(int(arr.size()), count); ++i)
        cout << setw(3) << int(arr[i]);
    cout << endl;

    cout << "[뒷부분]" << endl;
    for (int i = max(0, int(arr.size()) - count); i < arr.size(); ++i)
        cout << setw(3) << int(arr[i]);
    cout << endl;
}

void printHistogram(const vector<float> &arr)
{
    const int bucketCount = 100;
    vector<int> hist(bucketCount, 0);

    for (float v : arr)
    {
        int idx = int(v);
        if (idx >= 0 && idx < bucketCount)
            hist[idx]++;
    }

    cout << "[히스토그램]" << endl;
    for (int i = 0; i < bucketCount; ++i)
    {
        if (hist[i] > 0)
        {
            cout << setw(2) << i << ": " << hist[i] << endl;
        }
    }
}

void printHistogramVisual(const vector<float> &arr, int maxBarLen = 50)
{
    const int bucketCount = 100;
    vector<int> hist(bucketCount, 0);

    for (float v : arr)
    {
        int idx = int(v);
        if (idx >= 0 && idx < bucketCount)
            hist[idx]++;
    }

    int maxCount = *max_element(hist.begin(), hist.end());

    cout << "[랜덤 값 분포 시각화]" << endl;
    for (int i = 0; i < bucketCount; ++i)
    {
        if (hist[i] > 0)
        {
            int barLen = int((float(hist[i]) / maxCount) * maxBarLen);
            cout << setw(2) << i << ": " << string(barLen, '#') << " (" << hist[i] << ")" << endl;
        }
    }
}

int main()
{

    const uint64_t size = 1024 * 1024 * 32;

    vector<float> arr(size);

    srand(uint32_t(time(nullptr)));
    for (int i = 0; i < size; ++i)
    {
        arr[i] = (float)(rand() % 100);
    }

    cout << "[정렬 전 원본 배열]" << endl;
    printArray(arr);         // 앞부분만
    printArrayMiddle(arr);   // 중간값
    printArrayHeadTail(arr); // 앞뒤
    // printHistogram(arr);       // 숫자별 빈도
    // printHistogramVisual(arr); // 텍스트 히스토그램

    // Reduce Sum 예제
    {
        thrust::host_vector<float> h_vec(arr);
        thrust::device_vector<float> d_vec = h_vec;

        float sum = 0.0f;

        timedRun("ReduceSum", [&]()
                 { sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>()); }); // 4 ms 근처

        std::cout << "Sum: " << sum << std::endl;
    }

    // 정렬 예제
    {
        auto temp = arr;

        timedRun("CPU Sorting", [&]()
                 {
                     // CPU 정렬은 std::sort()로, std::vector는 std::sort!
                     std::sort(temp.begin(), temp.end()); // thrust::sort(temp.begin(), temp.end());
                 });                                      // 290 ms 근처

        cout << "[CPU Sorted 결과]" << endl;
        printArray(temp);         // 앞부분만
        printArrayMiddle(temp);   // 중간값
        printArrayHeadTail(temp); // 앞뒤
        // printHistogram(temp);       // 숫자별 빈도
        // printHistogramVisual(temp); // 텍스트 히스토그램

        thrust::host_vector<float> h_vec(arr);
        thrust::device_vector<float> d_vec = h_vec;

        timedRun("GPU Sorting", [&]()
                 { thrust::sort(d_vec.begin(), d_vec.end()); }); // 10 ms 근처

        thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
        std::copy(h_vec.begin(), h_vec.end(), arr.begin());

        cout << "[CUDA Sorted 결과]" << endl;
        printArray(arr);         // 앞부분만
        printArrayMiddle(arr);   // 중간값
        printArrayHeadTail(arr); // 앞뒤
        // printHistogram(arr);       // 숫자별 빈도
        // printHistogramVisual(arr); // 텍스트 히스토그램

        if (std::equal(temp.begin(), temp.end(), arr.begin()))
            cout << "O: CPU와 GPU 정렬 결과가 동일합니다!" << endl;
        else
            cout << "X: CPU와 GPU 정렬 결과가 다릅니다!" << endl;
    }

    return 0;
}
