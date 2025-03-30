#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

template<typename T>
void printVector(const vector<T>& a)
{
	for (int v : a)
		cout << setw(3) << v;
	cout << endl;
}

int main()
{
	using clock = std::chrono::high_resolution_clock;

	//int size = 37;
	int size = 1024 * 1024 * 512;

	vector<int> a(size);
	vector<int> b(size);
	vector<int> cMulti(size);     // CPU에서 멀티쓰레딩으로 계산한 결과 저장
	vector<int> cSingle(size);  // 정답 확인용

	for (int i = 0; i < int(size); i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
		cSingle[i] = a[i] + b[i];
	}

	// CPU 멀티쓰레딩으로 벡터 더하기
	{
		auto addFunc = [&](int start, int end, int size)
			{
				//for (int i = start; i < end; i++)
				//	if (i < size)
				//		cMulti[i] = TODO;
				for (int i = start; i < end; i++)
					if (i < size)
						cMulti[i] = a[i] + b[i];
			};

		printf("Start multithreading\n");
		auto start = clock::now(); // 시간 측정 시작

		int numThreads = thread::hardware_concurrency();
		int perThread = int(ceil(float(size) / numThreads));

		vector<thread> threadList(numThreads);

		for (int r = 0; r < 100; r++) // 한 번으로는 CPU 사용량 관찰이 쉽지 않기 때문에 여러 번 반복
		{
			//for (int t = 0; t < numThreads; t++)
			//    threadList[t] = thread(addFunc, TODO, TODO, size);
			for (int t = 0; t < numThreads; t++)
				threadList[t] = thread(addFunc, t * perThread, (t + 1) * perThread, size);

			//for (int t = 0; t < numThreads; t++)
			//    threadList[t].join();
			for (int t = 0; t < numThreads; t++)
			   threadList[t].join();
			
		}

		printf("Time taken: %f ms\n", duration<float, milli>(clock::now() - start).count());
	}

	if (size < 40) { // 눈으로 봐서 확인 가능한 크기일 때는 출력
		printVector(a);
		printVector(b);
		printVector(cMulti);
		printVector(cSingle);
	}

	for (int i = 0; i < size; i++)
		if (cMulti[i] != cSingle[i]) {
			cout << "Wrong result" << endl;
			return 1;
		}

	cout << "Correct" << endl;

	return 0;
}
