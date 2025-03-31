#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
	// 부동소수점 더하기에서는 상대적으로 아주 큰 숫자와 작은 숫자를 더하면 작은 숫자를 무시합니다.
	// 아래 예시는 더하는 순서에 따라 결과가 달라지는 사례를 보여줍니다.

	{
		float sum = 1.0e8f;
		for (int i = 0; i < 10000; i++)
			sum += 1.0f; // 계속 무시됨

		cout << std::setprecision(20) << "Sum: " << sum << endl; // Sum: 100000000
	}

	{
		float sum = 0.0f;
		for (int i = 0; i < 10000; i++)
			sum += 1.0f; // 작은 숫자들끼리 먼저 더해지기 때문에 제대로 더해져서 합이 커짐

		sum += 1.0e8f;

		cout << std::setprecision(20) << "Sum: " << sum << endl; // Sum: 100010000
	}
}

