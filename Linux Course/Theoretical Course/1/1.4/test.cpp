#include <iostream>

using namespace std;

class CLWeightingAdder
{
public:
    CLWeightingAdder(int iAugend, int iWeight)
    {
	m_iAugend = iAugend;
	m_iWeight = iWeight;
    }

    int Add(int iAddend)
    {
	return m_iAugend * m_iWeight + iAddend;
    }

private:
    int m_iAugend;
    int m_iWeight;
};

int main()
{
    CLWeightingAdder adder(2, 3);
    cout << adder.Add(5) << endl;

    return 0;
}
