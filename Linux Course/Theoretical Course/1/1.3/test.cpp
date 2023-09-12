#include <iostream>

using namespace std;

class CLAdder
{
public:
    explicit CLAdder(int iAugend)
    {
	m_iAugend = iAugend;
    }

    int Add(int iAddend)
    {
	return m_iAugend + iAddend;
    }

private:
    int m_iAugend;
};

int main()
{
    CLAdder adder(2);
    cout << adder.Add(4) << endl;
   
    return 0;
}
