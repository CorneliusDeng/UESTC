#include <iostream>

using namespace std;

class CLAdder
{
public:
    explicit CLAdder(int iAugend)
    {
	m_iAugend = iAugend;
    }

    virtual ~CLAdder()
    {
    }

    virtual int Add(int iAddend)
    {
	return m_iAugend + iAddend;
    }

protected:
    int m_iAugend;
};

class CLWeightingAdder : public CLAdder
{
public:
    CLWeightingAdder(int iAugend, int iWeight) : CLAdder(iAugend)
    {
	m_iWeight = iWeight;
    }

    virtual ~CLWeightingAdder()
    {
    }

    virtual int Add(int iAddend)
    {
	return m_iAugend * m_iWeight + iAddend;
    }

protected:
    int m_iWeight;
};

int main()
{
    CLAdder adder(2);
    cout << adder.Add(4) << endl;

    CLWeightingAdder wadder(3, 4);
    cout << wadder.Add(4) << endl;
   
    return 0;
}
