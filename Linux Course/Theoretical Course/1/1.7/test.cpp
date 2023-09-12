#include <iostream>

using namespace std;

class ILAdder
{
public:
    ILAdder()
    {
    }

    virtual ~ILAdder()
    {
    }

    virtual int Add(int iAddend) = 0;
};

class CLAdder : public ILAdder
{
public:
    explicit CLAdder(unsigned int iAugend)
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

private:
    unsigned int m_iAugend;
};

class CLWeightingAdder : public ILAdder
{
public:
    CLWeightingAdder(int iAugend, int iWeight)
    {
	m_iWeight = iWeight;
	m_iAugend = iAugend;
    }

    virtual ~CLWeightingAdder()
    {
    }

    virtual int Add(int iAddend)
    {
	return m_iAugend * m_iWeight + iAddend;
    }

private:
    int m_iAugend;
    int m_iWeight;
};

void f(ILAdder *pAdder)
{
    cout << pAdder->Add(4) << endl;
}

int main()
{
    CLAdder adder(2);
    f(&adder);

    CLWeightingAdder wadder(3, 4);
    f(&wadder);
   
    return 0;
}
