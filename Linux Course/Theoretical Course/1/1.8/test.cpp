#include <iostream>

using namespace std;

template<typename T>
class ILAdder
{
public:
    ILAdder()
    {
    }

    virtual ~ILAdder()
    {
    }

    int Add(int iAddend)
    {
	T *pThis = (T *)(this);
	return pThis->AddImpl(iAddend);
    }
};

class CLAdder : public ILAdder<CLAdder>
{
public:
    explicit CLAdder(unsigned int iAugend)
    {
	m_iAugend = iAugend;
    }

    virtual ~CLAdder()
    {
    }

    int AddImpl(int iAddend)
    {
	return m_iAugend + iAddend;
    }

private:
    unsigned int m_iAugend;
};

class CLWeightingAdder : public ILAdder<CLWeightingAdder>
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

    int AddImpl(int iAddend)
    {
	return m_iAugend * m_iWeight + iAddend;
    }

private:
    int m_iAugend;
    int m_iWeight;
};

template<typename T>
void f(ILAdder<T> *pAdder)
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
