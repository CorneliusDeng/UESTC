#include <iostream>

using namespace std;

template<typename T>
class CLAdder : public T
{
public:
    CLAdder() : T()
    {
    }

    virtual ~CLAdder()
    {
    }

    int Add(int iAddend)
    {
	T *pThis = (T *)(this);
	return pThis->AddImpl(iAddend);
    }
};

class CLNormalImpl
{
public:
    CLNormalImpl()
    {
	m_iAugend = 0;
    }

    void Set(unsigned int iAugend)
    {
	m_iAugend = iAugend;
    }

    virtual ~CLNormalImpl()
    {
    }

    int AddImpl(int iAddend)
    {
	return m_iAugend + iAddend;
    }

private:
    unsigned int m_iAugend;
};


class CLWeightingImpl
{
public:
    CLWeightingImpl()
    {
	m_iWeight = 0;
	m_iAugend = 0;
    }

    void Set(int iAugend, int iWeight)
    {
	m_iWeight = iWeight;
	m_iAugend = iAugend;
    }

    virtual ~CLWeightingImpl()
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
void f(CLAdder<T> *pAdder)
{
    cout << pAdder->Add(4) << endl;
}

int main()
{
    CLAdder<CLNormalImpl> adder;
    adder.Set(2);
    f(&adder);

    CLAdder<CLWeightingImpl> wadder;
    wadder.Set(3, 4);
    f(&wadder);
   
    return 0;
}
