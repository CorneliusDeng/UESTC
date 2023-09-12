#include <iostream>

using namespace std;

struct SLAdder;

typedef int (*FUNC_ADD)(struct SLAdder *, int);

struct SLAdder
{
    int iAugend;
    FUNC_ADD pFuncAdd;
};

int Add(struct SLAdder *pSLAdder, int iAddend)
{
    return pSLAdder->iAugend + iAddend;
}

int main()
{
    struct SLAdder adder;
    adder.pFuncAdd = Add;
    
    adder.iAugend = 3;
    
    cout << adder.pFuncAdd(&adder, 5) << endl;

    return 0;
}
