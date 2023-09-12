#include <iostream>

using namespace std;

struct SLAugend
{
    int iAugend;
};

int Add(struct SLAugend *pSLAugend, int iAddend)
{
    return pSLAugend->iAugend + iAddend;
}

int main()
{
    struct SLAugend augend;
    
    augend.iAugend = 2;
    
    cout << Add(&augend, 5) << endl;

    return 0;
}
