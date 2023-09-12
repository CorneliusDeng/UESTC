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

struct SLWeightingAugend
{
    int iAugend;
    int iWeight;
};

int WeightingAdd(struct SLWeightingAugend *pSLWeightingAugend, int iAddend)
{
    return pSLWeightingAugend->iWeight * pSLWeightingAugend->iAugend + iAddend;
}

int main()
{
    struct SLWeightingAugend augend;
    
    augend.iAugend = 2;
    augend.iWeight = 3;
    
    cout << WeightingAdd(&augend, 5) << endl;

    return 0;
}
