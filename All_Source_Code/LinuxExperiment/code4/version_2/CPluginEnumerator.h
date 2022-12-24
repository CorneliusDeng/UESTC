#ifndef CPLUGINENUMERATOR_H
#define CPLUGINENUMERATOR_H

#include <vector>
#include <string>

using namespace std;

class CPluginEnumerator
{
public:
    CPluginEnumerator();
    virtual ~CPluginEnumerator();

    bool GetPluginNames(vector<string>& vstrPluginNames);
};

#endif
