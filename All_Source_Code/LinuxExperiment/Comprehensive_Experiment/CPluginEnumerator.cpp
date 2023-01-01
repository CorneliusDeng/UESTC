#include "CPluginEnumerator.h"
#include <dirent.h>
#include <string.h>

CPluginEnumerator::CPluginEnumerator()
{
}

CPluginEnumerator::~CPluginEnumerator()
{
}

bool CPluginEnumerator::GetPluginNames(vector<string>& vstrPluginNames)
{
	// 打开目录
    DIR *dir = opendir("./plugin");
    if(dir == 0)
		return false;
    
    for(;;)
    {
		// 读取目录项
		struct dirent *pentry = readdir(dir);
		if(pentry == 0)
			break;

		// 用于比较两个字符串并根据比较结果返回整数
		if(strcmp(pentry->d_name, ".") == 0)
			continue;

		if(strcmp(pentry->d_name, "..") == 0)
			continue;

		string str = "./plugin/";
		str += pentry->d_name;
		vstrPluginNames.push_back(str);
    }

	// 关闭目录
    closedir(dir);

    return true;
}