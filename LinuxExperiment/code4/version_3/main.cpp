#include <iostream>
#include "CPluginController.h"
#include <string.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char **argv)
{
	// 运行main方法要带参数
	if(argc != 2)
	{
		cout << "Parameters error" << endl;
		return 0;
	}

	// 展示help内容
	if(strcmp(argv[1], "help") == 0)
	{
		CPluginController pc;
		pc.ProcessHelp();

		return 0;
	}

	// 获取功能号
	int FunctionID = atoi(argv[1]); 
	CPluginController pc;

	// 初始化
	pc.InitializeController();

	// 处理请求
	pc.ProcessRequest(FunctionID);

	// 卸载动态链接库
	pc.UninitializeController();

	return 0;
}
