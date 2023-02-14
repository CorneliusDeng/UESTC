#include <iostream>
#include "CPluginController.h"
#include <stdlib.h>
#include <string.h>

using namespace std;

int main(int argc, char **argv)
{
	// 命令行参数是两个
	if(argc == 2)
	{
		if(strcmp(argv[1], "help") == 0)
		{
			CPluginController pc;

			pc.ProcessHelp();

			return 0;
		}
		else
		{
			// atoi函数把字符串转换成整型数，获取功能号
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
	}

	// 命令行参数是三个
	else if(argc == 3)
	{
		CPluginController pc;

		char *Function = argv[1];

		// 操作的文件名
		char *Document = argv[2];

		pc.InitializeController();

		// 判断插件是否存在
		if(pc.IfProcess(Function) == false)
		{
			cout << "No this plugin!" << endl;
		}
		else
		{
			pc.ProcessFunction(Function,Document);
		};

		pc.UninitializeController();
		return 0;
	}
	else
	{
		cout << "Parameters error" << endl;
		return 0;
	}
}