#include "CPluginController.h"
#include "CPluginEnumerator.h"
#include "dlfcn.h"

CPluginController::CPluginController(void)
{
}

CPluginController::~CPluginController(void)
{
}

// 初始化
bool CPluginController::InitializeController(void)
{
	std::vector<std::string> vstrPluginNames;
	
	// 打开事先约定的插件目录，逐项读取目录项
	CPluginEnumerator enumerator;
	if(!enumerator.GetPluginNames(vstrPluginNames))
		return false;

	for(unsigned int i=0 ; i<vstrPluginNames.size(); i++)
	{
		// 打开动态链接库
		void *hinstLib = dlopen(vstrPluginNames[i].c_str(), RTLD_LAZY);
		if(hinstLib != NULL) 
		{ 
			// push_back() 在Vector最后添加一个元素（参数为要插入的值）
			m_vhForPlugin.push_back(hinstLib);
			
			// 映射动态链接库中的函数
			PROC_PRINT DllPrint = (PROC_PRINT)dlsym(hinstLib, "Print");
			PROC_GETID DllGetID = (PROC_GETID)dlsym(hinstLib, "GetID");
			if((NULL != DllPrint) && (NULL != DllGetID))
			{
				m_vPrintFunc.push_back(DllPrint);
				m_vGetIDFunc.push_back(DllGetID);
			}
		}
	}

	return true;
}

// 处理请求
bool CPluginController::ProcessRequest(int FunctionID)
{
	for(unsigned int i = 0; i < m_vGetIDFunc.size(); i++)
	{
		if((m_vGetIDFunc[i])() == FunctionID)
		{
			(m_vPrintFunc[i])();
			break;
		}
	}

	return true;
}

// 提示输出 
bool CPluginController::ProcessHelp(void)
{
	std::vector<std::string> vstrPluginNames;

	CPluginEnumerator enumerator;
	if(!enumerator.GetPluginNames(vstrPluginNames))
		return false;

	for(unsigned int i=0 ; i<vstrPluginNames.size(); i++)
	{
		PROC_HELP DllProc; 

		void *hinstLib = dlopen(vstrPluginNames[i].c_str(), RTLD_LAZY);
		if(hinstLib != NULL) 
		{ 
			DllProc = (PROC_HELP)dlsym(hinstLib, "Help"); 
			if(NULL != DllProc) 
			{
				(DllProc)();
			}

			dlclose(hinstLib);
		}
	}

	return true;
}

// 卸载动态链接库
bool CPluginController::UninitializeController()
{
	for(unsigned int i = 0; i < m_vhForPlugin.size(); i++)
	{
		dlclose(m_vhForPlugin[i]);
	}

	return true;
}