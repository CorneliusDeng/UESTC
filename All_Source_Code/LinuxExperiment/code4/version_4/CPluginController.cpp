#include "CPluginController.h"
#include "CPluginEnumerator.h"
#include "IPrintPlugin.h"
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
		typedef int (*PLUGIN_CREATE)(IPrintPlugin**);
		PLUGIN_CREATE CreateProc; 

		IPrintPlugin *pPlugin = NULL;

		// 打开动态链接库
		void* hinstLib = dlopen(vstrPluginNames[i].c_str(), RTLD_LAZY); 

		if(hinstLib != NULL) 
		{ 
			// push_back() 在Vector最后添加一个元素（参数为要插入的值）
			m_vhForPlugin.push_back(hinstLib);

			// 映射动态链接库中的函数，CreateObj是唯一的接口函数
			CreateProc = (PLUGIN_CREATE)dlsym(hinstLib, "CreateObj"); 

			if(NULL != CreateProc) 
			{
				(CreateProc)(&pPlugin);

				if(pPlugin != NULL)
				{
					m_vpPlugin.push_back(pPlugin);
				}
			}
		}
	}

	return true;
}

// 处理请求
bool CPluginController::ProcessRequest(int FunctionID)
{
	for(unsigned int i = 0; i < m_vpPlugin.size(); i++)
	{
		if(m_vpPlugin[i]->GetID() == FunctionID)
		{
			m_vpPlugin[i]->Print();
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
		typedef int (*PLUGIN_CREATE)(IPrintPlugin**);
		PLUGIN_CREATE CreateProc; 

		IPrintPlugin *pPlugin = NULL;

		void* hinstLib = dlopen(vstrPluginNames[i].c_str(), RTLD_LAZY); 

		if(hinstLib != NULL) 
		{ 
			CreateProc = (PLUGIN_CREATE)dlsym(hinstLib, "CreateObj"); 

			if(NULL != CreateProc) 
			{
				(CreateProc)(&pPlugin);

				if(pPlugin != NULL)
				{
					pPlugin->Help();
				}
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