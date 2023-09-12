#ifndef CPLUGINCONTROLLER_H
#define CPLUGINCONTROLLER_H

#include <vector>

typedef void (*PROC_PRINT)(void); 
typedef void (*PROC_HELP)(void); 
typedef int (*PROC_GETID)(void); 

class CPluginController
{
public:
	CPluginController(void);
	virtual ~CPluginController(void);
	
	bool InitializeController(void);
	bool UninitializeController(void);

	bool ProcessHelp(void);
	bool ProcessRequest(int FunctionID);

private:
	std::vector<void *> m_vhForPlugin;
	std::vector<PROC_PRINT> m_vPrintFunc;
	std::vector<PROC_GETID> m_vGetIDFunc;
};

#endif
