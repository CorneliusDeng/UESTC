#ifndef CPLUGINCONTROLLER_H
#define CPLUGINCONTROLLER_H
#include <vector>

class IPrintPlugin;

class CPluginController
{
public:

	CPluginController(void);

	virtual ~CPluginController(void);

	bool InitializeController(void);

	bool UninitializeController(void);

	bool ProcessHelp(void);

	bool ProcessRequest(int FunctionID);

	bool IfProcess(char *Function);

	bool ProcessFunction(char *Function,char *Document);
	
private:
	
	std::vector<void *> m_vhForPlugin;
	std::vector<IPrintPlugin*> m_vpPlugin;

};

#endif
