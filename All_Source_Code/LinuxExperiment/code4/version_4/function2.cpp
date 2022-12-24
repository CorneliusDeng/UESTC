#include <iostream>
#include "IPrintPlugin.h"

using namespace std;

const int FUNC_ID = 2;

class CPrintPlugin : public IPrintPlugin
{
	public:
		CPrintPlugin()
		{
		}

		virtual ~CPrintPlugin()
		{
		}

		virtual void Print()
		{
			cout << "Hello China!" << endl;
		}

		virtual void Help()
		{
			cout << "Function ID " << FUNC_ID << " : This function will print hello china." << endl;
		}

		virtual int GetID(void)
		{
			return FUNC_ID;
		}
};

extern "C" void CreateObj(IPrintPlugin **ppPlugin)
{
	static CPrintPlugin plugin;
	*ppPlugin = &plugin;
}
