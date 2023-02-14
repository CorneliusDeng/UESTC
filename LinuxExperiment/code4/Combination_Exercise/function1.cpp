#include <iostream>
#include "IPrintPlugin.h"
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// 统计某文件的行数
using namespace std;

const int FUNC_ID = 1;

char FUNC_NAME[]= "1";

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
			cout << "Statistics the document line!" << endl;
		}

		virtual void Help()
		{
			cout << "Function ID " << FUNC_ID << " : This function will statistics the document line." << endl;
		}

		virtual int GetID(void)
		{
			return FUNC_ID;
		}

		virtual char *GetName()
		{
			return FUNC_NAME;
		}

		virtual void Fun(char *Document)
		{

			int fp;
			char temp;
			long num=0;//统计

			// 打开文件
			if((fp=open(Document,O_RDONLY))==-1)
			{
				cout << "Can not open: " << Document << endl;
				return;
			};

			// 读取到换行，行数加1
			while(read(fp,&temp,1))
			{
				if(temp=='\n')
				{
					num++;
				}
			};

			close(fp);

			cout << "Counting " << Document << ", the number of lines is: " << num << endl;
		}
};

// 仅导出一个接口函数CreateObj
extern "C" void CreateObj(IPrintPlugin **ppPlugin)
{
	static CPrintPlugin plugin;
	*ppPlugin = &plugin;
}