#include <iostream>
#include "IPrintPlugin.h"
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// 统计某文件的字节数
using namespace std;

const int FUNC_ID = 2;

char FUNC_NAME[]="2";

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
			cout << "statistics the document words!" << endl;
		}

		virtual void Help()
		{
			cout << "Function ID " << FUNC_ID << " : This function will statistics the document words." << endl;
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
			long num=0; 

			// 打开文件
			if((fp=open(Document,O_RDONLY))==-1)
			{
				cout<<"Can not open: "<<Document<<endl;
				return ;
			};
			
			while(read(fp,&temp,1))
			{
				num++;
			};

			close(fp);

			if(num == 0)
			{
				cout << "Empty file: " << Document<< endl;
				return ;
			};

			cout << "Counting " << Document << ", the number of words is: "<< num << endl;
		};

};

// 仅导出一个接口函数CreateObj
extern "C" void CreateObj(IPrintPlugin **ppPlugin)
{
	static CPrintPlugin plugin;
	*ppPlugin = &plugin;

}
