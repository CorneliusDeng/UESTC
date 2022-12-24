#include <iostream>
#include <stdlib.h>



/*********************************************************************************
 Function: Body1、Body2、welcome
 Description：登录界面，功能选择
 Input parameter：option
 Out parameter：function
 Relevance：NULL
 Return: NULL
 Auhtor：Cornelius Deng
 DATE: 2022.12
**********************************************************************************/

void Body1()
{
	int choice = 1;

	do
	{
		system("cls");   
		cout <<"                          图书管理系统(管理员)                    " <<endl;
		cout <<"     ****************成都信息工程大学图书馆欢迎你*****************" << endl;
		cout <<"           *  1 图书采编入库               2 清除库存图书        *" << endl;
		cout <<"           *  3 查询图书信息               4 增删读者信息        *" << endl;
		cout <<"           *                  0 退出系统                         *" << endl;
		cout <<"     *************************************************************" << endl;
		cout <<"                     请输入您的需求功能编号:" << endl;
		cout <<"           *  5 查询读者信息               6 统计计算            *\n" << endl;
		cin << choice;
		switch(choice)	
		{
			case 1: Add(); break;
			case 2: Delete(); break;
			case 3: BooksInfo(); break;
			case 4: ReaderInfo(); break;
			case 5: SearchReaInfo(); break;
			case 6: Calculate();Read2(book, student, borrow);break;
			case 0: system("cls");Save(book, student, borrow);
			cout <<"\n\n\n\n\n\t所有数据已更新保存");
			cout <<"\n\n\n\n\n\t成都信息工程大学图书馆期待与您再次相遇");
			cout <<"\n\n\n\n\n\t^^^^^^^^^^感谢您的使用，再见^^^^^^^^^^！\n\n");
			system("PAUSE");
			exit(0);
			default :
				cout <<"\n\n\t\t输入有误！！\n");
				system("PAUSE");system("cls"); break;
		}
	}while(choice != 0);
}


void Body2()
{
	int choice = 1;
	do
	{
		system("cls");   
		printf("                           图书管理系统(读者)                     \n");
		printf("     ****************成都信息工程大学图书馆欢迎你*****************\n");
		printf("           *  1 图书检索                   2 借阅图书            *\n");
		printf("           *  3 归还图书                   4 查看榜单            *\n");
		printf("           *                0 退出系统                           *\n");
		printf("     *************************************************************\n");
		printf("                      请输入您的需求功能编号:");
		
		scanf("%d", &choice);

		switch(choice)
		{
			case 1: BooksInfo(); break;
			case 2: Borrow(); break;
			case 3: Return(); break;
			case 4: Calculate();break;
			case 0: system("cls");Save(book, student, borrow);
			cout<< "\n\n\n\n\n\t所有数据已更新保存" <<endl;
			cout << "\n\n\n\n\n\t成都信息工程大学图书馆期待与您再次相遇";
			cout << "\n\n\n\n\n\t^^^^^^^^^^感谢您的使用，再见^^^^^^^^^^" << endl;
			system("PAUSE");
			exit(0);
			default :
				cout<<"\n\n\t\t输入有误！！" <<endl;
				system("PAUSE");system("cls"); break;
		}
	}while(choice != 0);
}

void welcome()
{
	int option;
	
	printf("************************************************************\n");
	printf("             **欢迎进入成都信息工程大学图书馆**\n");
	printf("************************************************************\n");
	printf("                 **请对您的身份进行选择**\n");
	printf("************************************************************\n");
	printf("                        1:管理员\n");
	printf("                        2:读者\n");
	printf("                        3:退出\n");
	printf("************************************************************\n");
	printf("                        我的选择是：");
	scanf("%d", &option);
	printf("************************************************************\n");
	


	switch(option)
	{
		case 1:
			system("cls");Body1();break;
		case 2:
			system("cls");Body2();break;
		case 3:
			exit(0);
		default :
			printf("\n\n\t\t您的输入有误！！\n");
			system("PAUSE"); system("cls"); break;

	}
}

int main()
{
	BookNum = 0;
	StudentNum = 0;
	BorrowNum = 0;
	
	Read(book, student, borrow);
	welcome();

	return 0;
}