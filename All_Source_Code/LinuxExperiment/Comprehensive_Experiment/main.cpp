#include "Statement.h"

void Body1()
{
	int choice = 1;
	char school[2] = "A";

	printf("请选择需要管理的高校商品库存信息（A/B）:");
	scanf("%s", school);
	if (strcmp(school, "A") == 0){
		do{
			printf("\n\n                    A校在线订购平台（管理员）                   \n");
			printf("           *  1 商品采编入库               2 商品删除            *\n");
			printf("           *  3 检索商品信息               4 编辑商品信息         *\n");
			printf("           *                  0 退出系统                        *\n");
			printf("   *************************************************************\n\n");
			printf("                     请输入您的需求功能编号:");
			scanf("%d", &choice);
			switch(choice)	
			{
				case 1: ItemsAdd(ItemsNumber_A, items_A); break;
				case 2: ItemsDelete(ItemsNumber_A, items_A); break;
				case 3: ItemsList(ItemsNumber_A, items_A); break;
				case 4: ItemsEdit(ItemsNumber_A, items_A); break;
				case 0: Serialize(items_A, filepath_A);
				printf("\n\n\n\n\n\t所有数据已更新保存");
				printf("\n\n\n\n\n\t^^^^^^^^^^感谢您的使用，再见^^^^^^^^^^\n\n");
				exit(0);
				default :
					printf("\n\n\t\t输入有误\n"); break;
			}
		}while(choice != 0);
	}
	else if (strcmp(school, "B") == 0){
		do{
			printf("\n\n                    B校在线订购平台（管理员）                   \n");
			printf("           *  1 商品采编入库               2 商品删除            *\n");
			printf("           *  3 检索商品信息               4 编辑商品信息         *\n");
			printf("           *                  0 退出系统                        *\n");
			printf("   *************************************************************\n\n");
			printf("                     请输入您的需求功能编号:");
			scanf("%d", &choice);
			switch(choice)	
			{
				case 1: ItemsAdd(ItemsNumber_B, items_B); break;
				case 2: ItemsDelete(ItemsNumber_B, items_B); break;
				case 3: ItemsList(ItemsNumber_B, items_B); break;
				case 4: ItemsEdit(ItemsNumber_B, items_B); break;
				case 0: Serialize(items_B, filepath_B);
				printf("\n\n\n\n\n\t所有数据已更新保存");
				printf("\n\n\n\n\n\t^^^^^^^^^^感谢您的使用，再见^^^^^^^^^^\n\n");
				exit(0);
				default :
					printf("\n\n\t\t输入有误\n"); break;
			}
		}while(choice != 0);
	}
	else{
		printf("\n输入有误！");
	}
}


void Body2(){
	int choice = 1;
	do
	{
		printf("\n\n                      在线订购平台(A校学生)                     \n");
		printf("           *  1 检索商品信息                  2 商品订购          *\n");
		printf("           *                0 退出系统                          *\n");
		printf("   *************************************************************\n\n");
		printf("                      请输入您的需求功能编号:");
		
		scanf("%d", &choice);

		switch(choice)
		{
			case 1: ItemsList(ItemsNumber_A, items_A); break;
			case 2: Purchase(ItemsNumber_A, items_A); break;
			case 0: Serialize(items_A, filepath_A);
			printf("\n\n\n\n\n\t所有数据已更新保存");
			printf("\n\n\n\n\n\t^^^^^^^^^^感谢您的使用，再见^^^^^^^^^^\n\n");
			exit(0);
			default :
				printf("\n\n\t\t输入有误\n"); break;
		}
	}while(choice != 0);
}

void Body3(){
	int choice = 1;
	do
	{
		printf("\n\n                      在线订购平台(B校学生)                     \n");
		printf("           *  1 检索商品信息                  2 商品订购          *\n");
		printf("           *                0 退出系统                          *\n");
		printf("   *************************************************************\n\n");
		printf("                      请输入您的需求功能编号:");
		
		scanf("%d", &choice);

		switch(choice)
		{
			case 1: ItemsList(ItemsNumber_B, items_B); break;
			case 2: Purchase(ItemsNumber_B, items_B); break;
			case 0: Serialize(items_B, filepath_B);
			printf("\n\n\n\n\n\t所有数据已更新保存");
			printf("\n\n\n\n\n\t^^^^^^^^^^感谢您的使用，再见^^^^^^^^^^\n\n");
			exit(0);
			default :
				printf("\n\n\t\t输入有误\n"); break;
		}
	}while(choice != 0);
}


void welcome(){
	int option;
	
	printf("************************************************************\n");
	printf("                  **欢迎进入在线选购平台**\n");
	printf("                  **请对您的身份进行选择**\n");
	printf("************************************************************\n");
	printf("                        1:管理员\n");
	printf("                        2:A校学生\n");
	printf("                        3:B校学生\n");
	printf("                        4:退出\n");
	printf("************************************************************\n");
	printf("                        我的选择是:");
	scanf("%d", &option);
	printf("************************************************************\n\n");
	
	switch(option)
	{
		case 1: Body1();break;
		case 2: Body2();break;
		case 3: Body3();break;
		case 4: exit(0);
		default :
			printf("\n\n\t\t您的输入有误\n"); break;

	}
}

int main()
{
	ItemsNumber_A = 0;
	ItemsNumber_B = 0;
	Deserialize(items_A, filepath_A); 
	Deserialize(items_B, filepath_B); 
	welcome();

	return 0;
}