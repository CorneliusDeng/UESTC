#include "Statement.h"

// 商品信息采编入库
void ItemsAdd(int ItemsNumber, ItemsInformation items[]){
	int i = 0, j = 0, flag = 0;
	char name[20]; // 商品名
    float price; // 商品价格
    int count; // 商品数量
	char choice;

	printf("--------------商品采编入库--------------\n");

	printf("商品名称: ");
	scanf("%s", name);

	printf("商品价格: ");
	scanf("%f", &price);

	printf("商品数量: ");
	scanf("%d", &count);
	printf("\n输入完毕\n");
	
	for (i = ItemsNumber; i < 100; i++){
		for(j = 0; j < ItemsNumber; j++){
			if(strcmp(items[j].name, name)==0) // 如过采购的商品已在仓库内了
			{
				flag = 1;
				items[j].count = items[j].count + count;
				printf("\n商品 %s 库存已增加 %d\n",items[j].name,count);
			}
		}
		if(flag == 0){
			strcpy(items[i].name,name);
            items[i].price = price;
			items[i].count = count;
			flag++;
			ItemsNumber++;
		}
		printf("是否要退出(y/n):");
		fflush(stdin); // 清空输入缓冲区
		scanf("%c", &choice);
		fflush(stdin);
		if(choice=='y'||choice=='Y'){
			break;
		}
	}
}

// 删除商品信息
int ItemsDelete(int ItemsNumber, ItemsInformation items[])
{
	int i, j, flag = -1;
	char name[20];

	printf("--------------清除商品库存--------------\n");
	printf("请输入想要删除的商品名称:");
	scanf("%s", name);
	for (i = 0; i < ItemsNumber; i++){
		if(strcmp(items[i].name,name) == 0){
			for(j = i; j < ItemsNumber; j++){
                strcpy(items[j].name,items[j+1].name);
                items[j].price = items[j+1].price;
			    items[j].count = items[j+1].count;

				ItemsNumber--;
			}
			printf("该商品已经删除！"); 
			flag = i;
		}
	}
	
	if(flag == -1){
		printf("没有找到相关记录");
		fflush(stdin);
	}
	return flag; 
}

// 通过商品名查询商品信息
int ItemsSearchByName(int ItemsNumber, ItemsInformation items[]){
    int i, flag = 0;
	char name[20];
	
    printf("请输入商品名称：\n");
	scanf("%s", name);

	putchar('\n');
	
	for (i = 0;i < ItemsNumber; i++){
		if (strcmp(name, items[i].name) == 0){
		   printf("名称：%s \t", items[i].name); 
		   printf("价格：%.2f \t", items[i].price);
		   printf("数量：%d\n", items[i].count);
		   flag = 1;
		   break;
		}
	}

	fflush(stdin);

	if (flag == 0){
		printf("\n未找到相关记录\n");
	}
	return 0;
}

// 商品信息列表
void ItemsList(int ItemsNumber, ItemsInformation items[]){
	printf("商品名\t价格\t库存数量\n");
	for(int i = 0; i < ItemsNumber; i++){
		printf("%s\t%.2f\t%d\n", items[i].name, items[i].price, items[i].count);
	}

}


// 商品信息编辑
int ItemsEdit(int ItemsNumber, ItemsInformation items[])
{
	int i = 0, j = 0, flag;
	char name[20]; // 商品名
	char nameEdit[20]; // 修改后的商品名
    float price; // 商品价格
    int count; // 商品数量
	char choice;

	printf("--------------商品信息编辑--------------\n");
	printf("请输入商品名称：\n");
	scanf("%s", name);
	
	putchar('\n');
	
	for (i = 0;i < ItemsNumber; i++){
		if (strcmp(name, items[i].name) == 0){
		   printf("名称：%s \t", items[i].name); 
		   printf("价格：%.2f \t", items[i].price);
		   printf("数量：%-5d\n", items[i].count);
		   flag = 1;
		   break;
		}
	}
	fflush(stdin);
	if (flag == 0){
		printf("\n未找到相关记录\n");
		return flag;
	}

	printf("请输入修改后的商品信息：\n");
	printf("商品名称: ");
	scanf("%s", nameEdit);

	printf("商品价格: ");
	scanf("%f", &price);

	printf("商品数量: ");
	scanf("%d", &count);
	printf("\n输入完毕\n");

	for (i = 0;i < ItemsNumber; i++){
		if (strcmp(name, items[i].name) == 0){
			strcpy(items[i].name,nameEdit);
			items[i].price = price;
			items[i].count = count;
			flag = 1;
			break;
		}
	}
	return flag; 
}

// 学生选购商品
int Purchase(int ItemsNumber, ItemsInformation items[]){
	char name[20]; // 已查询到的商品名称
	int i, j, flag = 0, order, ordernum; // order为当前商品的数组位置，ordernum为购买数量

    printf("--------------学生订购系统--------------\n");

	printf("请输入要购买的商品名称:");
	scanf("%s", name);

	for (i = 0; i < ItemsNumber; i++)  //查询商品
	{
		if(strcmp(items[i].name, name) == 0)
		{
			flag = 1;
			order = i;
			break;
		}
	}
	if (flag == 0)  //商品不存在
	{
		printf("商品不存在,请重新输入\n");
		fflush(stdin);
		return flag;
	}
	else  //商品存在
	{
		printf("%s单价是%.2f,现有%d件\n", items[order].name, items[order].price, items[order].count);

        if(items[order].count == 0)  //该商品已售空
        {
            printf("暂时没有库存,订购失败\n");
            fflush(stdin);
            return 0;
        }
        else  //有库存
        {
            printf("请输入商品购买数量：");
            scanf("%d", &ordernum);
            if (ordernum > items[order].count){
                printf("购买数量大于库存数量,订购失败\n");
                fflush(stdin);
                return 0;
            }
            else{
                printf("订购的商品信息如下：\n");
                float money = items[order].price * ordernum;
                printf("购买商品 %s %d 件, 共需付款 %.2f 元", items[order].name, items[order].count, money);
                items[order].count -= ordernum;
            }
            printf("\n购买成功，祝您购物愉快\n");
            fflush(stdin);
            return 0;
        }
    }
}