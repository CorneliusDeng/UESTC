#include <stdio.h>
#include <stdlib.h>
#include <ncurses.h>
#include <string.h>

typedef struct item
{
	char name[100]; // 商品名称
	float price; // 商品价格 
	int count; // 商品数量   
}ItemsInformation;

int ItemsNumber_A; // A校商品基数
ItemsInformation items_A[100];
char *filepath_A = "Items_A.txt";

int ItemsNumber_B; // B校商品基数
ItemsInformation items_B[100];
char *filepath_B = "Items_B.txt";

void Body1();
void Body2();
void Body3();
void welcome();

void Deserialize(ItemsInformation items[], char filepath[100]); // 反序列化，读取商品信息
void Serialize(ItemsInformation items[], char filepath[100]); // 序列化，保存商品信息到文件中


void ItemsAdd(int ItemsNumber, ItemsInformation items[]); // 商品信息采编入库

int ItemsDelete(int ItemsNumber, ItemsInformation items[]); // 删除商品信息

int ItemsSearchByName(int ItemsNumber, ItemsInformation items[]); // 通过商品名查询商品信息

int ItemsEdit(int ItemsNumber, ItemsInformation items[]); // 商品信息编辑

void ItemsList(int ItemsNumber, ItemsInformation items[]); // 商品信息列表

int Purchase(int ItemsNumber, ItemsInformation items[]); // A校学生选购商品