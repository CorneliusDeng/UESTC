#include "Statement.h"

// 反序列化，读取商品信息
void Deserialize(ItemsInformation items[], char *filepath)
{
	int i = 0;
	FILE *fp;

	if ((fp = fopen(filepath, "r+")) == NULL)
	{
		printf("open failed!");
		exit(0);
	}
	for (i = 0; !feof(fp); i++)
		{
			if(fscanf(fp,"%s\t%f\t%d\n", items[i].name, &items[i].price, &items[i].count) == 3)
			{
				ItemsNumber_A++;
			}
		}
	fclose(fp);
}

// 序列化，保存商品信息到文件中
void Serialize(ItemsInformation items[], char *filepath)
{
	int i = 0;
	FILE *fp;

	if ((fp = fopen(filepath, "w+")) == NULL)
	{
		printf("open failed!");
		exit(0);
	}

	for (i = 0; i < ItemsNumber_A; i++)
	{
		fprintf(fp,"%s\t%f\t%d\n", items[i].name, items[i].price, items[i].count);
	}
	fclose(fp);
}