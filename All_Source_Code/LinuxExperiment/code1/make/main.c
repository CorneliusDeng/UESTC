#include "tool.h"
#include "bar.h"
#include <stdio.h>

int main()
{
	int arr[5] = {1,9,3,8,0};
	int min = find_min(arr, 5);
	int max = find_max(arr, 5);
	printf("min = %d\n", min);
	printf("max = %d\n", max);	
}
