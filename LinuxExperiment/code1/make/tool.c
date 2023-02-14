#include "tool.h"

int find_max(int arr[], int n)
{
	int m = arr[0];
	for(int i = 0; i < n; i++)
	{
		if(arr[i] > m)
		{
			m = arr[i];
		}
	}
	return m;
}
