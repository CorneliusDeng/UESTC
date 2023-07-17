#include <stdio.h>   
int main(int argc, char * argv [ ])
{
    int *p = NULL;

    printf("address [0x%p]\r\n", p);
    *p = 0;

    return 0;
}