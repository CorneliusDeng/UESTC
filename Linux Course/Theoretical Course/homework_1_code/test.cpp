#include "FileOption.h"

// 测试带缓存的文件操作类
int main(){
	FileOption file1("test.txt", 1024, 'r'); // 读文件
	FileOption file2("copy.txt", 1024, 'w'); // 写文件

	int len ; // len 存储每次读取的字符数
	char* ch = new char[1024]; // 存储读取的数据

	while(0 < (len = file1.Read(ch, 0, 1024))){ // 循环读取数据，直到读取失败或者到达文件末尾
		file2.Write(ch, 0, len); // 将读取到的数据写入到 file2 对象中
	}
}