#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <map>
#include <deque>

#define MAX_BUFFER_SIZE 64 * 1024 * 1024 // 默认缓存大小为64MB

class FileSort{
public: 
	FileSort(){}; // 构造函数
	~FileSort(){}; // 析构函数
	
	std::deque<int64_t> Get_Single_File(const char* file_path); // 获取单个文件中的数据
	
	void writeToFile(std::deque<int64_t>& data, const std::string& filename); // 写文件
	
	std::map<std::string, std::deque<int64_t>> Get_All_File(const char* dir_path); // 获取指定目录下所有文件的数据，并组成键值对
	
	std::deque<int64_t> mergeSort(std::deque<int64_t> &data); // 对一个队列进行归并排序，返回一个新的有序队列
	
	std::deque<int64_t> merge(std::deque<int64_t> left, std::deque<int64_t> right); // 合并两个有序队列
};