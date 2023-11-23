#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <string>

#define defaultCharBuffer 1024 // 默认缓存数组大小为1024

class FileOption{
public:
	FileOption(const char* path_name, int size = defaultCharBuffer, char type = 'r');
	~FileOption();
	int Read(char* read_buffer, int off, int len); // 读文件
	int Write(const char* write_buffer, int off, int len); // 写文件
private:
	void fill(); // 从文件读取数据到缓存
	void flush(); // 刷缓存
	bool set_offset(int offset); // 设置文件的读写位置
	char* read_buffer; // 读缓存
	char* write_buffer; // 写缓存
	int read_buffer_len, write_buffer_len; // 读写缓存的长度
	int read_buffer_size, read_next_char; // 读缓存区的大小，下一个要从读缓冲区中读取的字符的位置
	int write_buffer_size, write_next_char; // 写缓存区的大小，下一个要写入到写缓冲区的字符的位置
	char file_option_type; // 文件操作类型
	int file_descriptor; // 文件描述符
	int read_cur; // 跟踪文件的读取位置
};