#include "FileOption.h"

// 构造函数，传入路径与缓存大小，根据传入的数据初始化
FileOption::FileOption(const char*pathname, int size, char type){
	file_descriptor = open(pathname, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR); // 打开文件
	if(file_descriptor == -1){
		throw -1;
	}
	if(size <= 0){
		throw -2;
	}
	file_option_type = type; // 文件操作类型
	read_buffer = new char[size]; // 读缓存数组
	write_buffer = new char[size]; // 写缓存数组
	read_buffer_len = write_buffer_len = 0; // 读写缓存的长度
	read_buffer_size = write_buffer_size = size; // 读写缓存区的大小
	read_next_char = write_next_char = 0; // 游标位置
	read_cur = 0; // 跟踪当前文件的读取位置
}

// 析构函数，关闭文件以及调用flush
FileOption::~FileOption(){
	if(file_descriptor != -1){
		flush();
		close(file_descriptor);
	}
}

// 刷缓存
void FileOption::flush(){
	if (file_option_type == 'r') // 如果文件操作类型是读操作，那么直接返回，不需要刷新写缓存
		return;
	ssize_t write_len = write(file_descriptor, write_buffer, write_buffer_len); // 将写缓存中的数据写入到文件
	if(-1 == write_len){ // 写入失败
		throw -1;
	}
	write_buffer_len = write_buffer_size - write_len; // 更新写缓存的长度，写缓存区的大小减去实际写入的长度
	write_next_char = 0; // 将下一个要写入的字符的位置设置为0
}

// 设置文件的读写位置
bool FileOption::set_offset(int offset){
	off_t position = lseek(file_descriptor, offset, SEEK_SET); // 使用 lseek 函数将文件的读写位置设置为 offset，返回新的读写位置
	if(position == -1){
		return false;
	}
	return true;
}

// 从文件读取数据到缓存
void FileOption::fill(){
	if (file_option_type == 'w') // 如果文件操作类型是写操作，那么直接返回，不需要填充读取缓冲区
		return;
	
	try{ // 刷缓存，避免因为写缓存的数据没有及时更新而出现的错误
		flush(); 
	}
	catch(int e){
		if(e == -1){
			std::cout << "刷缓存失败" << std::endl;
		}
	}
	int end = lseek(file_descriptor, 0, SEEK_END);  // 使用 lseek 函数获取文件的末尾位置
	set_offset(read_cur); // 使用 set_offset 函数将文件的读写位置设置为当前读取位置
	int count = read_buffer_size <= (end - read_cur) ? read_buffer_size : (end - read_cur); // 如果当前读缓存区长度小于等于当前位置到文件末尾的字符数，那么读取读缓存区大小的数据，否则读取剩余的所有字符
	ssize_t read_len = read(file_descriptor, read_buffer, count); // 从文件中读取指定长度的数据到读缓存
	if(-1 == read_len){ // 读取失败
		throw -1;
	}else if(0 == read_len){ // 到达文件末尾
		throw 0;
	}else { // 读取成功
		read_buffer_len = read_len; // 更新读缓存的长度
		read_next_char = 0; // 将下一个要读取的字符的位置设置为0
		read_cur += read_len; // 更新当前读取位置
	}
}

// 从文件中读取数据
int FileOption::Read(char* rbuf, int off, int len){
	int i, end;
	for(i = off, end = off + len; i < end; i++){ // 从偏移量处开始读取指定长度的字符数
		if(read_buffer_len == 0){ // 如果读缓存的长度为0，那么调用 fill 函数
			try{
				fill();
			}
			catch(int e){
				if(e == -1){ // 读取失败
					return -1;
				}
				if(e == 0){ // 已到达文件末尾
					return i - off; // 返回已经读取的字符数
				}
			}
		}
		rbuf[i] = read_buffer[read_next_char]; // 从读缓存中读取一个字符，存储到字符数组中
		read_next_char++; // 更新下一个要读取的字符的位置
		read_buffer_len--; // 减少读缓存长度
	}
	return i - off; // 返回已经读取的字符数
}

// 将数据写入到文件
int FileOption::Write(const char* wbuf, int off, int len){
	int i, end;
	for(i = off, end = off + len; i < end; i++){ // 从偏移量处开始写入指定长度的字符
		//缓存数组写满则调用flush（）
		if(write_buffer_len >= write_buffer_size){ // 如果写缓存的长度不小于写缓存区的大小，刷新写缓存区
			try{
				flush();
			}
			catch(int e){
				if(e == -1){ // 刷新失败
					return -1;
				}
			}
		}
		write_buffer[write_next_char] = wbuf[i]; // 将字符数组中的一个字符写入到写缓存
		write_next_char++; // 更新下一个要写入的字符的位置
		write_buffer_len++; // 增加写缓存的长度
	}
	return i - off; // 返回已经写入的字符数
}