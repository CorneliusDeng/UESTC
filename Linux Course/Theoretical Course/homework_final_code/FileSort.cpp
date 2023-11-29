#include "FileSort.h"

// 获取单个文件中的数据
std::deque<int64_t> FileSort::Get_Single_File(const char* file_path){
    std::ifstream file(file_path, std::ios::in); // 以文本模式打开文件

    if(!file.is_open()){
        std::cerr << "failed to open file in Get_Single_File: " << file_path << std::endl;
        exit(1);
    }

    std::deque<int64_t> file_data; // 创建一个双端队列来存储文件中的数据
    int64_t number;
    while(file >> number){ // 循环读取文件中的每一个64位有符号数
        if(file_data.size() * sizeof(int64_t) < MAX_BUFFER_SIZE){
            file_data.push_back(number); // 将读取到的数添加到双端队列中
        }else{
            file_data.pop_front();
            file_data.push_back(number);
        }
    }

    file.close();
    return file_data;
}

// 写文件
void FileSort::writeToFile(std::deque<int64_t>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::out); // 创建一个 ofstream 对象，以文本模式打开文件

    if(!file.is_open()){ // 如果文件打开失败，打印错误信息并退出程序
        std::cerr << "failed to open the writed file: " << filename << std::endl; 
        exit(1);
    }

    while(!data.empty()){ // 当数据不为空时，循环写入数据
        int64_t number = data.front(); // 获取数据的第一个元素
        file << number << "\n"; // 将数字写入文件
        data.pop_front(); // 删除已经写入的数据
    }

    file.close();
}

// 获取指定目录下所有文件的数据，并组成键值对
std::map<std::string, std::deque<int64_t>> FileSort::Get_All_File(const char* dir_path){

    std::map<std::string, std::deque<int64_t>> file_data; // 以键值对的形式存储文件路径和其对应的数据

    struct stat st; // stat 结构体变量，用于目录的信息
    int ret = stat(dir_path, &st); // 调用stat函数获取目录的信息
    if(ret == -1){ // 如果stat函数调用失败
        perror("stat");
        exit(1);
    }

    if(S_ISDIR(st.st_mode)) { // 如果 st.st_mode 表示的是一个目录
        DIR* dir = opendir(dir_path); // 打开这个目录
        if(dir == NULL) { // 如果打开失败，打印错误信息并退出程序
            perror("fail to open directory"); 
            exit(1);
        }

        struct dirent* entry; // dirent 结构体变量，用于存储目录中的文件信息
        while((entry = readdir(dir)) != NULL) { // 循环读取目录中的每一个文件
            if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) { // 如果文件名是 "." 或 ".."，则跳过这个文件
                continue;
            }
            std::string file_path = std::string(dir_path) + entry->d_name; // 构造文件的完整路径
            std::deque<int64_t> data = Get_Single_File(file_path.c_str());
            file_data[file_path] = data; // 组成键值对，将文件路径作为键，文件数据作为值
        }

        closedir(dir); // 关闭目录
    } 
    else {
        std::cout << "do not get a valid directory" << std::endl;
    }

    return file_data; // 返回map
}

// 对一个队列进行归并排序，返回一个新的有序队列
std::deque<int64_t> FileSort::mergeSort(std::deque<int64_t> &data) {
    if(data.size() <= 1) { // 如果输入队列的大小小于或等于1，直接返回输入队列
        return data;
    }

    auto middle = data.begin() + data.size() / 2; // 创建一个迭代器，指向输入队列的中间位置
    // 创建两个新的队列，分别存储输入队列的左半部分和右半部分
    std::deque<int64_t> left(data.begin(), middle);
    std::deque<int64_t> right(middle, data.end());

    // 对左半部分和右半部分分别进行归并排序
    left = mergeSort(left);
    right = mergeSort(right);

    // 将排序后的左半部分和右半部分合并成一个有序的队列，并返回
    return merge(left, right);
}

// 合并两个有序队列
std::deque<int64_t> FileSort::merge(std::deque<int64_t> left, std::deque<int64_t> right) {
    std::deque<int64_t> result; // 存储结果
    auto left_it = left.begin(), right_it = right.begin(); // 创建两个迭代器，分别指向两个输入队列的开始

    // 当两个输入队列都不为空时，执行以下操作
    while(left_it != left.end() && right_it != right.end()) {
        if(*left_it <= *right_it) { // 如果左边队列的第一个元素小于或等于右边队列的第一个元素
            result.push_back(*left_it); // 将左边队列的第一个元素添加到结果队列中，并将左边队列的迭代器向前移动一位
            ++left_it;
        } else {
            result.push_back(*right_it); // 否则，将右边队列的第一个元素添加到结果队列中，并将右边队列的迭代器向前移动一位
            ++right_it;
        }
    }

    // 如果左边队列还有剩余的元素，将它们全部添加到结果队列中
    while(left_it != left.end()) {
        result.push_back(*left_it);
        ++left_it;
    }

    // 如果右边队列还有剩余的元素，将它们全部添加到结果队列中
    while(right_it != right.end()) {
        result.push_back(*right_it);
        ++right_it;
    }

    return result;
}