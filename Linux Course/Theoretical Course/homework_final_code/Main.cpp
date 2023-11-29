#include "FileSort.h"
#include "ThreadPool.h"
#include <random>

// 用于在指定目录下生成指定数量的文件，每个文件包含随机数量的随机64位有符号整数
void generate_files(const std::string& directory, int num_files, int max_count) {
    std::random_device rd; // 创建一个随机设备
    std::mt19937_64 gen(rd()); // 使用随机设备初始化一个64位的Mersenne Twister引擎
    std::uniform_int_distribution<int64_t> dist(INT64_MIN, INT64_MAX); // 创建一个均匀分布的随机数生成器，范围从INT64_MIN到INT64_MAX，用于生成随机的64位有符号整数
    std::uniform_int_distribution<int> dist_count(1, max_count); // 创建一个均匀分布的随机数生成器，用于生成每个文件中的随机数字数量,范围从1到max_count

    // 循环创建指定数量的文件
    for (int i = 1; i <= num_files; ++i) {
        std::string file_name = "file" + std::to_string(i) + ".txt";
        // 生成文件的完整路径，将目录路径和文件名组合在一起
        std::string file_path = directory;
        if (directory.back() != '/') {
            file_path += "/";
        }
        file_path += file_name;

        // 创建一个输出文件流，打开文件
        std::ofstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file in function generate_files: " + file_path);
        }

        int count = dist_count(gen); // 生成一个随机的数字数量，这将是这个文件中的数字数量
        for (int j = 0; j < count; ++j) { // 在每个文件中写入随机数量的随机数
            int64_t random_number = dist(gen);
            file << random_number << "\n";
        }
        file.close();
    }
}


int main(int argc, char* argv[]){

    if(argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <directory_to_read> <file_to_write> <num_files> <max_count>\n";
        return 1;
    }

    std::string directory_to_read = argv[1];
    std::string file_to_write = argv[2];
    int num_files = std::stoi(argv[3]);
    int max_count = std::stoi(argv[4]);

    generate_files(directory_to_read, num_files, max_count); // 在directory_to_read路径下生成指定个数的文件，每个文件包含随机数量的随机64位有符号整数

    std::cout << "已生成指定数量的随机数据文件 \n";

    ThreadPool pool(4); // 创建4个核心的线程池
    pool.init(); // 初始化线程池

    FileSort file_sort;

    // 提交 Get_All_File 任务    
    auto future1 = pool.submit([&]() { return file_sort.Get_All_File(directory_to_read.c_str()); }); 
    // 等待任务完成并获取结果
    std::map<std::string, std::deque<int64_t>> file_data = future1.get();

    std::cout << "\n获取所有数据文件成功 \n";

    // 创建一个队列，用于存储所有文件的排序后的数据
    std::deque<int64_t> all_sorted_data;

    // 对每个文件的数据进行排序，并将排序后的数据添加到 all_sorted_data 中
    for(auto &pair : file_data) {
        auto future2 = pool.submit([&]() { return file_sort.mergeSort(pair.second); });
        std::deque<int64_t> sorted_data = future2.get();
        all_sorted_data.insert(all_sorted_data.end(), sorted_data.begin(), sorted_data.end());
    }
    std::cout << "\n单个数据文件内部排序成功 \n";

    // for (const auto& num : all_sorted_data) {
    //     std::cout << num << "\n";
    // }
    // std::cout << std::endl;


    // 对 all_sorted_data 进行排序
    auto future3 = pool.submit([&]() { return file_sort.mergeSort(all_sorted_data); });
    all_sorted_data = future3.get();
    std::cout << "\n所有数据排序成功 \n";
    
    // for (const auto& num : all_sorted_data) {
    //     std::cout << num << "\n";
    // }
    // std::cout << std::endl;

    // 写入文件
    file_sort.writeToFile(all_sorted_data, file_to_write.c_str());
    
    pool.shutdown(); // 关闭线程池

    std::cout << "\n线程池已关闭，操作完成 \n";
}