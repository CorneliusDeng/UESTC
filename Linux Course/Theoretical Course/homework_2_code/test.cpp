#include <iostream>
#include <string>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <time.h>
#include <pwd.h>
#include <grp.h>
#include <dirent.h>

// 实现“ls -l”的基本功能
void Print_File_Info(const char* path){

    struct stat st; // stat 结构体变量，用于存储文件或目录的信息
    int ret = stat(path, &st); // 调用stat函数获取文件/目录的信息
    if(ret == -1){ // 如果stat函数调用失败
        perror("stat");
        exit(1);
    }

    std::string perms(11, 0); // 定义一个字符串，用于存储文件类型和权限

    switch(st.st_mode & S_IFMT) // 根据文件类型标志位判断文件类型
    {
        case S_IFLNK:
            perms[0] = 'l'; // 符号链接
            break;
        case S_IFDIR:
            perms[0] = 'd'; // 目录
            break;
        case S_IFREG:
            perms[0] = '-'; // 普通文件
            break;
        case S_IFBLK:
            perms[0] = 'b'; // 块设备
            break;
        case S_IFCHR:
            perms[0] = 'c'; // 字符设备
            break;
        case S_IFSOCK:
            perms[0] = 's'; // 套接字
            break;
        case S_IFIFO:
            perms[0] = 'p'; // 管道
            break;
        default:
            perms[0] = '?'; // 未知类型
            break;
    }
    // 判断文件的访问权限
    // 文件所有者
    perms[1] = (st.st_mode & S_IRUSR) ? 'r' : '-';
    perms[2] = (st.st_mode & S_IWUSR) ? 'w' : '-';
    perms[3] = (st.st_mode & S_IXUSR) ? 'x' : '-';
    // 文件所属组
    perms[4] = (st.st_mode & S_IRGRP) ? 'r' : '-';
    perms[5] = (st.st_mode & S_IWGRP) ? 'w' : '-';
    perms[6] = (st.st_mode & S_IXGRP) ? 'x' : '-';
    // 其他人
    perms[7] = (st.st_mode & S_IROTH) ? 'r' : '-';
    perms[8] = (st.st_mode & S_IWOTH) ? 'w' : '-';
    perms[9] = (st.st_mode & S_IXOTH) ? 'x' : '-';

    
    int linkNum = st.st_nlink; // 硬链接计数
    std::string fileUser = getpwuid(st.st_uid)->pw_name; // 文件所有者
    std::string fileGrp = getgrgid(st.st_gid)->gr_name; // 文件所属组
    int fileSize = (int)st.st_size; // 文件大小
    std::string time = ctime(&st.st_mtime);
    std::string mtime = time.substr(0, time.size()-1); // 修改时间

    std::cout << perms << "  " << linkNum << "  " << fileUser << "  " << fileGrp << "  " << fileSize << "  " << mtime << "  " << path << "\n";
}

int main(int argc, char* argv[]){   

    // 如果没有提供足够的参数（至少需要一个文件或目录名），则打印提示
    if(argc < 2){
        std::cout << "Execution format : ./out file_name or directory_name\n";
        exit(1);
    }

    struct stat st; // stat 结构体变量，用于存储文件或目录的信息
    int ret = stat(argv[1], &st); // 调用 stat 函数获取 argv[1]（文件或目录名）的信息，并将信息存储在 st 中
    if(ret == -1) { // 如果stat函数调用失败
        perror("stat");
        exit(1);
    }

    if(S_ISDIR(st.st_mode)) { // 如果 st.st_mode 表示的是一个目录
        DIR* dir = opendir(argv[1]); // 打开这个目录
        if(dir == NULL) { // 如果打开失败，打印错误信息并退出程序
            perror("fail to open directory"); 
            exit(1);
        }

        struct dirent* entry; // dirent 结构体变量，用于存储目录中的文件信息
        while((entry = readdir(dir)) != NULL) { // 循环读取目录中的每一个文件
            if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) { // 如果文件名是 "." 或 ".."，则跳过这个文件
                continue;
            }
            std::string filePath = std::string(argv[1]) + entry->d_name; // 构造文件的完整路径
            Print_File_Info(filePath.c_str()); // 打印文件信息
        }

        closedir(dir); // 关闭目录
    } 
    else {// 如果 st.st_mode 表示的是一个文件，直接打印文件的信息
        Print_File_Info(argv[1]); 
    }

    return 0;
}