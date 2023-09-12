#ifndef BinarySearchTree_H
#define BinarySearchTree_H
#include"Table.h"

#define INDEX_FILE_PATH "index_" //索引文件前缀

struct BSTNode{
    Element data;
    BSTNode *lchild;
    BSTNode *rchild;
    BSTNode():lchild(nullptr),rchild(nullptr){};
};

class BSTtree{
private:
    
    int File;

public:BSTNode *root;
    BSTtree();
    ~BSTtree();

    bool InsertNode(BSTNode *&root,const std::pair<LwjType,LwjType> &data);
    bool RealInsertNode(const std::pair<LwjType,LwjType> &data,int col);
    std::vector<LwjType> search(BSTNode *root,int left,int right);
    std::vector<LwjType> Realsearch(int left,int right);

    bool RealWriteFile(int col);
    bool WriteFile(BSTNode *root);
    bool RealReadFile(int col);
    bool ReadFile(BSTNode *&root);

    void print(BSTNode *root);
    void Realprint();

    void Deletetree(BSTNode *root);


};


#endif 