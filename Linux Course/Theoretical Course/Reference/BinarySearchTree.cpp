#include"BinarySearchTree.h"
#include <unistd.h>
#include <vector>


BSTtree::BSTtree(){
    root = nullptr;
}

BSTtree::~BSTtree(){
    //递归删除
    Deletetree(root);
}

bool BSTtree::InsertNode(BSTNode * &root,const std::pair<LwjType,LwjType> &data){
    if(root==nullptr){
        root = new BSTNode();
        root->data.key = data.first;
        root->data.val.push_back(data.second);
        //添加到文件里面去
        return true;
    }
    bool f = 1;
    if(data.first == root->data.key){
        root->data.val.push_back(data.second);
    }else if(data.first < root->data.key){
        f = InsertNode(root->lchild,data);
    }else{
        f = InsertNode(root->rchild,data);
    }
    return f;
}


bool BSTtree::RealInsertNode(const std::pair<LwjType,LwjType> &data,int col){
    bool f = InsertNode(root,data);
    //std::cout<<root->data.key<<"\n";
    RealWriteFile(col);
    return f;
}

void BSTtree::print(BSTNode *root){
    if(root==nullptr)return;
    std::cout<<root->data.key << " " ;
    for(int i = 0;i < root->data.val.size();i++){
        std::cout << root->data.val[i] << " " ;
    }
    std::cout << "\n";
    print(root->lchild);
    print(root->rchild);
}

void BSTtree::Realprint(){
    print(root);
}

std::vector<LwjType> BSTtree::search(BSTNode *root,int left,int right){
    if(left>right)return std::vector<LwjType>();
    if(root->lchild==nullptr&&root->rchild==nullptr){
        if(left <= root->data.key && right >= root->data.key)return root->data.val;
        else return std::vector<LwjType>();
    }
    std::vector<LwjType>v1,v2,ans;
    if(left <= root->data.key && right >= root->data.key){
        ans = root->data.val;
        if(root->lchild)v1 = search(root->lchild,left,root->data.key-1);
        if(root->rchild)v2 = search(root->rchild,root->data.key+1,right);
    }else if(left >= root->data.key){
        if(left==root->data.key){
            v1 = root->data.val;
            left++;
        }
        if(root->rchild)ans = search(root->rchild,left,right);
    }else if(right <= root->data.key){
        if(right==root->data.key){
            v1 = root->data.val;
            right--;
        }
        if(root->lchild)ans = search(root->lchild,left,right);
    }
    for(int i = 0;i < v1.size();i++)ans.push_back(v1[i]);
    for(int i = 0;i < v2.size();i++)ans.push_back(v2[i]);
    return ans;
}

std::vector<LwjType> BSTtree::Realsearch(int left,int right){
    return search(root,left,right);
}

bool BSTtree::RealWriteFile(int col){
    char IndexName[200];
    sprintf(IndexName,"%s%d",INDEX_FILE_PATH,col);
    File = open(IndexName, O_RDWR|O_CREAT,S_IRUSR | S_IWUSR);

    //std::cout<<"开始写索引文件\n";
    if(-1==File){
        std::cout << "文件打开失败!" << "\n";
        return false;
    }
    int f = WriteFile(root);
    close(File);
    return f;
}

bool BSTtree::RealReadFile(int col){
    char IndexName[200];
    sprintf(IndexName,"%s%d",INDEX_FILE_PATH,col);
    File = open(IndexName, O_RDWR|O_CREAT,S_IRUSR | S_IWUSR);

    //std::cout<<"开始读取索引文件\n";
    if(-1==File){
        std::cout << "文件打开失败!" << "\n";
        return false;
    }
    int f = ReadFile(root);
    close(File);
    return f;
}


bool BSTtree::WriteFile(BSTNode *root){
    int type;
    //如果是空节点
    if(root==nullptr){
        type = 0;
        int f = write(File,&type,sizeof(int));
        if(-1==f){
            std::cout << "索引文件读入失败!" << "\n";
            return false;
        }
        return true;
    }


    //如果是真节点
    type = 1;
    int status = write(File,&type,sizeof(int));
    if(-1==status){
        std::cout << "索引文件读入失败!" << "\n";
        return false;
    }

    status = write(File,&(root->data.key),sizeof(root->data.key));
    if(-1==status){
        std::cout << "索引文件读入失败!" << "\n";
        return false;
    }

    int ValByte = root->data.val.size()*sizeof(LwjType);
    status = write(File,&ValByte,sizeof(ValByte));
    if(-1==status){
        std::cout << "索引文件读入失败!" << "\n";
        return false;
    }

    for(int i = 0;i < root->data.val.size();i++){
        status = write(File,&(root->data.val[i]),sizeof(root->data.val[i]));
        if(-1==status){
            std::cout << "索引文件读入失败!" << "\n";
            return false;
        }
    }

    bool f = 1;
    f = (f & WriteFile(root->lchild));
    f = (f & WriteFile(root->rchild));
    return f;
}


bool BSTtree::ReadFile(BSTNode * &root){
    int type;
    int status = read(File,&type,sizeof(int));
    if(-1==status){
        std::cout << "索引文件写入失败!" << "\n";
        return false;
    }
    if(type==0){
        root = nullptr;
        return true;
    }
    
    //创建一个节点
    root = new BSTNode();
    status = read(File,&(root->data.key),sizeof(root->data.key));
    
    if(-1==status){
        std::cout << "索引文件写入失败!" << "\n";
        return false;
    }

    int ValByte;
    status = read(File,&ValByte,sizeof(ValByte));

    if(-1==status){
        std::cout << "索引文件写入失败!" << "\n";
        return false;
    }

    int num = ValByte/sizeof(LwjType);
    
    root->data.val.resize(num);
    for(int i = 0;i < num;i++){
        status = read(File,&(root->data.val[i]),sizeof(root->data.val[i]));
        if(-1==status){
            std::cout << "索引文件写入失败!" << "\n";
            return false;
        }
    }

    bool f = 1;
    f = (f & ReadFile(root->lchild));
    f = (f & ReadFile(root->rchild));

    return f;
}

void BSTtree::Deletetree(BSTNode *root){
    if(root==nullptr)return;
    Deletetree(root->lchild);
    Deletetree(root->rchild);
    delete root;
}