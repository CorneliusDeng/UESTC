#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>

// 使用std:: queue实现队列的线程安全
template <typename T>
class SafeQueue
{
private:
    std::queue<T> m_queue; // 利用模板函数构造队列
    std::mutex m_mutex; // 访问互斥信号量

public:
    SafeQueue();
    SafeQueue(SafeQueue &&other);
    ~SafeQueue();

    bool empty(); // 判断返回队列是否为空
    int size(); // 返回队列大小
    
    void enqueue(T &t) // 队列添加元素
    {
        std::unique_lock<std::mutex> lock(m_mutex); // 互斥信号量加锁
        m_queue.emplace(t); // 入队
    }

    bool dequeue(T &t) // 队列取出元素
    {
        std::unique_lock<std::mutex> lock(m_mutex); // 互斥信号量加锁

        if (m_queue.empty())
            return false;
        t = std::move(m_queue.front()); // 将队列的第一个元素移动给t，移动操作比复制操作快

        m_queue.pop(); // 移除队首的元素

        return true;
    }
};

class ThreadPool
{
private:
    class ThreadWorker; // 内置线程工作类
    bool m_shutdown; // 线程池是否关闭
    SafeQueue<std::function<void()>> m_queue; // 执行函数安全队列，即任务队列
    std::vector<std::thread> m_threads; // 工作线程队列
    std::mutex m_conditional_mutex; // 线程休眠锁互斥变量
    std::condition_variable m_conditional_lock; // 线程环境锁，可以让线程处于休眠或者唤醒状态

public:
    ThreadPool(const int n_threads = 4); // 线程池构造函数
    ThreadPool(const ThreadPool &) = delete; // 禁止拷贝构造函数
    ThreadPool(ThreadPool &&) = delete; // 禁止移动构造函数
    ThreadPool &operator=(const ThreadPool &) = delete; // 禁止拷贝赋值函数
    ThreadPool &operator=(ThreadPool &&) = delete; // 禁止移动赋值函数

    void init(); // 初始化线程池
    void shutdown(); // 等待线程池中的任务执行完毕并关闭线程池

    // 使用可变模版函数，提交一个函数（以及其参数）到线程池中异步执行
    template <typename F, typename... Args>
    auto submit(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
    {
        // 传入的函数和参数绑定在一起，创建一个可调用的函数对象 func
        std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...); 

        // 将func封装到一个共享指针的可异步执行的packaged_task中
        auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

        // 封装函数对象，以便我们可以将其放入队列中
        std::function<void()> warpper_func = [task_ptr]()
        {
            (*task_ptr)();
        };

        m_queue.enqueue(warpper_func); // 将任务压入队列

        m_conditional_lock.notify_one(); // 唤醒一个等待中的线程

        return task_ptr->get_future(); // 返回异步执行的结果
    }
};

#endif