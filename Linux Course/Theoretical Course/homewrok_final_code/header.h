#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>

// Thread safe implementation of a Queue using a std::queue
template <typename T>
class SafeQueue
{
private:
    std::queue<T> m_queue; //利用模板函数构造队列
    std::mutex m_mutex; // 访问互斥信号量

public:
    SafeQueue();
    SafeQueue(SafeQueue &&other);
    ~SafeQueue();

    bool empty();
    int size();
    // 队列添加元素
    void enqueue(T &t)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.emplace(t);
    }

    // 队列取出元素
    bool dequeue(T &t)
    {
        std::unique_lock<std::mutex> lock(m_mutex); // 队列加锁

        if (m_queue.empty())
            return false;
        t = std::move(m_queue.front()); // 取出队首元素，返回队首元素值，并进行右值引用

        m_queue.pop(); // 弹出入队的第一个元素

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
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    void init(); // Inits thread pool
    void shutdown(); // Waits until threads finish their current task and shutdowns the pool

    // Submit a function to be executed asynchronously by the pool
    template <typename F, typename... Args>
    auto submit(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
    {
        // Create a function with bounded parameter ready to execute
        std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...); // 连接函数和参数定义，特殊函数类型，避免左右值错误

        // Encapsulate it into a shared pointer in order to be able to copy construct
        auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

        // Warp packaged task into void function
        std::function<void()> warpper_func = [task_ptr]()
        {
            (*task_ptr)();
        };

        // 队列通用安全封包函数，并压入安全队列
        m_queue.enqueue(warpper_func);

        // 唤醒一个等待中的线程
        m_conditional_lock.notify_one();

        // 返回先前注册的任务指针
        return task_ptr->get_future();
    }
};

#endif