# 概述

CUDA是建立在NVIDIA的CPUs上的一个通用并行计算平台和编程模型，基于CUDA编程可以利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。近年来，GPU最成功的一个应用就是深度学习领域，基于GPU的并行计算已经成为训练深度学习模型的标配。

GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是CPU的协处理器，因此当我们在说GPU并行计算时，其实是指的基于CPU+GPU的异构计算架构。CPU和主存被称为主机端（Host），GPU和显存（显卡内存）被称为设备端（Device），CPU无法直接读取显存数据，GPU无法直接读取主存数据，主机与设备必须通过总线（Bus）相互通信，如下图所示。

<img src="https://pic3.zhimg.com/80/v2-df49a98a67c5b8ce55f1a9afcf21d982_720w.webp" style="zoom:100%;" />

可以看到GPU包括更多的运算核心，其特别适合数据并行的计算密集型任务，如大型矩阵运算，而CPU的运算核心较少，但是其可以实现复杂的逻辑运算，因此其适合控制密集型任务。另外，CPU上的线程是重量级的，上下文切换开销大，但是GPU由于存在很多核心，其线程是轻量级的。因此，基于CPU+GPU的异构计算平台可以优势互补，CPU负责处理逻辑复杂的串行程序，而GPU重点处理数据密集型的并行计算程序，从而发挥最大功效。



# GPU程序与CPU程序的区别

一个传统的CPU程序的执行顺序如下图所示：

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMTg1Ni5wbmc?x-oss-process=image/format,png" style="zoom:50%;" />

- CPU程序是顺序执行的，一般需要：
  - 初始化
  - CPU计算
  - 得到计算结果

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMTkwOC5wbmc?x-oss-process=image/format,png" style="zoom:50%;" />

- 当引入GPU后，计算流程变为
  - 初始化，并将必要的数据拷贝到GPU设备的显存上
  - CPU调用GPU函数，启动GPU多个核心同时进行计算
  - CPU与GPU异步计算
  - 将GPU计算结果拷贝回主机端，得到计算结果
- 与传统的Python CPU代码不同的是
  - 使用`from numba import cuda`引入`cuda`库
  - 在GPU函数上添加`@cuda.jit`装饰符，表示该函数是一个在GPU设备上运行的函数，GPU函数又被称为**核函数**
  - 主函数调用GPU核函数时，需要添加如`[1, 2]`这样的**执行配置**，这个配置是在告知GPU以多大的并行粒度同时进行计算。`gpu_print[1, 2]()`表示同时开启2个线程并行地执行`gpu_print`函数，函数将被并行地执行2次。方括号中第一个数字表示整个grid有多少个block，方括号中第二个数字表示一个block有多少个thread
  - GPU核函数的启动方式是**异步**的：启动GPU函数后，CPU不会等待GPU函数执行完毕才执行下一行代码。必要时，需要调用`cuda.synchronize()`，告知CPU等待GPU执行完核函数后，再进行CPU端后续计算。这个过程被称为**同步**，也就是GPU执行流程图中的红线部分



# CUDA编程基础知识

## Thread层次结构

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMTkxOS5wbmc?x-oss-process=image/format,png" style="zoom:50%;" />

CUDA将核函数所定义的运算称为**线程（Thread）**，多个线程组成一个**块（Block）**，多个块组成**网格（Grid）**。这样一个grid可以定义成千上万个线程，也就解决了并行执行上万次操作的问题

实际上，线程（thread）是一个编程上的软件概念。从硬件来看，thread运行在一个CUDA核心上，多个thread组成的block运行在Streaming Multiprocessor，多个block组成的grid运行在一个GPU显卡上

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMTkyNi5wbmc?x-oss-process=image/format,png" style="zoom:80%;" />

CUDA提供了一系列内置变量，以记录thread和block的大小及索引下标。以`[2, 4]`这样的配置为例：`blockDim.x`变量表示block的大小是4，即每个block有4个thread，`threadIdx.x`变量是一个从0到`blockDim.x - 1`（4-1=3）的索引下标，记录这是第几个thread；`gridDim.x`变量表示grid的大小是2，即每个grid有2个block，`blockIdx.x`变量是一个从0到`gridDim.x - 1`（2-1=1）的索引下标，记录这是第几个block。某个thread在整个grid中的位置编号为：`threadIdx.x + blockIdx.x * blockDim.x`

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMTkzMy5wbmc?x-oss-process=image/format,png" style="zoom:50%;" />

在实际使用中，我们一般将CPU代码中互相不依赖的的`for`循环适当替换成CUDA代码。注意，当线程数与计算次数不一致时，一定要使用判断语句，以保证某个线程的计算不会影响其他线程的数据

## Block大小设置

- 不同的执行配置会影响GPU程序的速度，一般需要多次调试才能找到较好的执行配置，在实际编程中，执行配置`[gridDim, blockDim]`应参考下面的方法：
  - block运行在SM上，不同硬件架构（Turing、Volta、Pascal…）的CUDA核心数不同，一般需要根据当前硬件来设置block的大小`blockDim`，一个block中的thread数最好是32、128、256的倍数。注意，限于当前硬件的设计，block大小不能超过1024
  - grid的大小`gridDim`，即一个grid中block的个数可以由总次数`N`除以`blockDim`，并向上取整

例如，我们想并行启动1000个thread，可以将blockDim设置为128，`1000 ÷ 128 = 7.8`，向上取整为8。使用时，执行配置可以写成`gpuWork[8, 128]()`，CUDA共启动`8 * 128 = 1024`个thread，实际计算时只使用前1000个thread，多余的24个thread不进行计算

`blockDim`是block中thread的个数，一个block中的`threadIdx`最大不超过`blockDim`；`gridDim`是grid中block的个数，一个grid中的`blockIdx`最大不超过`gridDim`

block和grid大小均是一维，实际编程使用的执行配置常常更复杂，block和grid的大小可以设置为二维甚至三维，如下图所示

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMTk1NC5wbmc?x-oss-process=image/format,png" style="zoom:40%;" />

## 内存分配

GPU计算时直接从显存中读取数据，因此每当计算时要将数据从主存拷贝到显存上，用CUDA的术语来说就是要把数据从主机端拷贝到设备端。CUDA强大之处在于它能自动将数据从主机和设备间相互拷贝，不需要程序员在代码中写明。这种方法对编程者来说非常方便，不必对原有的CPU代码做大量改动。

以一个向量加法为例，代码及其优化

- 实验结果可以发现，GPU比CPU慢，原因主要在于：
  - 向量加法的这个计算比较简单，CPU的numpy已经优化到了极致，无法突出GPU的优势，我们要解决实际问题往往比这个复杂得多，当解决复杂问题时，优化后的GPU代码将远快于CPU代码
  - 这份代码使用CUDA默认的统一内存管理机制，没有对数据的拷贝做优化。CUDA的统一内存系统是当GPU运行到某块数据发现不在设备端时，再去主机端中将数据拷贝过来，当执行完核函数后，又将所有的内存拷贝回主存。在上面的代码中，输入的两个向量是只读的，没必要再拷贝回主存
  - 这份代码没有做流水线优化。CUDA并非同时计算2千万个数据，一般分批流水线工作：一边对2000万中的某批数据进行计算，一边将下一批数据从主存拷贝过来。计算占用的是CUDA核心，数据拷贝占用的是总线，所需资源不同，互相不存在竞争关系。这种机制被称为流水线
- Numba对Numpy的比较友好，编程中一定要使用Numpy的数据类型。用到的比较多的内存分配函数有
  - `cuda.device_array()`： 在设备上分配一个空向量，类似于`numpy.empty()`
  - `cuda.to_device()`：将主机的数据拷贝到设备
  - `cuda.copy_to_host()`：将设备的数据拷贝回主机

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMjAwNS5qcGc?x-oss-process=image/format,png" style="zoom:60%;" />





