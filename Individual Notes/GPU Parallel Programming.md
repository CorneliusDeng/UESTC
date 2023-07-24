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

以一个向量加法为例，[代码及其优化](https://github.com/CorneliusDeng/UESTC/blob/main/CUDA%20Code/vector_addition.ipynb)

- 实验结果可以发现，GPU比CPU慢，原因主要在于：
  - 向量加法的这个计算比较简单，CPU的numpy已经优化到了极致，无法突出GPU的优势，我们要解决实际问题往往比这个复杂得多，当解决复杂问题时，优化后的GPU代码将远快于CPU代码
  - 这份代码使用CUDA默认的统一内存管理机制，没有对数据的拷贝做优化。CUDA的统一内存系统是当GPU运行到某块数据发现不在设备端时，再去主机端中将数据拷贝过来，当执行完核函数后，又将所有的内存拷贝回主存。在上面的代码中，输入的两个向量是只读的，没必要再拷贝回主存
  - 这份代码没有做流水线优化。CUDA并非同时计算2千万个数据，一般分批流水线工作：一边对2000万中的某批数据进行计算，一边将下一批数据从主存拷贝过来。计算占用的是CUDA核心，数据拷贝占用的是总线，所需资源不同，互相不存在竞争关系。这种机制被称为流水线
- Numba对Numpy的比较友好，编程中一定要使用Numpy的数据类型。用到的比较多的内存分配函数有
  - `cuda.device_array()`： 在设备上分配一个空向量，类似于`numpy.empty()`
  - `cuda.to_device()`：将主机的数据拷贝到设备
  - `cuda.copy_to_host()`：将设备的数据拷贝回主机

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIwLTAyMjAwNS5qcGc?x-oss-process=image/format,png" style="zoom:60%;" />



# CUDA优化方向

CPU + GPU 是一种异构计算的组合，各有独立的内存，GPU的优势是更多的计算核心。该架构在并行计算上有很大优势，但是数据需要从主机和设备间相互拷贝，会造成一定的延迟。因此，要从下面两个方面来优化GPU程序

- 充分利用GPU的多核心，最大化并行执行度
- 优化内存使用，最大化数据吞吐量，减少不必要的数据拷贝

[Code_Path]()

## 并行计算优化

### 网格跨度

CUDA的执行配置：`[gridDim, blockDim]`中的`blockDim`最大只能是1024，英伟达给出的官方回复是`gridDim`最大为一个32位整数的最大值，也就是2,147,483,648，大约二十亿。这个数字已经非常大了，足以应付绝大多数的计算，但是如果对并行计算的维度有更高需求呢？网格跨度有更好的并行计算效率

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MDk1Mi5wbmc?x-oss-process=image/format,png" style="zoom:40%;" />

这里仍然以`[2, 4]`的执行配置为例，该执行配置中整个grid只能并行启动8个线程，假如我们要并行计算的数据是32，会发现后面8号至31号数据共计24个数据无法被计算

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTEyMC5wbmc?x-oss-process=image/format,png" style="zoom:40%;" />

我们可以在0号线程中，处理第0、8、16、24号数据，就能解决数据远大于执行配置中的线程总数的问题，用程序表示，就是在核函数里再写个for循环。注意，跨步大小为网格中线程总数，用`gridDim.x * blockDim.x`来计算。`for`循环的step是网格中线程总数，这也是为什么将这种方式称为**网格跨步**。如果网格总线程数为1024，那么0号线程将计算第0、1024、2048…号的数据。这里我们也不用再明确使用`if (idx < N)`来判断是否越界，因为`for`循环也有这个判断。

- 使用网格跨步的优势主要有：
  - 扩展性：可以解决数据量比线程数大的问题
  - 线程复用：CUDA线程启动和销毁都有开销，主要是线程内存空间初始化的开销；不使用网格跨步，CUDA需要启动大于计算数的线程，每个线程内只做一件事情，做完就要被销毁；使用网格跨步，线程内有`for`循环，每个线程可以干更多事情，所有线程的启动销毁开销更少
  - 方便调试：我们可以把核函数的执行配置写为`[1, 1]`，那么核函数的跨步大小就成为了1，核函数里的`for`循环与CPU函数中顺序执行的`for`循环的逻辑一样，非常方便验证CUDA并行计算与原来的CPU函数计算逻辑是否一致

### 多流

GPU最多就上千个核心，同一时间只能并行执行上千个任务。当我们处理千万级别的数据，整个大任务无法被GPU一次执行，所有的计算任务需要放在一个队列中，排队顺序执行。CUDA将放入队列顺序执行的一系列操作称为**流（Stream）**

- 由于异构计算的硬件特性，CUDA中以下操作是相互独立的，通过编程，是可以操作他们并发地执行的：
  - 主机端上的计算
  - 设备端的计算（核函数）
  - 数据从主机和设备间相互拷贝
  - 数据从设备内拷贝或转移
  - 数据从多个GPU设备间拷贝或转移

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTE1NS5wbmc?x-oss-process=image/format,png" style="zoom:70%;" />

针对这种互相独立的硬件架构，CUDA使用多流作为一种高并发的方案：把一个大任务中的上述几部分拆分开，放到多个流中，每次只对一部分数据进行拷贝、计算和回写，并把这个流程做成流水线。因为数据拷贝不占用计算资源，计算不占用数据拷贝的总线（Bus）资源，因此计算和数据拷贝完全可以并发执行。如下图所示，将数据拷贝和函数计算**重叠**起来的，形成流水线，能获得非常大的性能提升。实际上，流水线作业的思想被广泛应用于CPU和GPU等计算机芯片设计上，以加速程序。

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTE1OS5wbmc?x-oss-process=image/format,png" style="zoom:70%;" />

以向量加法为例，上图中第一行的Stream 0部分是[之前采用的逻辑](https://github.com/CorneliusDeng/UESTC/blob/main/CUDA%20Code/vector_addition.ipynb)，没有使用多流技术，程序的三大步骤是顺序执行的：先从主机拷贝初始化数据到设备（Host To Device）；在设备上执行核函数（Kernel）；将计算结果从设备拷贝回主机（Device To Host）。当数据量很大时，每个步骤的耗时很长，后面的步骤必须等前面执行完毕才能继续，整体的耗时相当长。以2000万维的向量加法为例，向量大约有几十M大小，将整个向量在主机和设备间拷贝将占用占用上百毫秒的时间，有可能远比核函数计算的时间多得多。将程序改为多流后，每次只计算一小部分，流水线并发执行，会得到非常大的性能提升。

- 默认情况下，CUDA使用0号流，又称默认流。不使用多流时，所有任务都在默认流中顺序执行，效率较低。多流存在一些规则：
  - 给定流内的所有操作会按序执行
  - 非默认流之间的不同操作，无法保证其执行顺序
  - 所有非默认流执行完后，才能执行默认流；默认流执行完后，才能执行其他非默认流

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTIwNS5wbmc?x-oss-process=image/format,png" style="zoom:50%;" />

- 参照上图，可将这三个规则解释为：
  - 非默认流1中，根据进流的先后顺序，核函数1和2是顺序执行的
  - 无法保证核函数2与核函数4的执行先后顺序，因为他们在不同的流中。他们执行的开始时间依赖于该流中前一个操作结束时间，例如核函数2的开始依赖于核函数1的结束，与核函数3、4完全不相关
  - 默认流有阻塞的作用。如图中红线所示，如果调用默认流，那么默认流会等非默认流都执行完才能执行；同样，默认流执行完，才能再次执行其他非默认流

可见，某个流内的操作是顺序的，非默认流之间是异步的，默认流有阻塞作用

如果想使用多流时，必须先定义流：`stream = numba.cuda.stream()`

CUDA的数据拷贝以及核函数都有专门的`stream`参数来接收流，以告知该操作放入哪个流中执行：`numba.cuda.to_device(obj, stream=0, copy=True, to=None)`，`numba.cuda.copy_to_host(self, ary=None, stream=0`

核函数调用的地方除了要写清执行配置，还要加一项`stream`参数：`kernel[blocks_per_grid, threads_per_block, stream=0]`

根据这些函数定义也可以知道，不指定`stream`参数时，这些函数都使用默认的0号流

## 内存优化

CPU和GPU组成异构计算架构，如果想从内存上优化程序，我们必须尽量减少主机与设备间的数据拷贝，并将更多计算从主机端转移到设备端。尽量在设备端初始化数据，并计算中间数据，并尽量不做无意义的数据回写

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTIxOS5wbmc?x-oss-process=image/format,png" style="zoom:100%;" />

GPU的内存结构如上图所示：GPU的计算核心都在Streaming Multiprocessor（SM）上，Multiprocessor里有计算核心可直接访问的寄存器（Register）和共享内存（Shared Memory）；多个SM可以读取显卡上的显存，包括全局内存（Global Memory）。每个Multiprocessor上的Shared Memory相当于该Multiprocessor上的一个缓存，一般都很小，GPU Telsa V100的Shared Memory也只有96KB。注意，Shared Memory和Global Memory的字面上都有共享的意思，但是不要将两者的概念混淆，Shared Memory离计算核心更近，延迟很低；Global Memory是整个显卡上的全局内存，延迟高。

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTIyNS5wbmc?x-oss-process=image/format,png" style="zoom:70%;" />

从软件角度来看，CUDA的线程可以访问不同级别的存储，每个Thread有独立的私有内存；每个Block中多个Thread都可以在该Block的Shared Memory中读写数据；整个Grid中所有Thread都可以读写Global Memory。Shared Memory的读写访问速度会远高于Global Memory。内存优化一般主要利用Shared Memory技术。下文将以矩阵乘法为例，展示如何使用Shared Memory来优化程序。

### 二维和三维执行配置

`threadIdx` 和`blockIdx`变量都是一维的，实际上，CUDA允许这两个变量最多为三维，一维、二维和三维的大小配置可以适应向量、矩阵和张量等不同的场景

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTIzMS5wbmc?x-oss-process=image/format,png" style="zoom:30%;" />

一个二维的执行配置如上图所示，其中，每个block有(3 * 4)个Thread，每个grid有(2 * 3)个Block。 二维块大小为 *(Dx, Dy)*，某个线程号 *(x, y)* 的公式为 **(x + y Dx)**；三维块大小为 *(Dx, Dy, Dz)*，某个线程号*(x, y, z)* 的公式为 **(x + y Dx + z Dx Dy)**。各个内置变量中`.x` `.y`和`.z`为不同维度下的值

例如，一个二维配置，某个线程在矩阵中的位置可以表示为：`col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y`,`row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x`

如何将二维Block映射到自己的数据上并没有固定的映射方法，一般情况将`.x`映射为矩阵的行，将`.y`映射为矩阵的列。Numba提供了一个更简单的方法帮我们计算线程的编号：`row, col = cuda.grid(2)`，其中，参数2表示这是一个2维的执行配置。1维或3维的时候，可以将参数改为1或3。

对应的执行配置也要改为二维：`threads_per_block = (16, 16)`，`blocks_per_grid = (32, 32)`，`gpu_kernel[blocks_per_grid, threads_per_block]`

`(16, 16)`的二维Block是一个常用的配置，共256个线程。每个Block的Thread个数最好是128、256或512，这与GPU的硬件架构高度相关

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTI0MS5wbmc?x-oss-process=image/format,png" style="zoom:100%;" />

一个`C = AB`的矩阵乘法运算，需要我们把A的某一行与B的某一列的所有元素一一相乘，求和后，将结果存储到结果矩阵C的(row, col)上。在这种实现中，每个线程都要读取A的一整行和B的一整列，共计算M行*P列。以计算第row行为例，计算C[row, 0]、C[row, 1]…C[row, p-1]这些点时都需要从Global Memory中把整个第row行读取一遍。可以算到，A矩阵中的每个点需要被读 B.width 次，B矩阵中的每个点需要被读 A.height 次。这样比较浪费时间。因此，可以将多次访问的数据放到Shared Memory中，减少重复读取的次数，并充分利用Shared Memory的延迟低的优势。

### Shared Memory

这个实现中，跟未做优化的版本相同的是，每个Thread计算结果矩阵中的一个元素，不同的是，每个CUDA Block会以一个 BLOCK_SIZE * BLOCK_SIZE 子矩阵为基本的计算单元。具体而言，需要声明Shared Memory区域，数据第一次会从Global Memory拷贝到Shared Memory上，接下来可多次重复利用Shared Memory上的数据。

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL2FpeGluZ3FpdS0xMjU4OTQ5NTk3LmNvcy5hcC1iZWlqaW5nLm15cWNsb3VkLmNvbS8yMDE5LTExLTIxLTA3MTI0Ny5wbmc?x-oss-process=image/format,png" style="zoom:100%;" />

- 总结
  - 声明Shared Memory。代码中使用了`cuda.shared.array(shape,type)`，shape为这块数据的向量维度大小，type为Numba数据类型，例如是int32还是float32。这个函数只能在设备端使用。定义好后，这块数据可被同一个Block的所有Thread共享。需要注意的是，这块数据虽然在核函数中定义，但它不是单个Thread的私有数据，它可被同Block中的所有Thread读写
  - 数据加载。每个Thread会将A中的一个元素加载到sA中，一个Block的 BLOCK_SIZE x BLOCK_SIZE 个Thread可以把sA填充满。`cuda.syncthreads()`会等待Block中所有Thread执行完之后才执行下一步。所以，当执行完这个函数的时候，sA和sB的数据已经拷贝好了
  - 数据复用。A中的某个点，只会被读取 B.width / BLOCK_SIZE 次；B中的某个点，只会被读 A.height / BLOCK_SIZE 次。`for n in range(BLOCK_SIZE)`这个循环做子矩阵向量乘法时，可多次复用sA和sB的数据
  - 子矩阵的数据汇总。我们以一个 BLOCK_SIZE x BLOCK_SIZE 的子矩阵为单位分别对A从左到右，对B从上到下平移并计算，共循环 A.width / BLOCK_SIZE 次。在某一步平移，会得到子矩阵的点积。`for m in range(math.ceil(A.shape[1] / BLOCK_SIZE))`这个循环起到了计算A从左到右与B从上到下点积的过程