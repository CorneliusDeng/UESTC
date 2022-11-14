# Introduction

- **Why do we need Distributed System?**

  Functional Separation（功能分离）、Inherent distribution（固有的分布性）、Power imbalance and load variation（负载均衡）、Reliability（可靠性）、Economies（经济性）

- **goal：**资源共享 (resource sharing)、协同计算 (collaborative computing)

- **definition：**A distributed system is one in which components located at networked computers communicate and coordinate their actions only by passing messages.

- **fundamental feature：**并发性 (concurrency)、无全局时钟 (non-global clock)、故障独立性 (independent failure)

- **challenge：**异构性（Heterogeneity）、开放性（Openness）、安全性（Security）、可伸缩性（Scalability）、故障处理（Failure handling）、并发行（Concurrency）、透明性（Transparency）

- **RPC：**Remote Procedure Call



# 系统模型

## **物理模型physical models**

考虑组成系统的计算机和设备的类型以及它们的互连，不涉及特定的技术细节，从计算机和所用网络技术的特定细节中抽象出来的分布式系统底层硬件元素的表示

早期、互联网规模的分布式系统特点：静态Static、分立discrete和自治autonomous 

## **体系结构模型architectural models**

从系统的计算元素执行的计算和通信任务方面来描述系统，一个系统的体系结构是用独立指定的组件以及这些组件之间的关系来表示的结构。

体系结构元素：分布式系统的基础构建块。体系结构模式：构建在体系结构元素之上，提供组合的、重复出现的结构。

通信范型：1⃣️进程间通信：相对底层的支持（套接字、多播、消息传递）。2⃣️远程调用：最常见的通信范型，双向交换（请求应答协议、远程过程调用RPC、远程方法调用RMI）。3⃣️间接通信：空间、时间解耦合（组通信、发布-订阅系统、消息队列、分布式共享内存DSM、元祖空间）。

角色和责任：1⃣️客户—服务器 Client-Server/CS/BS。2⃣️对等体系结构 Peer-to-Peer/P2P。

放置：对象或服务等实体如何映射到底层的物理分布式设施上。

常见的放置策略：1⃣️将服务映射到多个服务器。2⃣️缓存：保存最近使用过的数据，在本地或代理服务器上；减少不必要的网络传输，减少服务器负担，还可以代理其它用户透过防火墙访问服务器。3⃣️移动代码：将代码下载到客户端运行，可以提高交互的效率。4⃣️移动代理：一个运行的程序在网络上的计算机之间穿梭，并执行的代码，代替一些机器执行任务。

体系结构模式：构建在相对原始的体系结构元素之上，提供组合的、重复出现的结构。

关键体系结构模型：1⃣️分层体系结构（layering architecture）：一个复杂的系统被分成若干层，每层利用下层提供的服务。2⃣️层次化体系结构（tiered architecture）：与分层体系结构互补，是一项组织给定层功能的技术。3⃣️瘦客户（thin client）：本地只是一个 GUI，应用程序在远程计算机上执行。

## **基础模型fundamental models**

采用抽象的观点描述大多数分布式系统面临的单个问题的方案，对体系结构模型中公共属性的一种更为形式化的描述。

一、交互模型：延迟的不确定性及缺乏全局时钟的影响，处理消息发送的性能问题，解决在分布式系统中设置时间限制的难题。两个变体：1⃣️同步分布式系统：有严格的时间限制假设；2⃣️异步分布式系统：无严格的时间限制假设，非常常见。

二、故障模型：组件、网络等故障的定义、分类，试图给出进程和信道故障的一个精确的约定。定义了什么是可靠通信、正确的进程和能出现的故障的形式，为分析故障带来的影响提供依据。故障分类：1⃣️遗漏故障Omission failures：进程或者通信通道没有正常的工作。2⃣️随机故障（拜占庭故障）Arbitrary failures：对系统影响最大的一种故障形式，而且错误很难探知，随机遗漏应有的处理步骤或进行不应有的处理步骤，该做的不做，不该做的却做了。3⃣️时序故障Timing failures：仅仅发生在同步分布式系统中，异步系统无时间保证，故不存在此类故障

三、安全模型：内、外部攻击定义、分类，讨论对进程的和信道的各种可能的威胁。引入安全通道的概念，安全模型的目的是提供依据，以此分析系统可能受到的侵害，并在设计系统时防止这些侵害的发生。



# 时间和全局状态

计算机时钟：晶体具有固定震荡频率，硬件时钟：𝐻𝑖 (𝑡)，软件时钟：𝐶i (𝑡)=𝑎𝐻𝑖 (𝑡)+𝑏
时钟漂移：振荡频率变化。电源稳定性，环境温度等
时钟偏移不可避免

## **同步物理时钟**

外部同步 External Sync：采用权威的外部时间源，时钟Ci在范围D内是准确的

内部同步 Internal Sync：无外部权威时间源，系统内时钟同步，时钟Ci在范围D内是准确的

若P(时钟集合)在范围D内外部同步，则P在范围2D内内部同步

- **Cristian方法：**

  ​	应用条件：存在时间服务器，作为外部时间源；消息往返时间与系统所要求的精度相比足够短

  ​	协议：进程p根据消息mr，mt计算消息往返时间Tround；根据服务器在mt中放置的时间t设置时钟为：t+Tround/2

  ​	若消息的最小传输时间为min，则精度为：+-(Tround/2 – min)


  - **Berkeley算法：**主机周期轮询从属机时间；主机通过消息往返时间估算从属机的时间（与Cristian方法类似）；主机计算容错平均值；主机发送每个从属机的调整量

- **网络时间协议(Network Time Protocol，NTP)：**可外部同步：使得跨Internet的用户能精确地与UTC (通用协调时间)同步；高可靠性：可处理连接丢失，采用冗余服务器、路径等；扩展性好：大量用户可经常同步，以抵消漂移率的影响；安全性强：防止恶意或偶然的干扰
  - 协议结构：层次结构 —— strata；主服务器直接与外部UTC同步；同步子网可重新配置
  - NTP服务器同步模式：1⃣️组播模式 multicast/broadcast mode，适用于高速LAN，准确度较低，但效率高。2⃣️服务器/客户端模式 server/client mode，与Cristian算法类似，准确度高于组播。3⃣️对称模式 symmetric mode，保留时序信息，准确度最高。

## **逻辑时间和逻辑时钟**

逻辑时间的引入：1⃣️节点具有独立时钟，缺乏全局时钟；后发生的事件有可能赋予较早的时间标记。2⃣️分布式系统中的物理时钟无法完美同步，消息传输延迟的不确定性。3⃣️事件排序是众多分布式算法的基石；互斥算法、死锁检测算法。

逻辑时钟：众多应用只要求所有节点具有相同时间基准，该时间不一定与物理时间相同。

Lamport(1978)指出：不进行交互的两个进程之间不需要时钟同步。对于不需要交互的两个进程而言，即使没有时钟同步，也无法察觉，更不会产生问题。所有的进程需要在事件的发生顺序上达成一致。

并发关系定义：X||Y：X→ Y 与 Y→ X均不成立，则称事件X、Y是并发的

- **Lamport时钟：**

  ​	机制：进程维护一个单调递增的软件计数器，充当逻辑时钟；用逻辑时钟为事件添加时间戳；按事件的时间戳大小为事件排序。对于不需要交互的两个进程而言，即使没有时钟同步，也无法察觉，更不会产生问题；所有的进程需要在事件的发生顺序上达成一致。
  ​	逻辑时钟修改规则：

  ​		LC1: 进程pi执行(issued)事件前，逻辑时钟Li = Li + 1
  ​		LC2：
  ​			a)进程pi发送消息m时，在m中添加时间戳t = Li
  ​			b)进程pj在接收(m, t)时，更新Lj = max(Lj, t)，执行LC1，即给事件recv(m)添加时间戳↔️Lj = max(Lj, t) + 1

  ​	应用：a→b ⇒ L(a) < L(b) ；但是，L(e) < L(b) ⇏ e→b

  ​	引入进程标示符创建事件的全序关系：1⃣️若e、e′分别为进程pi、pj中发生的事件，则其全局逻辑时间戳分别为(Ti, i)、(Tj, j)。2⃣️e→e′ ⟺ Ti<Tj || (Ti=Tj && i<j)，该排序没有实际物理意义。


  - **向量时钟：**

    ​	克服Lamport时钟的缺点：L(e) < L(e′)不能推出e→e′

    ​	与 Lamport clock的区别：提前 +1

    ​	物理意义：观察到的对应进程的最新状态，有全局信息。而Lamport时钟只有部分全局信息。

    ​	每个进程维护自己的向量时钟Vi

    ​		VC1：初始情况下，Vi[j]=0,i,j=1,2,...N.
    ​		VC2：在pi给事件加时间戳之前，设置Vi[i]= Vi[i]+1。
    ​		VC3：pi在它发送的每个消息中包括t＝Vi。
    ​		VC4：当pi接收到消息中的时间戳t时，设置Vi[j]=max(Vi[j],t[j]),j=1,2,...,N。取两个向量时间戳的最大值称为合并操作。
    
    - 定义
      - 结论：V1 = V2。前提：iff  V1[i] = V2[i], i = 1, … , n
      - 结论：V1 ≤ V2。前提：iff  V1[i] ≤ V2[i], i = 1, … , n
      - 结论：V1 < V2。前提：iff  V1 ≤ V2  ∧  V1 ≠ V2 
      - 结论：V1 || V2。前提：iff  not (V1 ≤ V2  or  V2 ≤ V1)
    - 可证明
      - e=e’ ⇔ V(e) = V(e’)
      - e→e’ ⇔ V(e) < V(e’)
      - e||e’ ⇔ V(e) || V(e’)

## **全局状态**

- **观察全局状态的必要性**

  - 分布式无用单元的收集

    - 基于对象的引用计数

    - 必须考虑信道和进程的状态

  - 分布式死锁检测
    - 观察系统中的“等待”关系图中是否存在循环
  - 分布式终止检测
    - 进程的状态有关——“主动”或“被动”（中途消息）
  - 分布式调试
    - 需要收集同一时刻系统中分布式变量的数值

- **全局状态和一致割集**

  观察进程集的状态——全局状态非常困难。根源：缺乏全局时间

  进程的历史，hi = <e_i^0,e_i^1,e_i^2…>

  进程历史的有限前缀h_i^k= < e_i^0,e_i^1,…,e_i^k >

  - 全局历史——单个进程历史的并集
    - H = h1 U h2 U… U hN
    - 所有进程所有事件的集合
  - 进程状态
    - s_i^k : 进程pi在第k个事件发生之前的状态
  - 全局状态——单个进程状态的集合  S = (s1, s2, … sN)
  - 割集——系统全局历史的子集（部分事件的集合，有序）
    - C = <h1^c1,h2^c2,…,hn^cn >
  - 割集的一致性
    - 割集C是一致的:  对于所有事件e ∈ C, f → e ⇒ f ∈ C （解释：如果后续事件在割集里，那么前序事件也在割集里）
    - 隐含表达：e∉C，则f不必属于C
    - 一致的全局状态——对应于一致割集的状态
    - 走向(Run): 往前走一步（可能有多种选择）形成时间的序列（状态的变化）→ 系统的一种可能执行过程
  - 全局历史中所有事件的全序
    - 与每个本地历史顺序一致
    - 不是所有的走向都经历一致的全局状态
  - 线性化（一致性）走向 linearization / consistent run
    - 所有的线性化走向只经历一致的全局状态
    - 若存在一个经过S和S’的线性化走向，则状态S’是从S可达

- **Chandy和Lamport的“快照”算法**

  - 目的：捕获一致的全局状态

  - 假设
    - 进程和通道均不会出现故障
    - 单向通道，提供FIFO顺序的消息传递
    - 进程之间存在强连通关系
    - 任一进程可在任一时间开始全局拍照
    - 拍照时，进程可继续执行，并发送和接收消息

  - 算法基本思想
    - 接入通道+外出通道
    - 进程状态+通道状态
    - 标记消息 marker message (flush message)
      - 标记接收规则：强制进程记录下自己的状态之后但在它们发送其他消息前发送一个标记，并记录接入通道消息
      - 标记发送规则：强制没有记录状态的进程去记录状态+清空信道

  - 进程pi的标记接收规则

    pi接收通道c上的标记消息：    

    - if (pi还没有记录它的状态) 
      - pi记录它的进程状态；// sender rule         
      - 将c的状态记成空集；         
      - 开始记录从其他接入通道上到达的消息   
    - else         
      - pi把c的状态记录到从保留它的状态以来它在c上接收到的消息集合中     

  - Propagating a snapshot

    - For all processes Pj (including the initiator), consider a message on channel Ckj

    - If we see marker message for the first time
      - Pj records own state and marks Ckj as empty
      - Send the marker message to all other processes (using N-1 outbound channels)
      - Start recording all incoming messages from channels Clj for l not equal to j (all other N-2 inbound ones)

    - Else
      - add all messages from inbound channels since we began recording to their states

  - 进程pi的标记发送规则：Initiator

    在pi记录了它的状态之后，对每个外出通道c:      

    ​	 (在pi从c上发送任何其他消息前)       

    ​	  pi在c上发送一个消息标记

  - Initiating a snapshot
    - Let’s say process Pi initiates the snapshot
    - Pi records its own state and prepares a special marker message (distinct from application messages)
    - Send the marker message to all other processes (using N-1 outbound channels)
    - Start recording all incoming messages from channels Cji for j not equal to i

  - 算法终止分析

    - 假设：一个进程已经收到了一个标记信息，在有限的时间内记录了它的状态，并在有限的时间里通过每个外出通道发送了标记信息
    - 若存在一条从进程pi到进程pj的信道，那么pi记录它的状态之后的有限时间内pj将记录它的状态
    - 进程和通道图是强连通的，因此在一些进程记录它的状态之后的有限时间内，所有进程将记录它们的状态和接入通道的状态
    - 每个进程收到在它所有的输入通道上的标记之后终止

  - Terminating a snapshot
    - All processes have received a marker (and recorded their own state)
    - All processes have received a marker on all the N-1 incoming channels (and recorded their states)
    - Later, a central server can gather the partial state to build a global snapshot

  - 快照算法记录的全局状态是一致的，对于所有事件ej ∈ C, ei → ej ⇒ ei ∈ C 

    - 设ei、ej分别为进程pi、pj中的事件，且ei → ej，则: 若ej∈C ⇒ ei∈C。即如果ej在pj记录其状态之前发生，那么ei必在pi记录其状态之前发生。证明思路如下：
      - i=j时，显然成立
      - i≠j时。假设 ei→m1→m2→…→mn→ej为实际HB关系。若ei未被记录，则ei出现在pi记录状态且发送marker 之后，那么根据FIFO性质，每个recv(mi)都发生在此进程收到marker后，则ej也在marker后，故不被pj记录，矛盾。


## 分布式调试

目的：对系统实际执行中的暂态（一致性全局状态）作出判断

方法：监控器进程（收集进程状态信息）

- 全局状态谓词φ的判断
  - 可能的φ（possibly）
    - 存在一个一致的全局状态S，H的一个线性化走向经历了这个全局状态S，而且该S使得φ(s)为True
    - There exists a consistent run such that predicate φ holds in a global state of the run
  - 明确的φ（definitely）
    - 对于H的所有线性化走向L，存在L经历的一个一致的全局状态S(对不同L可以不同)，而且该S使得φ(s)为True
    - For every consistent run, there exists a global state of it in which predicate φ holds.

- 观察一致的全局状态
  - 进程的状态信息附有向量时钟值：monitor
  - 全局状态的一致性判断——CGS条件
    - 设S=(s1,s2,…,sN)是从监控器进程接收到的状态信息中得出的全局状态，V(si)是从pi接收到的状态si的向量时间戳，则S是一致的全局状态当且仅当：V(si)[i]>=V(sj)[i] 　(i,j = 1,2,…, N)
    - 即若一个进程的状态依赖于另一个进程的状态，则全局状态也包含了它所依赖的状态。



# 协调和协定

- 构建分布式系统的主要动力：资源共享和协作
- 分布式系统的进程需要协调动作和对共享资源达成协定
- 分布式中的协作
  - 互斥
  - 选举
  - 组播
    - 可靠性和排序语义
  - 进程间的协定
    - 共识和拜占庭协定
- 故障模型
  - 良性故障(fail stop)
  - 随机故障(arbitrary failure)
- 网络
  - 网络分区 partitioning
  - 非对称路由 asymmetric
  - 连接的非传递性 intransitive
- 故障检测器 failure detector
  - 不可靠的故障检测器 timeout,产生值:  Unsuspected和Suspected
  - 可靠的故障检测器,产生值:  Unsuspected和Failed

## 分布式互斥Distributed Mutual Exclusion

- 目的：仅基于消息投递，实现对资源的互斥访问

- 假设：异步系统(eventually deliver)，无故障进程，可靠的消息投递(no faked messages)

- 执行临界区的应用层协议
  - enter( )           //进入临界区——若必要，可以阻塞进入
  - resourceAccesses( )          //在临界区访问共享资源
  - exit( )              //离开临界区——其它进程现在可以进入
  
- 基本要求
  - 安全性 safety：在临界区内一次最多有一个进程可以执行
  - 活性 liveness：进入和离开临界区的请求最终成功执行
  - 顺序 order：进入临界区的顺序与进入请求的happen-before顺序一致
  
- 性能评价
  - 带宽消耗 bandwidth：在每个enter和exit操作中发送的消息数
  - 客户延迟 client delay：进程进入、退出临界区的等待时间（请求进入到进入，请求退出到退出的延迟）
  - 吞吐量 throughput：用一个进程离开临界区和下一个进程进入临界区之间的同步延迟（synchronization delay）来衡量这个影响，当同步延迟较短时，吞吐量较大
  
- **中央服务器算法 Central Server**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Central%20Server.png)

  - 基本思想：使用一个服务器来授予进入临界区的许可
  - 满足安全性和活性要求，但不满足顺序要求,异步系统请求顺序无法保证in-order

  - 性能
    - 带宽消耗
      - enter:２个消息，即请求消息+授权消息
      - exit:   1个消息，即释放消息
    - 客户延迟
      - enter: request + grant = 1 round-trip
      - exit: release 
    - 同步延迟
      - 1个消息的往返时间：1 release + 1 grant

- **基于环的算法 Ring-based**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Ring-based.png)

  - 基本思想：把进程安排在一个逻辑环中，通过获得在进程间沿着环单向（如顺时针）传递的消息为形式的令牌来实现互斥。Pass token along the ring，Retain or pass immediately

  - 满足安全性和活性要求，不满足顺序要求

  - 性能
    - 带宽消耗
      - 由于令牌的投递，会持续消耗带宽
    - 客户延迟 (enter)
      - Min: 0个消息，正好收到令牌
      - Max: N个消息，刚刚投递了令牌
    - 同步延迟
      - Min: 1个消息，进程依次进入临界区
      - Max: N个消息，一个进程连续进入临界区，期间无其他进程进入临界区
  
- **基于组播和逻辑时钟的算法 Multicast & LC**

  - 基本思想：进程进入临界区需要所有其它进程的同意，组播+应答
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Multicast%20%26%20LC.png)
  - 并发控制，采用Lamport clock避免死锁
  
  - 满足安全性、活性和顺序要求
  
  - 性能
    - 带宽消耗
      - enter: 2(N－1)，即(N－1)个请求、 (N－1)个应答
    - 客户延迟 (enter)
      - 1 round-trip  (multicast)
    - 同步延迟
      - 1个消息的传输时间（无exit，vs round-trip）
  
- **Maekawa投票算法 Maekawa voting**

  - 基本思想：进程进入临界区不需要所有进程同意（部分即可）
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Voting%20set.png)
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Maekawa%20voting.png)
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Maekawa%20voting%20Deadlock.png)
  - Maekawa算法改进后[Sanders 1987]可满足安全性、活性和顺序性，进程按→关系维护请求队列
  - 性能

    - 带宽消耗
      - 3√n：进入需要2√n个消息，退出需要√n个消息，Ricard: 2(N-1)
    - 客户延迟 (enter)
      - 1 round-trip 
    - 同步延迟
      - 1 round-trip

## 选举Election

- 基本概念
  - 选举算法：选择一个唯一的进程来扮演特定角色的算法
  - 召集选举：一个进程启动了选举算法的一次运行
  - 参与者：进程参加了选举算法的某次运行
  - 非参与者：进程当前没有参加任何选举算法
  - 进程标识符：唯一且可按全序排列的任何数值
  
- 基本要求
  - 安全性：参与进程Pi的electedi =⊥或electedi = P（有效进程pid最大值）
    - 当进程第一次成为一次选举的参与者时，它把变量值置为特殊值“⊥”，表示该值还没有定义
  - 活性：所有进程Pi都参加并且最终置electedi ≠⊥或进程Pi崩溃
  
- 性能评价
  - 带宽消耗：与发送消息的总数成比例
  - 周转时间：从启动算法到终止算法之间的串行消息传输的次数
  
- **基于环的选举算法 Ring-based Election**

  - 基本思想：按逻辑环排列一组进程，id不必有序
  - 目的：在异步系统中选举具有最大标识符的进程作为协调者
  - 算法过程
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Ring-based%20Election.png)
  - 算法示例：选举从进程17开始。到目前为止，所遇到的最大的进程标识符是24。参与的进程用深色表示
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Ring-based%20Election%20Example.png)
  - 性能
    - 最坏情况：启动选举算法的逆时针邻居具有最大标识符，共计需要3N-1个消息，周转时间为3N-1
      - 到达该邻居需要N-1个消息，并且还需要N个消息才能完成一个回路，才能宣布它当选，接着当选消息被发送N-1次
    - 最好情况：周转时间为2N
  - 该算法不具备容错功能

- **霸道算法 Bully Election**

  - 为什么叫霸道算法

    When a process is started to replace a crashed process, it begins an election. If it has the highest process identifier, then it will decide that it is the coordinator and announce this to the other processes. Thus it will become the coordinator, even though the current coordinator is functioning. It is for this reason that the algorithm is called the ‘bully’ algorithm.

  - 假定

    - 同步系统，使用超时检测进程故障
    - 通道可靠，但允许进程崩溃
    - 每个进程知道哪些进程具有更大的标识符
    - 每个进程均可以和所有其它进程通信

  - 该算法存在3种类型的消息：

    - 选举消息用于宣布选举，应答消息用于回复选举消息，协调者消息用于宣布当选进程的身份：新的“协调者”
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Bully%20Election.png)

  - 算法示例：p4、p3相继出现故障的选举过程，p1先发现
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Bully%20Election%20Example.png)

  - 性能

    - 最好情况：标识符次大的进程发起选举，发送N-2个协调者消息，周转时间为1个消息
    - 最坏情况：标识符最小的进程发起选举，然后N-1个进程一起开始选举，周转时间=依次elect(N-2)+协调者消息


## 组播通信 Multicast

- 组播/广播
  - 组播：发送一个消息给进程组中的每个进程
  - 广播：发送一个消息给系统中的所有进程
- 组播面临的挑战
  - 效率
    - 带宽使用
    - 总传输时间
  - 投递保证
    - 可靠性
    - 顺序
  - 进程组管理
    - 进程可任意加入或退出进程组
- 系统模型
  - multicast(g, m)：进程发送消息给进程组g的所有成员
  - deliver(m)：投递由组播发送的消息到调用进程
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/System%20Modal.png)
- 封闭组和开放组
  - 封闭组：只有组的成员可以组播到该组
  - 开放组：组外的进程也可以向该组发送消息
- **基本组播 Basic Muticast**
  - 原语：B-multicast、B-deliver
    A correct process will eventually deliver the msg, as long as the multicaster does not crash.
  - 实现B-multicast的一个简单方法是使用一个可靠的一对一send操作
    - B-multicast(g, m)：对每个进程p∈g，send(p, m)
    - 进程p receive(m)时：p执行B-deliver(m)
- **可靠组播 Reliable Muticast**
  - 一个可靠组播应当满足如下性质
    - 完整性 Integrity：一个正确的进程p传递一个消息m至多一次
    - 有效性 Validity：如果一个正确的进程组播消息m，那么它终将传递m
    - 协定 Agreement：如果一个正确的进程投递消息m，那么group中其它正确的进程终将传递m
  - 与B-multicast的区别：B-multicast 不保证协定（sender中途失效）
- **用B-multicast实现可靠组播**
  - 基本思想
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Reliable%20Muticast.png)
  - 算法示例
    - 进程p崩溃，消息m没有投递到r和s进程
    - 然而，进程q继续投递消息m给正确的进程r和s
    - 此后，正确的进程r和s继续投递消息至组内其他进程
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/RM%20reply%20BM%20Example.png)
  - 算法评价
    - 满足完整性：B-multicast中的通信通道可靠以及B-multicast特性
    - 满足有效性：一个正确的进程终将B-deliver消息到它自己
    - 遵循协定：每个正确的进程在B-deliver消息中都B-multicast该消息到其它进程（sender没有crash，eventually deliver）
    - 效率低：每个消息被发送到每个进程|g|次，累计|g|^2次
- **用IP组播实现可靠组播**
  - 将IP组播、捎带确认法和否定确认相结合
    - 基于IP组播：IP组播通信通常是成功的
    - 捎带确认piggyback：在发送给组中的消息中捎带确认（已经收到了什么）
    - 否认确认negative acknowledgement：进程检测到有遗漏消息时，发送单独的应答（请求）消息
  - 算法思想
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/RM%20apply%20IPM.png)
  - 保留队列 hold-back queue
    - 保留队列并不是可靠性必须的，但它简化了协议，使我们能使用序号来代表已投递的消息集。也提供了投递顺序保证。
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/hold-back%20queue.png)
  - 算法评价
    - 完整性
      - 通过检测重复消息和IP组播性质实现
    - 有效性
      - 仅在IP组播具有有效性时成立
    - 协定
      - 需要：进程无限组播消息（保证有机会探测消息丢失，因为是被动NACK）+无限保留消息副本时成立（一旦收到NACK，重发）不现实
      - 某些派生协议实现了协定
- **统一性质 Uniform Properties**
  - 统一性质：无论进程是否正确都成立的性质
  - 统一协定 Uniform agreement：如果一个进程投递消息m，不论该进程是正确的还是出故障，在group(m)中的所有正确的进程终将投递m。统一协定允许一个进程在投递一个消息后崩溃。
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Uniform%20Properties.png)
- **有序组播**
  - **FIFO组播**
    - 如果一个正确的进程发出multicast(g, m)，然后发出multicast(g, m’)，那么每个投递m’的正确的进程将在m’前投递m
    - 保证每个进程发送的消息在其它进程中的接收顺序一致
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/FIFO%20Multicast.png)
    - 实现
      - 基于序号实现
      - FO-multicast/FO-deliver
      - 算法：与基于IP组播的可靠组播类似，即采用Sgp、Rgq和保留队列
  - **因果排序组播 causal ordering**
    - 如果multicast(g, m) → multicast(g, m’) ，那么任何投递m’的正确进程将在m’前投递m
    - C1→C3, C2||C3，区别于FIFO：跨进程
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/causal%20ordering.png)
    - 实现
      - 向量时钟，每个进程维护自己的向量时钟
      - CO-multicast：在向量时钟的相应分量上加1，附加VC到消息
      - CO-deliver：根据时间戳递交消息
      - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/causal%20ordering%20implementation.png)
  - **全排序组播 total ordering**
    - 如果一个正确的进程在投递m’前投递消息m，那么其它投递m’的正确进程将在m’前投递m
    - 所有进程对所有消息deliver顺序一致，不关心其它属性
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/total%20ordering.png)
    - 实现
      - 为组播消息指定全排序标识符 ，以便每个进程基于这些标识符做出相同的排序决定
      - 每个进程将<m, id(m)>放入保留队列，sequencer除外
      - sequencer维护一个组特定的序号Sg，用来给它B-deliver的消息指定连续且不断增加的序号。它通过给g发送B-deliver顺序消息来宣布序号。每个进程维护一个本地的rg
      - ISIS算法

## 共识和相关问题 Consensus

- 共识问题(consensus)

  - 一个或多个进程提议了一个值后，应达成一致意见
  - 共识问题、拜占庭将军和交互一致性问题

- 共识算法

  - 符号

    - pi: 进程i
    - vi: 进程pi的提议值（proposed value）
    - di: 进程pi的决定变量（decision value）

  - 基本要求

    - 终止性 Termination：每个正确的进程最终设置它的决定变量
    - 协定性 Agreement：如果pi和pj是正确的且已进入决定状态，那么di=dj，其中i, j=1,2,…N
    - 完整性/有效性 Integrity/Validity：如果正确的进程都提议了同一个值，那么处于决定状态的任何正确进程将选择该值

  - 步骤

    - 每个进程组播它的提议值

    - 每个进程收集其它进程的提议值

    - 每个进程计算V = majority(v1, v2, …, vN) 

    - majority()函数为抽象函数，可以是max()、min()等

  - 分析

    - 终止性：由组播操作的可靠性保证
    - 协定性和完整性：由majority()函数定义和可靠组播的完整性保证

- 拜占庭将军问题(Byzantine Generals)

  - 问题描述
    - 3个或更多将军协商是进攻还是撤退
    - 1个将军（司令）发布命令，其他将军决定是进攻还是撤退
    - 一个或多个将军可能会叛变
    - 所有未叛变的将军执行相同的命令 
  - 与共识问题的区别：一个独立的进程提供一个值，其他进程决定是否采用
  - 算法要求
    - 终止性：每个正确进程最终设置它的决定变量
    - 协定性：所有正确进程的决定值都相同　
    - 完整性：若司令正确，则所有正确进程采用司令提议的值

- 交互一致性(interactive consistency)

  - 每个进程都提供一个值，正确的进程最终就一个值向量达成一致
    决定向量: 向量中的每个分量与一个进程的值对应
  - 算法要求
    - 终止性：每个正确进程最终设置它的决定变量
    - 协定性：所有正确进程的决定向量都相同
    - 完整性：如果进程pi是正确的，那么所有正确的进程都把vi作为他们决定向量中第i个分量

- 共识问题与其他问题的关联

  - 目的：重用已有的解决方案
  - 问题定义
    - 共识问题C：Ci(v1,v2,…vN)返回进程pi的决定值
    - 拜占庭将军BG：BGi(j,v)返回进程pi的决定值，其中pj是司令，它建议的值是v
    - 交互一致性问题IC：ICi(v1,v2,…,vN)[j]返回进程pi的决定向量的第j个分量
  - 从BG构造IC
    - 将BG算法运算N次，每次都以不同的进程pj作为司令
    - ICi(v1,v2, …, vN )[j] = BGi(j,vj), (i, j = 1, 2, …, N)
  - 从IC构造C
    - Ci(v1,v2, …, vN ) = majority(ICi(v1,v2, …, vN )[1],…,  ICi(v1,v2, …, vN )[N])
  - 从C构造BG
    - 令进程pj把它提议的值v发送给它自己以及其余进程
    - 所有的进程都用它们收到的那组值v1,v2, …, vN作为参数运行C算法
    - BGi(j,vj) =Ci(v1,v2, …, vN ), (i = 1, 2, …, N)

- 同步系统中的共识问题

  - 故障假设：N个进程中最多有f个进程会出现崩溃故障
  - 算法
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Synchronization%20Consensus.png)
  - 算法分析
    - 终止性质：由同步系统保证
    - 协定性和完整性
      - 假设pi得到的值是v，而pj不是，则存在pk，pk在把v传送给pi后，还没来得及传送给pj就崩溃了
      - 每个回合至少有一个进程崩溃，但假设至多有f个进程崩溃
      - 而进行了f+1回合（即便只剩一个回合，B-multicast的语义可以保证所有correct process看到的集合一致），因此得出矛盾

- **Paxos : CFT Consensus**

  - Goal：Allow a group of processes to agree on a value

  - Requirements

    - Safety
      - Only a value that has been proposed may be chosen.
      - Only a single value is chosen.
      - A node never learns that a value has been chosen unless it actually has been.

    - Liveness (enough processes remain up-and-running)
      - Some proposed value is eventually chosen.
      - If a value has been chosen, a node can eventually learn the value.

  - Assumptions

    - The distributed system is partially synchronous (in fact, it may even be asynchronous).
    - Communication between processes may be unreliable, meaning that messages may be lost, duplicated, delayed or reordered.
    - Messages that are corrupted can be detected as such (and thus subsequently ignored).
    - Processes may exhibit crash failures, but not arbitrary failures, nor do processes collude.

  - Quorum-Based Consensus：not guarantee better result

    - Create a fault-tolerant consensus algorithm that does not block if a majority of processes are working
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Quorum-Based%20Consensus.png)

  - Paxos players

    - Client：makes a request

    - Proposers：Get a request from a client and run the protocol to get everyone in the cluster to agree

    - Acceptors：Multiple processes that remember the state of the protocol，Quorum = any majority of acceptors

    - Learners：Accept agreement from majority of acceptors，Execute the request and/or sends a response back to the client

    - Proposal：An alternative proposed by a proposer. Consists of a unique number and a proposed value 

      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Paxos%20players.png)
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Paxos%20workflow.png)

  - Basic Paxos Algorithm

    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Basic%20Paxos%20Algorithm%201.png)
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Basic%20Paxos%20Algorithm%202.png)

  - Paxos: keep trying if you need to

    - A proposal N may fail because
      - The acceptor may have made a new promise to ignore all proposals less than some value M >N
      - A proposer does not receive a quorum of responses: either promise or accept

- BFT共识：BGP

  - 随机故障假设：N个进程中最多有f个进程会出现随机故障
  - N≤3f：无解决方法
    - 将N个将军分成3组，n1+n2+n3=N，且n1, n2, n3 ≤ N/3
    - 让进程p1, p2, p3分别模仿n1, n2, n3个将军
    - 若存在一个解决方法，即达成一致且满足完整性条件。与三个进程的不可能性结论矛盾
    - f个坏进程可能不发消息，故需保证N-f个消息可确定结果，但无法区分N-f是否都是好的，故最坏结果是N-f里包括f个坏消息，那么剩余的好消息要多余坏消息，即N-f-f>f，即N>3f

  - N≥3f+1：Lamport于1982给出了解决算法
    - 定理：对于任意m，如果有多于3m的将军和至多m个叛徒，算法OM(m)达到共识。
    - f+1轮，O(N*(f+1))条消息
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Lamport%20BGP.png)

- **Raft**
  - 复制状态机（Replicated State Machine）
    - 复制日志（Replicated log），共识模块确保复制状态机执行的复制日志一致。
    - 多个服务器，从相同的初始状态开始，执行相同的一串命令（复制日志），产生相同的最终状态。
  - 领导人选举——服务器状态
    - 一个 Raft 集群包含若干个服务器节点，满足2f+1
    - 在任何时刻，每一个服务器都处于三个状态之一
      - 领导人（Leader）: handles all client interactions, log replication
      - 跟随者（Follower）: completely passive
      - 候选者（Candidate）: used to elect a new leader
    - 在通常情况下，系统中只有一个领导人并且其他的节点全部都是跟随者
    - 服务器启动时均为跟随者
    - 领导人发送心跳信息(empty AppendEntries RPCs)给跟随者以维护领导人身份
    - 如果超时（100-500ms）没有收到RPCs信息，跟随者认为领导人崩溃，发起新的选举
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%201.png)
  - 领导人选举——任期
    - 时间划分成任期
      - 选举时，要么选出1个领导人，要么失败
      - 所有的操作都是在单一领导人下进行
    - 每个服务器持有当前的任期值
    - 任期的核心价值是识别过期的信息
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%202.png)
  - 领导人选举
    - 开始选举：递增当前任期，改变为候选人状态，为自己投票
    - 向所有其他服务器发送RequestVoteRPC，重试直到发生下列情况之一
      - 收到大多数服务器的投票：成为领导人，向所有其他服务器发送AppendEntries RPC心跳信号
      - 收到来自有效领导人的RPC：返回到追随者状态
      - 没有人赢得选举（选举超时结束）：递增任期，开始新的选举
    - 安全性：每届任期最多允许产生一个领导人
      - 每个服务器在每届任期内只投票一次（投票信息存在磁盘上）
      - 两个不同的候选人不能在同一任期内获得多数票
      - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%203.png)
    - 有效性：某个候选人最终获胜
      - 每个人在[T, 2T]中随机选择选举超时时间
      - 一个服务器通常在其他候选人开始启动选举之前发起选举并获胜
      - 如果T >> 网络RTT，则工作良好 
  - 日志复制
    - 日志结构
      - 日志条目 = < index, term, command >
      - 日志存储在稳定的存储器（磁盘）上；在崩溃后仍能恢复。
      - 如果日志条目已存储在大多数服务器上，则提交该条目。由于日志条目已存储在磁盘上，因此该条目最终会被提交执行。
    - 过程
      - 客户端向领导人发送命令
      - 领导人将命令附加到其日志中
      - 领导人向跟随者发送AppendEntries RPCs，要求其他服务器复制日志条目（命令）
      - 一旦其他服务器复制了新的日志条目（命令）
        - 领导人将命令传递给它的状态机，并将结果发送给客户端
        - 如果跟随者崩溃了，领导人会重试AppendEntries RPCs，直到跟随者最终更新了最新的日志条
        - 目跟随者将命令传递给他们的状态机
      - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%204.png)
    - 如果不同服务器上的日志条目具有相同的索引和任期
      - 存储了相同的命令 < index, term, command >
      - 该条目之前条目都是相同的
    - 如果给定的条目被提交，其前面的所有条目也被提交了
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%205.png)
    - AppendEntries RPCs包含本次提交的日志前一条日志的index和term
    - Follower中对应index的entry的必须与请求一致，否则拒绝请求
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%206.png)
    - 领导人变更
      - 新的领导人不会进行特殊操作，仅仅按照正常流程进行
        - 领导人的日志总是正确的
        - 最终，跟随者的日志与领导人的日志完全一致
        - 旧领导人可能会留下部分复制的条目
      - 多次崩溃会留下许多不相干的日志条目
  - 安全性
    - Raft的安全性属性：一旦某个状态机执行一条日志，所有的状态机，必须以相同的顺序执行相同日志记录的命令。
    - 安全性保证
      - 领导人永远不会改写他日志中的条目
      - 只有领导人日志中的条目才可以提交
      - 所有的日志在应用状态机之前都要被提交
    - 挑选最好的领导人
      - 选择最有可能包含所有提交条目的候选者
        - 在RequestVote RPCs中，候选者包括index+最后一个任期内的日志条目
        - 投票者V拒绝投票，如果它的日志是 "更完整的"
        - 相较大多数候选者，领导人将拥有 “最完整 ”的日志
  - 平衡旧领导人(Neutralizing Old Leaders)
    - 领导人暂时中断联系
      -   其他服务器选出新的领导人
      - 旧领导人重新连接
      - 旧领导人试图提交日志条
    - 任期用来识别旧领导人（和候选者）
      - 每个RPC都包含发送方的任期
      - 发送方的任期 < 接收方，则接收方拒绝RPC（通过发送方处理的ACK...）
      - 接收方的任期 < 发送方，则接收方恢复为跟随者，更新任期，处理RPC
    - 选举更新大多数服务器的任期
      - 被废黜的服务器不能提交新的日志条目
  - 客户端协议
    - 向领导人发送命令：如果客户端不知道领导人，则可以联系任一服务器，服务器将客户端重定向到领导人
    - 领导人只在命令写入日志、提交并由领导人执行（在状态机上执行）后做出响应
    - 如果请求超时（例如，领导人崩溃）：客户端向新的领导人重新发出命令（在可能的重定向之后）
    - 即使在领导人崩溃的情况下，也要确保仅完成一次的语义（exactly-once semantics ）
      - 客户端应该在每个命令中嵌入唯一的请求ID
      - 这个唯一的请求ID包括在日志条目中
      - 在接受请求之前，领导人会检查日志中是否有相同ID的条目
  - 配置变更
    - 查看配置。 {领导，{成员}，设置}
    - 共识必须支持对配置的更改：替换故障机器，改变副本的数量
    - 不能直接从一个配置切换到另一个配置：可能会出现相互冲突的大多数（ conflicting majorities ）
    - Raft中集群切换的过渡配置，称为联合共识（ Joint Consensus ），需要新、旧配置中的大多数服务器来选举、提交
    - 配置变更只是一个日志条目；收到后立即应用（无论提交与否）
    - 一旦联合共识得到提交，就开始复制最终配置的日志条目
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%207.png)
    - 任何配置中的任何服务器都可以作为leader
    - 如果领导人不在Cnew配置中，一旦Cnew配置提交，旧领导人就必须退出
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Raft%208.png)
  - 小结
    - Raft 是一种管理复制日志的一致性算法。它提供了和 Paxos 算法相同的功能和性能，但是它的算法结构和 Paxos 不同，使得 Raft 算法更加容易理解，并且更容易构建实际的系统
    - 为了提升可理解性，Raft 将一致性算法分解成了几个关键模块，例如领导人选举、日志复制和安全。同时它通过实施一个更强的一致性来减少需要考虑的状态数量
    - Raft 算法还包括一个新的机制来允许集群成员的动态改变，它通过多数派来保证算法安全



# 事务和并发控制

## 事务

- 事务
  - 由客户定义的针对服务器对象的一组操作，它们组成一个不可分割的单元，由服务器执行
  - 目标：在多个事务访问对象以及服务器面临故障的情况下，保证所有由服务器管理的对象始终保持一个一致的状态
- 事务故障模型
  - 对持久性存储的写操作可能发生故障
  - 服务器可能偶尔崩溃
  - 消息传递可能有任意长的延迟。消息可能丢失、重复或者损害
  - 事务能够处理进程的崩溃故障和通信的遗漏故障，但是不能处理拜占庭故障
- ACID特性
  - 原子性 Atomicity
  - 一致性 Consistency
  - 隔离性 Isolation
  - 持久性 Durability
- 三种执行情况：成功执行、被客户放弃、被服务器放弃
  - 一旦事务被放弃，服务器必须保证清除所有效果，使该事务的影响对其他事务不可见
  - 事务放弃相关的两个问题
    - 脏数据读取：某个事务读取了另一个未提交事务写入的数据
      - 事务U读取了未提交事务T写入的数据U提交后T被放弃U被提交不可能被取消
    - 过早写入：数据库在放弃事务时，将变量的值恢复到该事务所有write操作的“前映像”
      - 两个事务对同一对象进行write操作如何保证前镜像进行事务恢复的正确性
- 事务放弃时的可恢复性
  - 事务可恢复性策略：推迟事务提交，直到它读取更新结果的其它事务都已提交
  - 连锁放弃：某个事务的放弃可能导致后续更多事务的放弃；防止方法：只允许事务读取已提交事务写入的对象
  - 为了保证使用前映像进行事务恢复时获得正确的结果，write操作必须等到前面修改同一对象的其它事务提交或放弃后才能进行
  - 事务的严格执行：read和write操作都推迟到写同一对象的其它事务提交或放弃后进行
  - 临时版本
    - 目的：事务放弃后，能够清除所有对象的更新
    - 方法：事务的所有操作更新将值存储在自己的临时版本中；事务提交时，临时版本的数据才会用来更新对象
- 并发事务中两个典型问题
  - 更新丢失问题
    - 两个事务都读取一个变量的旧数据并用它来计算新数
  - 不一致检索
    - 更新与检索并发
- 串行等价
  - 串行等价性：如果并发事务交错执行操作的效果等同于按某种次序一次执行一个事务的效果，那么这种交错执行是一种串行等价的交错执行
  - 使用串行等价性作为并发执行的判断标准，可防止更新丢失和不一致检索问题
  - 两个事务串行等价的充要条件是：两个事务中所有的冲突操作都按相同的次序在它们访问的对象上执行
  - 串行等价可作为一个标准用于生产并发控制协议，并发控制协议用于将访问的并发事务串行化
  - 串行等价的判断：不同对象的冲突操作对应的事务执行次序相同
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Serial%20Equivalent.png)
- 冲突操作
  - 冲突操作：如果两个操作的执行效果和他们的执行次序相关，称这两个操作相互冲突（conflict）
  - Read和Write操作的冲突规则
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Read%20Write%20Conflict.png)

- 三种并发控制方法
  - 并发控制：并发控制协议都是基于串行相等的标准，源于用来解决操作冲突的规则。
  - 方法
    - 锁
    - 乐观并发控制
    - 时间戳排序

## 锁

- 互斥锁是一种简单的事务串行化实现机制
  - 事务访问对象前请求加锁
  - 若对象已被其它事务锁住，则请求被挂起，直至对象被解锁
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Lock%20Example.png)

- 两阶段加锁
  - 为了保证两个事务的所有冲突操作必须以相同的次序执行，每个事务的第一阶段是一个“增长”阶段，事务不断地获取新锁；
    在第二个阶段，事务释放它的锁（“收缩阶段”），这称为两阶段加锁(two-phase locking).
  - 所有在事务执行过程中获取的新锁必须在事务提交或放弃后才能释放，称为严格的两阶段加锁
  - 为了保证可恢复性，锁必须在所有被更新的对象写入持久存储后才能释放
  - 目的是防止事务放弃导致的脏数据读取、过早写入等问题
- 并发控制使用的粒度
  - 如果并发控制同时应用到所有对象，服务器中对象的并发访问将会严重受限
  - 如果将所有账户都锁住，任何时候只有一个柜台能处理联机事务
  - 访问必须被串行化的部分对象应尽量少
- 读锁和写锁
  - 目的：提高并发度
  - 支持多个并发事务同时读取某个对象（不同事务对同一对象的read操作不冲突）
  - 允许一个事务写对象（write操作前给对象加写锁）
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/read%20lock%20and%20write%20lock.png)
- 事务的操作冲突规则
  - 如果事务T已经对某个对象进行了读操作，那么并发事务U在事务T提交或放弃前不能写该对象
  - 如果事务T已经对某个对象进行了写操作，那么并发事务U在事务T提交或放弃前不能写或读该对象
- 死锁
  - 两个事务都在等待并且只有对方释放锁后才能继续执行
  - 死锁是一种状态，在该状态下一组事务中的每一个事务都在等待其它事务释放某个锁
  - 预防死锁
    - 每个事务在开始运行时锁住它要访问的所有对象
      -  一个简单的原子操作
      - 不必要的资源访问限制
      - 无法预计将要访问的对象
    - 预定次序加锁
      - 过早加锁
      - 减少并发度
  - 死锁检测
    - 维护等待图
    - 检测等待图中是否存在环路
    - 若存在环路，则选择放弃一个事务
  - 锁超时：解除死锁最常用的方法之一
    - 每个锁都有一个时间期限
    - 超过时间期限的锁成为可剥夺锁
    - 若存在等待可剥夺锁保护的对象，则对象解锁
- 锁机制的缺点
  - 锁的维护开销大：只读事务(查询)，不可能改变数据完整性，通常也需要锁保证数据不被其它事务修改，但锁只在最坏的情况下起作用
  - 会引起死锁：超时和检测对交互程序都不理想
  - 并发度低：为避免连锁放弃，事务结束才释放

## 乐观并发控制

- 乐观策略

  - 基于事实：在大多数应用中，两个客户事务访问同一个对象的可能性很低。
  - 方法
    - 访问对象时不作检查操作
    - 事务提交时检测冲突
    - 若存在冲突，则放弃一些事务

- 事务的三个阶段

  - 工作阶段
    - 每个事务拥有所修改对象的临时版本：放弃时没有副作用，临时值（写操作）对其他事务不可见    
    - 每个事务维护访问对象的两个集合：读集合和写集合    
  - 验证阶段
    - 在收到closeTransaction请求，判断是否与其它事务存在冲突 ：成功，则允许提交；失败，则放弃当前事务或者放弃冲突的事务
  - 更新阶段
    - 只读事务通过验证立即提交
    - 写事务在对象的临时版本记录到持久存储器后提交

- 事务的验证

  - 通过读-写冲突规则确保某个事务的执行对其他重叠事务而言是串行等价的
  - 重叠事务是指该事务启动时还没有提交的任何事务
  - 每个事务在进入验证阶段前被赋予一个事务号
    - 事务号是整数，并按升序分配，定义了事务所处的时间位置
    - 事务按事务号顺序进入验证阶段
    - 事务按事务号提交
  - 对事务Tv的验证测试是基于Ti和Tv之间的操作冲突 
    - 事务Tv对事务Ti而言是可串行化的，符合以下规则
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/check%20rules.png)

- 向后验证

  - 向后：较早的重叠事务(向时间后退的方向进行验证)

  - **检查它的读集是否和其它较早重叠事务的写集是否重叠**

  - 规则1：Ti不能读取Tv写的对象。Ti读时Tv还没有写，自动满足
    规则2： Tv不能读取Ti写的对象，需要验证

  - 算法

    ```c
    startTn:  Tv进入工作阶段时已分配的最大事务号
    finishTn: Tv进入验证阶段时已分配的最大事务号
    
    Boolean valid = true
    For ( int Ti = startTn +1; Ti <= finishTn; Ti ++) {
    	if (read set of Tv intersects write set of Ti)
        valid = false
    }
    ```

  - 验证失败后，冲突解决方法：放弃当前进行验证的事务

  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Back%20Check.png)

- 向前验证

  - 向前：重叠的活动事务（工作阶段的事务）

  - **比较它的写集合和所有重叠的活动事务的读集合**

  - 规则1：Ti不能读取Tv写的对象，需要验证
    规则2：Tv不能读取Ti写的对象，活动事务不会在Tv完成前写，自动满足

  - 算法

    ```c
    设活动事务具有连续的事务标示符active_1~active_n
    
    Boolean valid = true
    for ( int Tid = active_1 ; Tid <= active_n; Tid ++){
      if (write set of Tv intersects read set of Tid) 
        valid = false
    }
    ```

  - 验证失败后，冲突解决方法

    - 放弃当前进行验证事务
    - 推迟验证
    - 放弃所有冲突的活动事务，提交已验证事务

  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Pro%20Check.png)

- 向前验证和向后验证的比较

  - 向前验证在处理冲突时比较灵活
  - 向后验证将较大的读集合和较早事务的写集合进行比较
  - 向前验证将较小的写集合和活动事务的读集合进行比较
  - 向后验证需要存储已提交事务的写集合
  - 向前验证不允许在验证过程中开始新事务

- 饥饿

  - 由于冲突，某个事务被反复放弃，阻止它最终提交的现象
  - 利用信号量，实现资源的互斥访问，避免事务饥饿

## 时间戳排序

- 基本思想
  - 事务中的每个操作在执行前先进行验证
- 时间戳
  - 每个事务在启动时被赋予一个唯一的时间戳
  - 时间戳定义了该事务在事务时间序列中的位置
- 冲突规则（每个对象只有一个版本）
  - 写请求有效：对象的最后一次读访问或写访问由一个较早的事务执行的情况
  - 读请求有效：对象的最后一次写访问由一个较早的事务执行的情况
- 基于时间戳的并发控制
  - 临时版本
    - 写操作记录在对象的临时版本中
    - 临时版本中的写操作对其它事务不可见
  - 写时间戳和读时间戳
    - 已提交对象的写时间戳比所有临时版本都要早
    - 读时间戳集用集合中的最大值来代表
    - 事务的读操作作用于时间戳小于该事务时间戳的最大写时间戳的对象版本上
- 操作冲突
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Time%20Order.png)
- 例子
  - 写操作
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Time%20Order%20Write.png)
  - 读操作
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Time%20Order%20Read.png)
  - 综合
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Time%20Order%20Example.png)

## 并发控制方法比较

- 控制时机

  - 两阶段加锁：事务执行中
  - 乐观并发控制：事务执行后
  - 时间戳排序：事务执行前

- 优缺点

  - 两阶段加锁：只在死锁时放弃，容易死锁

  - 乐观并发控制：执行过程中不进行检测，提交前必须经过验证；不能通过验证的事务不断放弃，可能引发饥饿

  - 时间戳排序：对只读事务有利，到来较晚的事务必须被放弃


# 复制

## 简介

- 复制的概念：在多个计算机中进行数据副本的维护
- 复制的动机
  - 增强性能
    - 浏览器对Web资源的缓存
    - 数据在多个服务器之间地透明复制
  - 提高可用性
    - 服务器故障：1－p^n
    - 网络分区和断链操作：预先复制
  - 增强容错能力
    - 正确性：允许一定数量和类型的故障（如崩溃、拜占庭）
    - 提供给客户的数据是否最新及客户对数据操作的结果
- 复制的基本要求
  - 复制透明性
    - 对客户屏蔽多个物理拷贝的存在
    - 客户仅对一个逻辑对象进行操作
  - 一致性
    - 在不同应用中有不同强度的一致性需求
    - 复制对象集合的操作必须满足应用需求

## 系统模型

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%201.png)

- 基本模型组件
  - 前端
    - 接收客户请求
    - 通过消息传递与多个副本管理器进行通信
  - 副本管理器（副本+操作副本的组件）
    - 接收前端请求
    - 对副本执行原子性操作
    - 副本管理器是一个状态机：当前状态+一组序列操作 => 一个新的确定状态
- 副本对象的操作
  - 请求：前端将请求发送至一个或多个副本管理器
  - 协调：保证执行的一致性
    - 是否执行请求达成一致
    - 确定该请求相对于其他请求的顺序（如FIFO、因果、全序）
      - FIFO：如果前端发送请求r，然后发送请求r’，那么任何正确的处理了r’的副本管理器，在处理r’之前处理r。
      - 因果序：如果请求r在请求r’发送之前发生（happen-before），那么任何正确的处理了r’的副本管理器，在处理r’之前处理r。
      - 全序：如果一个正确的副本管理器在处理请求r’之前处理请求r，那么任何正确的副本管理器在处理r’之前处理r。
  - 执行：副本管理器执行请求，执行效果可去除
  - 协定：就提交请求的执行结果达成一致，可共同决定执行或放弃
  - 响应：一个或多个副本管理器响应前端

## 容错服务

- 在进程出现故障时仍能提供正确的服务
- 复制是提高系统容错能力的有效手段之一
  - 为用户提供一个单一的镜像
  - 副本之间需要保持严格的一致性
- 副本之间的不一致性将导致容错能力失效
- 复制系统正确行为的判断标准
  - 线性化能力
    - 一个被复制的共享对象服务，如果对于任何执行，存在某一个由全体客户操作的交错序列，满足以下两个准则，则该服务被认为是可线性化：操作的交错执行序列符合对象的（单个）副本所遵循的规约；操作的交错执行序列和实际运行中的次序实时一致
    - 目标：所有的客户看到的副本是正确的一致的。
    - 系统应该造成单个副本的错觉；无论哪个客户，读都会返回最近写的结果；无论哪个客户，所有后续读都应返回相同的结果，直到下一次写。
    - 可线性化（Linearizability）≠ 可串行化（ Serializability）
  - 顺序一致性
    - 线性化能力对实时性要求过高，更多的时候要求顺序一致性。
    - 一个被复制的共享对象服务被称为顺序一致性，满足以下两个准则：操作的交错序列符合对象的（单个）正确副本所遵循的规约；操作在交错执行中的次序和每个客户程序中执行的次序一致。
  - 可线性化和顺序一致性比较
    - 可线性化：操作的次序由时间决定，链式复制可以提供可线性化，被动复制（主备份）可以提供可线性化
    - 顺序一致性：每个客户保留程序的次序，主动复制可以提供顺序一致性

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%202.png)

------

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%203.png)

------

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%204.png)

- 链式复制（支持可线性化的技术）
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%205.png)
- 被动(主备份)复制（支持可线性化的技术）
  - 一个主副本管理器＋多个次副本管理器
  - 若主副本管理器出现故障，则某个备份副本管理器将提升为主副本管理器
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%206.png)
  - 被动复制时的事件次序
    - 请求：前端将请求发送给主副本管理器
    - 协调：主副本管理器按接收次序对请求排序
    - 执行：主副本管理器依次执行请求并存储应答
    - 协定
      - 若请求为更新操作，则主副本管理器向每个备份副本管理器发送更新后的状态、应答和唯一标识符
      - 备份副本管理器返回确认
    - 响应
      - 主副本管理器将响应前端
      - 前端将应答发送给客户
- 主动复制（支持顺序一致性的技术）
  - 副本管理器地位对等，前端组播消息至副本管理器组
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%207.png)
  - 主动复制时的事件次序
    - 请求：前端使用全序、可靠的组播原语将请求组播到副本管理器组
    - 协调：组通信系统以同样的次序(全序)将请求传递到每个副本管理器
    - 执行：每个副本管理器以相同的方式执行请求
    - 协定：鉴于组播的传递语义，不需要该阶段
    - 响应
      - 每个副本管理器将应答发送给前端，接收的应答数量取决于故障模型的假设和组播算法
      - 前端将应答发送给客户

## 复制数据上的事务

- 对客户而言，有复制对象的事务看上去应该和没有复制对象的事务一样
- 作用于复制对象的事务应该和它们在一个对象集上的一次执行具有一样的效果，这种性质叫做单拷贝串行化
- 单拷贝串行化可以通过“读一个/写所有”实现复制方案

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%208.png)

- 复制方案
  - 服务器崩溃—本地验证
    - 客户对一个逻辑对象的读请求可由任何可用的副本管理器执行，而写请求必须由具有该对象的所有可用副本管理器执行
    - 只要可用的副本管理器集没有变化，本地的并发控制和读一个/写所有复制一样可获得单拷贝串行化
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%209.png)
    - X在事务T执行getBalance后出现故障，而N在事务U执行getBalance后出现故障，因此，副本管理器X上对A的并发控制并不会阻止事务U在副本管理器Y上更新A，同理副本管理器N上对B的并发控制也不会阻止T在副本管理器M和P上的更新B
    - 该现象与单拷贝串行化相违背，因此需要额外的并发控制
    - 假设T在X、M、P上成功提交，U在Y、M、P、N上成功提交，且事务执行中未发生故障，则存在以下矛盾：
      - N出故障->T在X上读对象A，T在M和P上写对象B->T提交->X出故障
      - X出故障->U在N上读对象B，U在Y上写对象A ->U提交->N出故障
    - 本地验证用来确保任何故障或恢复事件不会在事务的执行过程中发生
    - 添加本地验证后，即可保证单拷贝串行化
    - 本地验证实现：检查事务开始时、事务提交时副本管理器集合是否发生了变化
  - 网络分区—乐观方法、悲观方法
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%2010.png)
    - 复制方案假设：网络分区终将修复，单个分区中的副本管理器必须保证在分区期间它们执行的任何请求在分区修复后不会造成不一致
    - 乐观方法：允许在所有的分区中进行操作，造成的不一致在分区修复后解决。如对更新进行验证，丢弃任何违背单拷贝串行化准则的更新
    - 悲观方法：阻止分区中可能导致不一致的操作。如操作仅在一个分区进行，分区修复后更新其余分区

## 高可用服务的实例研究

- gossip协议
  - gossip算法又被称为反熵（Anti-Entropy），熵是物理学上的一个概念，代表杂乱无章，而反熵就是在杂乱无章中寻求一致
  - gossip的特点：在一个有界网络中，每个节点都随机地与其他节点通信，经过一番杂乱无章的通信，最终所有节点的状态都会达成一致。每个节点最初可能知道所有其他节点，也可能仅知道几个邻居节点，只要这些节点可以通过网络连通，最终他们的状态都是一致的
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%2011.png)
- 容错能力和高可用性
  - 容错能力
    - 只要可能，所有正确的副本管理器都能及时收到更新，并在将控制传递回客户以前达成一致
  - 高可用性
    - 采用较弱程度的一致性，提高共享数据的可用性
    - 实例：闲聊体系架构、Bayou和Coda
- 闲聊体系结构
  - 体系结构
    - 前端可以选择任意副本管理器
    - 提供两种基本操作：查询＋更新
    - 副本管理器定期通过gossip消息来传递客户的更新
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%2012.png)
  - 系统的两个保证
    - 随着时间的推移，每个用户总能获得一致服务
      - 副本管理器提供的数据能反映迄今为止客户已经观测到的更新
    - 副本之间松弛的一致性
      - 所有副本管理器最终将收到所有更新
      - 两个客户可能会观察到不同的副本
      - 客户可能观察到过时数据
  - 查询和更新操作流程
    - 请求：前端将请求发送至副本管理器
      - 查询：客户可能阻塞
      - 更新：无阻塞
    - 更新响应：副本管理器立即应答收到的更新请求
    - 协调：收到请求的副本管理器并不处理操作，直到它能根据所要求的次序约束处理请求为止
    - 执行： 副本管理器执行请求查询
    - 响应：副本管理器对查询请求立即作出应答
    - 协定：副本管理器通过交换gossip消息进行相互更新
      - gossip消息的交换是偶尔的
      - 发现消息丢失后，才和特定的副本管理器交换消息
  - 前端的版本时间戳
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%2013.png)
    - 为了控制操作处理次序，每个前端维持了一个向量时间戳，用来反映前端访问的最新数据值的版本
    - 客户通过访问相同的gossip服务　 和相互直接通信来交互数据
    - 每个前端维护一个向量时间戳
      - 每个副本管理器有一条对应的记录
      - 更新或查询信息中包含时间戳
      - 合并操作返回的时间戳与前端时间戳
    - 向量时间戳的作用
      - 反映前端访问的最新数据值
  - 副本管理器状态
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%2014.png)
    - 值：副本管理器维护的应用状态的值
    - 值的时间戳：更新的向量时间戳
    - 更新日志：记录更新操作
    - 副本的时间戳：已经被副本服务器接收到的更新的时间戳
    - 已执行操作表：记录已经执行的更新的唯一标识符，防止重复执行
    - 时间戳表：确定何时一个更新已经应用于所有的副本管理器
  - 查询操作
    - 副本管理器收到查询
      - q.pre ≤ valueTS
        - 立即响应
        - 返回消息中的时间戳为valueTS
      - 否则将消息保存到保留队列
      - 如：q.pre(2,4,6),valueTS(2,5,5)
    - 前端收到查询响应
      - 合并时间戳：frontEndTS:=merge(frontEndTS,new)
  - 按因果次序处理更新
    - 前端发送更新请求：(u.op, u.prev, u.id)
    - 副本管理器i接收请求
      - 丢弃：操作已经处理过
      - 否则，将更新记录日志
        - ts =u.prev, ts[i]=ts[i]+1
        - logRecord= <i, ts, u.op, u.prev, u.id>
      - 副本管理器将ts返回给前端
        - frontEndTS=merge(frontEndTS, ts)
      - 更新请求u的稳定性条件　　
        - u.prev≤valueTS
      - 副本管理器的更新操作
        - value := apply(value, r.u.op)
        - valueTS := merge(valueTS, r.ts)
        - executed := executed∪{r.u.id}
  - 强制的和即时的更新操作	      
    - 强制更新和即时更新需要特殊处理，强制更新是全序加因果序，保证更新的强制次序的基本方法是在与更新相关的时间戳后加入一个唯一的序号，并以这个序号的次序来处理它们
  - gossip消息            
    - 副本管理器通过gossip消息来更新自身的状态。通过时间戳表里的记录来估计其它副本管理器还没有收到哪些更新
  - gossip消息m的格式      
    - 日志m.log和副本时间戳m.ts
  - 收到gossip消息后执行的操作
    - 日志合并　 
      - 若r.ts ≤replicsTS，则丢弃
      - 将记录加入到日志，合并时间戳　　
        - replicaTS := merge(replicaTS, m.ts)
    - 执行任何以前没有执行并已经稳定了的更新　  
      - 根据向量时间戳的偏序“≤”对更新进行排序，并依次执行更新
    - 当知道更新已执行并没有重复执行的危险时，删除日志和已执行操作表中的记录　  
      - 若tableTS[i][c] ≥ r.ts[c]，则丢弃r
  - 更新传播
    - gossip体系结构未规定具体的更新传播策略
    - 如何选择合适的gossip消息的发送频率？　  
      - 分钟、小时或天？——由具体应用需求决定
    - 如何选择合适的合作者(副本管理器)
      - 随机策略：使用加权概率来选择更合适的合作者
      - 确定策略：使用副本管理器状态的函数来选择合作者
      - 拓扑策略：网格、环、树
  - 目标：保证服务的高可用性
  - 存在问题：不适合接近实时的更新复制、可扩展性问题
- Bayou系统和操作变换方法
  - Bayou系统简介
    - 与gossisp体系结构和基于时间戳的反熵协议类似
    - 提供的一致性保证弱于顺序一致性
    - 能够进行冲突检测和冲突解决
      - 操作变换：一个或多个相冲突的操作被取消或改变以解决冲突的过程
      - 例子：行政主管和秘书同时预约 ，其中行政主管为离线更新——行政主管上线后，Bayou系统检测到冲突，然后批准行政主管的预约而取消秘书的预约
  - 提交的更新和临时更新
    - 临时的更新：更新首次应用于数据库时，被标记为临时的
    - 提交的更新：Bayou将临时的更新以规范次序放置，并添加提交标识
    - 数据库副本状态：提交的更新序列＋临时的更新序列
    - 更新重排序：新更新到达或某个临时更新被修改为提交的更新
    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/Copy%2015.png)
    - 临时更新ti成为下一个提交更新，并被插入到最新提交更新cN之后
  - 依赖检查和合并过程
    - 依赖检查
      - 一个更新执行时是否会产生冲突
      - 例子：写－写冲突、读－写冲突检测
    - 合并过程
      - 改变将要执行的操作，避免冲突，并获得相似效果
      - 无法合并→系统报错
  - Bayou系统讨论
    - 复制对于应用而言是不透明的
    - 复杂度高
- Coda文件系统
  - AFS的主要缺陷：只提供有限复制，规模受限，不适用于大规模共享的文件访问
  - Coda目标：提供一个共享的文件存储
  - Coda对AFS的扩展
    - 采用文件卷复制技术——提高吞吐率和容错性
    - 在客户计算机上缓存文件副本——断链处理
  - Coda体系结构
    - Venus/Vice进程
      - Venus：前端和副本管理器的混合体
      - Vice：副本管理器
    - 卷存储组(VSG) ：持有一个文件卷副本的服务器集合
    - 可用的卷存储组(AVSG)：打开一个文件的客户能访问的VSG的某个子集
    - 断链操作
      - AVSG为空时，客户缓存文件
      - 手工干预解决冲突
    - 设计原则：服务器上的拷贝比客户计算机缓存中的拷贝更可靠
  - 复制策略
    - 乐观策略：在网络分区和断链操作期间，仍然可以进行文件修改
    - 实现
      - Coda版本向量(CVV, Code Version Vector )
      - 作为时间戳附加在每个版本的文件上
      - CVV中的每个元素是一个估计值，表示服务器上文件修改次数的估计
      - 目的提供足够的关于每个文件副本的更新历史，以检测出潜在的冲突，进行手工干预和自动更新
    - 例如：CVV=（2，2，1）
      - 文件在服务器1上收到2个更新
      - 文件在服务器2上收到2个更新
      - 文件在服务器3上收到1个更新
    - 冲突检测
      - 若一个站点的CVV大于或等于所有其它站点相应的CVV，则不存在冲突。→自动更新
      - 若对于两个CVV而言，v1≥v2与v2≥v1均不成立，则存在一个冲突。 →手工干预
    - 文件关闭
      - Venus进程发送更新消息（包括CVV和文件的新内容）到AVSG
      - AVSG中的每个服务器更新文件，并返回确认
      - Venus计算新的CVV，增加相应服务器的修改记数，并分发新的CVV



# 分布式文件系统

## 简介

- 本地文件系统
  - 文件系统提供文件的管理
    - 命名空间
    - 文件操作的API：create, delete, open, close, read, write, append …
    - 物理空间的存储管理：块分配、回收等
    - 安全保护：访问控制
  - 层次化的命名空间：文件和目录
  - 文件系统已被安装：不同的文件系统可以在同一个命名空间中
- 传统分布式文件系统
  - 目的：模拟本地文件系统的行为
    - 文件没有被复制
    - 没有严格的性能保证
  - 但是
    - 文件位于远程的服务器上
    - 多个远程客户可以访问服务器
  - 为什么？
    - 用户有多台计算机
    - 数据被多个用户共享
    - 统一的数据管理（企业）
- 分布式文件系统的需求
  - 透明性：分布式文件系统应该如同本地文件系统
    - 访问透明性：客户无需了解文件的分布性，通过一组文件操作访问本地/远程文件，需要支持相同的一组文件操作，程序要如同本地文件系统一样在分布式文件系统上工作
    - 位置透明性：客户使用单一的文件命名空间。所有用户看到同样的命名空间
    - 移动透明性：当文件移动时，客户的系统管理表不必修改。多个文件或文件卷可以被系统管理员自由移动。
    - 性能透明性：负载在一个特定范围内变化时，性能可接受，系统提供合理的一致性能
    - 伸缩透明性：文件服务可扩充，以满足负载和网络规模增长的需要，系统可以通过增加服务器逐步扩展
  - 并发的文件更新：并发控制，客户改变文件操作不影响其他客户
  - 文件复制：分担文件服务负载，更好的性能和系统容错能力
  - 硬件和操作系统的异构性：接口定义明确，在不同操作系统和计算机上实现同样服务
  - 容错
    - 为了处理瞬时通信故障，设计可以基于最多一次的调用语义
    - 幂等操作：支持最少一次语义，重复执行的效果与执行一次的相同
    - 无状态服务器：在崩溃后无恢复重启
  - 一致性：单个拷贝更新语义，多个进程并发访问或修改文件时，它们只看到仅有一个文件拷贝存在
  - 安全性：客户请求需要认证，访问控制基于正确的用户身份
  - 效率：至少和传统文件系统相同的能力，并满足一定的性能需求

## 文件服务体系结构

- 本地文件系统布局
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%201.png)
- Unix inode
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%202.png)
- UNIX中的路径解析——/programs/pong.c
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%203.png)
- 文件服务的三个组件
  - 客户模块
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%204.png)
    - 运行在客户计算机上
    - 提供应用程序对远程文件服务透明存取的支持
    - 缓存最近使用的文件块提高性能
  - 目录服务
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%205.png)
    - 提供文件名到UFID的映射
  - 平面文件服务
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%206.png)
    - 基于UFID，对文件内容进行操作
    - read、write: 最重要的文件操作，均需要一个参数i来指定文件的读写位置
- 平面文件服务接口讨论
  - 平面文件服务操作和UNIX进行比较
    - 无open和close操作，通过引用合适的UFID可以立即访问文件
    - read和write操作需要一个开始位置，UNIX的read、write操作中无此参数
  - 与UNIX文件系统相比在容错方面的影响
    - 可重复性操作：除了create，其它所有的操作都是幂等的
    - 无状态服务器：在文件上进行操作不需要读-写指针，故障后重启无需客户或服务器恢复任何状态
- 访问控制
  - UNIX文件系统：用户进行open调用时，系统核对其访问权限
  - 无状态的DFS
    - DFS接口对大众公开，存在安全隐患
    - 文件服务器不能保留用户标识（UID），否则服务就变成有状态了
    - 实施访问控制的两个方法
      - 基于权能的认证（访问能力列表，每个主体都附加一个该主体可访问的客体的明细表）
      - 将每个请求与UID关联起来，每次请求都进行访问检查
    - NFS (Networked File System)和AFS (Andrew File System)使用Kerberos认证解决伪造用户标识的安全问题
      ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%207.png)
- 层次文件系统
  - 目录树
    - 目录是一种特殊的文件：包含通过它访问的文件名和目录名
    - 路径名：表示一个文件或者是目录，多部分命名，如“/etc/rc.d/init.d/nfsd”
  - 遍历目录
    - 通过多次查找操作来解析路径名
    - 客户端缓存目录

## SUN网络文件系统

- NFS体系结构
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%208.png)

- 虚拟文件系统（VFS）

  - 在UNIX内核的一个转换层
    - 支持挂载不同文件系统（EXT2、NTFS、NFS、……）
    - 不同文件系统可共存
    - 区分本地和远程文件
  - 跟踪本地和远程可用的文件系统
  - 将请求传递到适当的本地或远程文件系统 
  - V节点
    - 本地文件：引用一个i节点
    - 远程文件：引用一个文件句柄
  - 文件句柄
    - 文件系统标识符：服务器可能服务多个文件系统
    - i节点数
      - 在一个特定的文件系统内是唯一的
      - i节点数是用于标识和定位文件的数值
    - i节点产生数
      - i节点在文件被删除后可被重用，每次重用时加1
      - 当i节点被回收时（i节点产生数变化），但文件仍然打开，抛出异常
    - 客户与服务器之间传递文件句柄，句柄对客户不透明

- 设计要点

  - 访问控制及认证
    - 与传统UNIX文件系统不同，NFS服务器是无状态的
    - 在用户发出每一个新的文件请求时，服务器必须重新对比用户ID和文件访问许可属性，判断是否允许用户进行访问
      - 将用户ID绑定到每个请求
      - 在NFS中嵌入Kerberos认证：在加载的同时认证客户；凭据，认证和安全通道
    - 将客户端集成到内核
      - 用户程序可以通过UNIX系统调用访问文件，不需要重新编译或者加载库
      - 一个客户模块通过使用一个共享缓存存储最近使用的文件块，为所有的用户级进程服务
      - 传输给服务器用于认证用户ID的密钥可以由内核保存，防止用户级客户冒用客户
    - NFS服务器接口：RFC 1813 中定义

- 安装服务

  - 文件服务器
    - 每一个服务器上都一个具有已知名字的文件（/etc/exports）——包含了本地文件系统中可被远程加载的文件名
  - 客户
    - 当客户想访问远程文件时，客户使用RPC使文件可用
    - 包含位置，远程目录的路径名

  - 远程安装
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%209.png)
    - 每台服务器都会记录可用于远程安装的本地文件
    - 客户使用 mount 命令进行远程安装，提供名称映射
    - 服务器1和2的people和users被安装到客户本地文件students和staff上

- 在NFS客户端可访问的本地和远程文件系统
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%2010.png)

  - 安装在客户 /usr/students上的文件系统实际上是位于服务器1上的/export/people下的一个子树，例如，用户可以使用/usr/students/john访问服务器1上的文件
  - 安装在客户 /usr/stuff上的文件系统实际上是位于服务器2上的/nfs/users下的一个子树，例如，用户可以使用/usr/staff/ann访问服务器2上的文件

- 路径名翻译

  - UNIX文件系统每次使用open 、create或stat系统调用时，一步步地将多部分文件路径名转为i节点引用
  - NFS服务器不进行路径名转换，需要由客户以交互方式完成路径名的翻译
  - 客户向远程服务器提交数个lookup请求，将指向远程安装目录的名字的每一部分转换为文件句柄

- 服务器缓存

  - UNIX文件系统的高速缓存
    - 预先读：将最近常用的页面装入内存
    - 延迟写
      - 该缓冲区将被其他页占用时才将该页的内容写入磁盘
      - 周期性同步写，如每隔30秒将改变的页面写到磁盘中（防止数据丢失）
  - NFS服务器的读缓存：和本地文件系统相同
  - NFS3服务器的写缓存，写操作提供两种选项
    - 写透：在给客户发送应答前先将应答写入磁盘（写操作持久性），写透操作可能引起性能的瓶颈问题
    - 内存缓存：写操作的数据存储在内存缓存中，当系统接收相关文件的commit操作时（用于写而打开的文件关闭时） ，数据再写入磁盘（提高性能）。

- 客户缓存

  - 了减少传输给服务器的请求数量，NFS客户模块将read, write, getattr, looup和readdir操作的结果缓存起来
  - 保持一致性：客户轮询服务器来检查他们所用的缓存数据是否是最新的。（读/写时，只发送查询信息）
  - 基于时间戳的缓存块验证（验证过程不能保证提供和传统UNIX系统一样的一致性）
    - 缓存中的每个数据块被标上两个时间戳
      - Tc：缓存条目上一次被验证的时间
      - Tm：服务器上一次修改文件块的时间
    - 有效性条件： (T- Tc < t) 或者(Tm_client = Tm_server)
      - 若T(当前时间)- Tc < t为真，无须进一步判断
      - 若为假，则需要从服务器获得获得Tm_server值（对服务器应用getattr操作）并比较Tm_serve与Tm_client
    - t: 更新时间间隔，选择t是对一致性和效率进行折衷，如文件3~30秒，目录30～60秒，目录更新风险更低
  - 减少对服务器进行getattr操作的几种方法
    - 当客户收到一个新的Tmserver值时，将该值应用于所有相关文件派生的缓存项
    - 将每一个文件操作的结果同当前文件属性一起发送，如果Tmserver值改变，客户便可用它来更新缓存中与文件相关的条目
    - 采用自适应算法来设置更新间隔值t，对于大多数文件而言，可以极大地减少调用数量

## Andrew文件系统

- 使用AFS的典型场景	
  - 客户打开一个远程文件：这个文件不在本地缓存时，AFS查找文件所在服务器，并请求传输此文件一个副本
  - 在客户机上存储文件副本
  - 客户在本地副本上进行读/写
  - 客户关闭文件：如果文件被更新，将它刷新至服务器，客户本地磁盘上的拷贝一直被保留，以供同一工作站的用户级进程下一次使用
- AFS设计时的考虑
  - 大多数文件，更新频率小，始终被同一用户存取
  - 本地缓存的磁盘空间大，例如：100MB
  - 设计策略基于以下假设
    - 文件比较小，大多数文件小于10KB
    - 读操作是写操作的6倍
    - 通常都是顺序存取，随机存取比较少见
    - 大多数文件是被某一个特定的用户访问，共享文件通常是被某一个特定的用户修改
    - 最近使用的文件很可能再次被使用
  - 不支持数据库文件
- AFS中的进程
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%2011.png)
  - AFS由两个软件组件实现，分别以UNIX进程Venus和Vice存在
  - Venus是运行在客户计算机上的用户进程，相当于抽象模型中的客户模块
    - 通过fid进行存取，类似NFS的UFID(卷号、文件句柄和唯一标识)
    - 一步一步地进行查找：把路径名翻译成fid
    - 文件缓存：一个文件分区用作文件缓，通常可以容纳百个一般大小的文件
    - 维护缓存一致性：回调机制
  - Vice是服务器软件的名字，是运行在每个服务器计算机上的用户级UNIX进程
    - 接收用fid表示的文件请求
    - 平面文件服务
- AFS中的文件
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%2012.png)
  - AFS中的文件分为本地的或共享的
  - 本地文件可作为普通的UNIX文件来处理，它们被存储在工作站磁盘上，只有本地用户可以访问它
  - 共享文件存储在服务器上，工作站在本地磁盘上缓存它们的拷贝
  - 本地文件仅作为临时文件（/tmp），其他标准UNIX文件（/bin）通过将本地文件目录中的文件符号链接（类似Windows快捷方式）到共享文件空间方式实现
- AFS中系统调用拦截
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Distributed%20Systems/DFS%2013.png)
  - UNIX修改版本内核截获那些指向共享名字空间文件的调用，如open、close和其它一些系统调用，并将它们传递给Venus进程
  - 每个工作站本地磁盘上都有一个文件分区被用作文件的缓存。Venus进程管理这一缓存。当文件分区已满，并且有新的文件需要从服务器拷贝过来时，它将最近最少使用的文件从缓存中删除
- 缓存一致性
  - 回调承诺
    - 由管理该文件的Vice服务器发送的一种标识，用于保证当其他客户修改此文件时通知Venus进程
    - 两种状态：有效或取消
    - 当Vice服务器执行一个更新文件请求时，它通知所有Venus进程将回调承诺标识设为取消状态
  - 打开文件
    - 若无文件或是文件的标识值为取消，Venus从服务器取得文件；若标识值为有效，Venus不需要引用Vice，直接使用缓存的文件拷贝
  - 关闭文件
    - 当应用程序更新文件时Venus刷新文件
    - Vice顺序执行对文件的更新命令
    - Vice通知所有的文件缓存设为取消状态
  - 当客户重启或者在时间T内没有收到回调信息，Venus将认为该文件已经无效
  - 可扩展性：由于大部分请求为读请求，与轮询相比，客户与服务器间的交互显著减少，提高了扩展性
- 更新语义
  - 缓存一致性目标：在不对性能产生严重影响的情况下，近似实现单个文件拷贝语义
  - AFS-1(F—File, S—Server)
    - 在成功的open操作后：latest（F，S）：文件F在客户C的当前值和在服务器S上的值相同
    - 成功的close操作后：updated（F，S）：客户C的文件F的值已经传播到服务器S上
    - 在失败的open，close操作后：failure（S）：open和close并没有在S上执行
  - AFS-2：较弱的open保证，客户可能会打开一个旧拷贝，而该文件已经被其他客户更新过了
    - 在成功的open操作后
      - Latest(F,S,0)：文件F在客户C的当前值和服务器S上的值相同（F的拷贝是最新版本）
      - 或者lostCallback(S,T) and inCache(F) and latest(F,S,T)。回调丢失（通信故障）不得不使用已缓存文件版本
        - lostCallback(S,Ts)：最近Ts时间内从服务器S传递到客户C的回调信息已经丢失
        - inCache(F)：在open操作前客户C的缓存中就包含文件F
        - latest(F,S,T)：被缓存的文件F的拷贝过期时间不会超过T秒（T通常设置为10分钟）
- 其他方面
  - 内核修改：修改UNIX内核，以支持Vice中使用文件句柄而不是UNIX文件描述符执行文件操作
  - 线程：Vice和Venus中使用非预先抢占性线程包，使客户和服务器并发处理请求
  - 只读复制：经常执行读操作，但很少修改的文件卷拷贝到多个服务器上
  - 批量传输：AFS以64KB的文件块进行传输以减小延迟
  - 部分文件缓存：当应用程序只需要读文件的一小部分时仍需将整个文件传输到客户端，显然效率低。AFSv3解决这个问题，允许文件数据以64KB块的形式传输和缓存
  - 位置数据库：每个服务器包含一个位置数据库的拷贝，用于将卷名映射到服务器
  - 性能：AFS主要目标是可扩展性。通过缓存整个文件和回调机制减少服务器的负载
  - 广域网支持：AFSv3支持多个管理单元，每个单元有自己的服务器、客户、系统管理员和用户，是一个完全自治的环境，但这些协作的单元可以共同为用户提供一个统一的、无缝的文件名空间

## DFS进展

- NFS的改进
  - Spritely NFS达到单个拷贝的更新语义
    - 多个客户对缓存副本进行并发读
    - 一个写操作，以及多个读操作在服务器相同的副本
  - NQNFS：更精确的一致性，通过租借来保证缓存一致性
  - WebNFS：通过Web直接访问NFS服务器
  - NFS第4版：使NFS适用于广域网和互联网应用

- 存储组织的改进
  - 廉价磁盘的冗余阵列（RAID）
    - 数据分解成固定大小的块
    - 存储在跨域多个磁盘的“条带”上
    - 冗余的错误更正代码
  - 日志结构的文件存储（LFS）
    - 内存积累若干写操作
    - 写到划分为大的、连续的、定长的段的磁盘上
- 新的设计方法
  - 以高伸缩性和高容错性的方式提供分布式文件数据的持久性存储系统
  - 把管理元数据和客户请求服务的职责与读写数据的职责相分离
  - offs（无服务文件系统）
    - 存储责任独立于管理和其它服务责任进行分布
    - 将文件数据分散存储到多个计算机上
    - 软件的RAID存储系统
    - 协同缓存
  - Frangipani
    - 将持久存储责任和其他文件服务活动相分离
    - Petal：为多个服务器磁盘提供了一个分布式的磁盘抽象
    - 日志结构的数据存储











