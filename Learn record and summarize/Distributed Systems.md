# Introduction

- **Why do we need Distributed System?**

  Functional Separation（功能分离）、Inherent distribution（固有的分布性）、Power imbalance and load variation（负载均衡）、Reliability（可靠性）、Economies（经济性）

- **goal：**资源共享 (resource sharing)、协同计算 (collaborative computing)

- **definition：**A distributed system is one in which components located at networked computers communicate and coordinate their actions only by passing messages.

- **fundamental feature：**并发性 (concurrency)、无全局时钟 (non-global clock)、故障独立性 (independent failure)

- **challenge：**异构性（Heterogeneity）、开放性（Openness）、安全性（Security）、可伸缩性（Scalability）、故障处理（Failure handling）、并发行（Concurrency）、透明性（Transparency）

- **RPC：**Remote Procedure Call



# 系统模型System Model

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



# 时间和全局状态Time Global State

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

网络时间协议(Network Time Protocol，NTP)：可外部同步：使得跨Internet的用户能精确地与UTC (通用协调时间)同步；高可靠性：可处理连接丢失，采用冗余服务器、路径等；扩展性好：大量用户可经常同步，以抵消漂移率的影响；安全性强：防止恶意或偶然的干扰

协议结构：层次结构 —— strata；主服务器直接与外部UTC同步；同步子网可重新配置

NTP服务器同步模式：1⃣️组播模式 multicast/broadcast mode，适用于高速LAN，准确度较低，但效率高。2⃣️服务器/客户端模式 server/client mode，与Cristian算法类似，准确度高于组播。3⃣️对称模式 symmetric mode，保留时序信息，准确度最高。

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

  ​	引入进程标示符创建事件的全序关系：1⃣️若e、e′分别为进程pi、pj中发生的事件，则其全局逻辑时间戳分别为(Ti, i)、(Tj, j)。2⃣️e→e′↔️Ti<Tj || (Ti=Tj && i<j)，该排序没有实际物理意义。


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



# 协调和协定Coordination Agreement

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

- **Paxos (Παξος): CFT Consensus**

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

    - Proposal：An alternative proposed by a proposer. Consists of a unique number and a proposed value (42, B)

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


