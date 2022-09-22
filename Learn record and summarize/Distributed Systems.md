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
