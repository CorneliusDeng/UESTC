## Week 1_2022.8.29~2022.9.4

- ### 课程

  - #### 高级算法设计与分析

    - **Kinds**

      Decision Problem (with yes-no answers)

      Optimal Value/Optimal Solution

      Numerical Calculation

      - **Programs and algorithms**

        A computer program is an instance, or concrete representation, for an algorithm in some programming language.
        • A program is to be read by computer
        • An algorithm is to be read by human being
        • Algorithms can be expressed by pseudocode just some short steps that are easy to read and understand

      - **Different Classes of Problems**

        P: a solution can be solved in polynomial time.

        NP: a solution can be checked in polynomial time.

        NPC: problems that may not have polynomial-time algorithms.

      - **The Stable Matching Problem**

        **Goal:**  Given n men and n women, find a "suitable" matching. Participants rate members of opposite sex. Each man lists women in order of preference from best to worst. Each woman lists men in order of preference from best to worst

        **Perfect matching:** everyone is matched monogamously. Each man gets exactly one woman. Each woman gets exactly one man.
        **Stability:** no incentive for some pair of participants to undermine assignment by joint action. In matching M, an unmatched pair m-w is unstable if man m and woman w prefer each other to current partners. Unstable pair m-w could each improve by eloping.
        **Stable matching:** perfect matching with no unstable pairs.
        **Stable matching problem:**  Given the preference lists of n men and n women, find a stable matching if one exists.

        **Propose-and-reject algorithm.** [Gale-Shapley 1962] Intuitive method that guarantees to find a stable matching.

        `Initialize each person to be free.`
        `while (some man is free and hasn't proposed to every woman) {`
        				` Choose such a man m`
         				`w = 1st woman on m's list to whom m has not yet proposed`
        				  `if (w is free)`
        							`assign m and w to be engaged`
        				  `else if (w prefers m to her fiancé m')`
        							`assign m and w to be engaged, and m' to be free`
        				 `else`
        						   `w rejects m`
        `}`

  - #### 分布式系统

    - **Why do we need Distributed System?**

      Functional Separation（功能分离）、Inherent distribution（固有的分布性）、Power imbalance and load variation（负载均衡）、Reliability（可靠性）、Economies（经济性）

    - **goal：**资源共享 (resource sharing)、协同计算 (collaborative computing)
    - **definition：**A distributed system is one in which components located at networked computers communicate and coordinate their actions only by passing messages.
    - **fundamental feature：**并发性 (concurrency)、无全局时钟 (non-global clock)、故障独立性 (independent failure)
    - **challenge：**异构性（Heterogeneity）、开放性（Openness）、安全性（Security）、可伸缩性（Scalability）、故障处理（Failure handling）、并发行（Concurrency）、透明性（Transparency）
    - **RPC：**Remote Procedure Call

  - #### English for Academic Communication and Presentation

    - **Five Authorities in Artificial Intelligence：**
    - **Five Top Journals in Artificial Intelligence：**
    - **Five Top Conferences in Artificial Intelligence：**



- ### 自研

  - #### 微服务概念

    微服务是一种架构，这种架构是将单个的整体应用程序分割成更小的项目关联的独立的服务。一个服务通常实现一组独立的特性或功能，包含自己的业务逻辑和适配器。各个微服务之间的关联通过暴露api来实现。这些独立的微服务不需要部署在同一个虚拟机，同一个系统和同一个应用服务器中。微服务架构风格是一种将单个应用程序开发为“一套小型服务”的方法，每个服务“运行在自己的进程中”，并通过轻量级机制(通常是HTTP资源API)进行通信。

    ##### 微服务架构常见概念

    - **服务治理：服务治理就是进行服务的自动化管理，其核心是服务的注册与发现**

      服务注册：服务实例将自身服务信息注册到注册中心。
      服务发现：服务实例通过注册中心，获取到注册到其中的服务实例的信息，通过这些信息去请求他们提供服务。
      服务剔除：服务注册中心将出问题的服务自动剔除到可用列表之外，使其不会被调用到。

    - **服务调用**

      在微服务架构中，通常存在多个服务之间的远程调用的需求，目前1主流的远程调用的技术有基于HTTP请求的RESTFul接口及基于TCP的RPC协议

    - **服务网关**

      随着微服务的不断增多，不同的微服务一般会有不同的网络地址，而外部客户端可能需要调用多个服务的接口才能完成一个业务需求，API网关直面意思是将所有API调用统一接入到API网关层，由网关层统一接入和输出。一个网关的基本功能有：统一接入、安全防护、协议适配、流量管控、长短链接支持、容错能力。有了网关之后，各个API服务提供团队可以专注于自己的的业务逻辑处理，而API网关更专注于安全、流量、路由等问题。

    - **服务容错**

      在微服务当中，一个请求经常会涉及到调用几个服务，如果其中某个服务不可用，没有做服务容错的话，极有可能会造成一连串的服务不可用，这就是雪崩效应。我们没法预防雪崩效应的发生，只能尽可能去做好容错。服务容错的三个核心思想是：不被外界环境影响、不被上游请求压垮、不被下游响应拖垮。

    - **链路追踪**

      随着微服务架构的流行，服务按照不同的维度进行拆分，一次请求往往需要涉及到多个服务。互联网应用构建在不同的软件模块集上，这些软件模块，有可能是由不同的团队开发、可能使用不同的编程语言来实现、有可能布在了几千台服务器，横跨多个不同的数据中心。因此，就需要对一次请求涉及的多个服务链路进行日志记录，性能监控即链路追踪。

  - #### 负载均衡概念

    - 当一台服务器的性能达到极限时，我们可以使用服务器集群来提高网站的整体性能。那么，在服务器集群中，需要有一台服务器充当调度者的角色，用户的所有请求都会首先由它接收，调度者再根据每台服务器的负载情况将请求分配给某一台后端服务器去处理。那么在这个过程中，调度者如何合理分配任务，保证所有后端服务器都将性能充分发挥，从而保持服务器集群的整体性能最优，这就是负载均衡问题。

  - #### 了解Redis和Docker的概念与简单操作

    - **Redis：Remote Dictionary Server，远程字典服务**

      redis会周期性的把更新的数据写入磁盘或者把修改操作写入追加的记录文件，并且在此基础上实现了master-slave(主从)同步，被人们称之为结构化数据库

    - **Docker：以容器虚拟化技术为基础**

      我们用的传统虚拟机需要模拟整台机器包括硬件，每台虚拟机都需要有自己的操作系统，虚拟机一旦被开启，预分配给它的资源将全部被占用。每一台虚拟机包括应用，必要的二进制和库，以及一个完整的用户操作系统。而容器技术是和我们的宿主机共享硬件资源及操作系统，可以实现资源的动态分配，容器包含应用和其所有的依赖包，但是与其他容器共享内核，容器在宿主机操作系统中，在用户空间以分离的进程运行。通过使用容器，我们可以轻松打包应用程序的代码、配置和依赖关系，将其变成容易使用的构建块，从而实现环境一致性、运营效率、开发人员生产力和版本控制等诸多目标。容器可以帮助保证应用程序快速、可靠、一致地部署，其间不受部署环境的影响。容器还赋予我们对资源更多的精细化控制能力，让我们的基础设施效率更高。

      Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。它是目前最流行的 Linux 容器解决方案。而 Linux 容器是 Linux 发展出了另一种虚拟化技术，简单来讲， Linux 容器不是模拟一个完整的操作系统，而是对进程进行隔离，相当于是在正常进程的外面套了一个保护层。对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。程序在这个虚拟容器里运行，就好像在真实的物理机上运行一样。有了 Docker ，就不用担心环境问题。

  - #### 逆向工程
  
    使用开源框架https://gitee.com/renrenio/renren-generator.git的代码生成器，生成基本CRUD代码，需修改renren-generator的application.yml和generator.properties配置信息



## Week 2_2022.9.5~2022.9.11

- ### 课程

- ### 自研

  
