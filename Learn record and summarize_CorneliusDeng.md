## Week 1_2022.8.29~2022.9.4

- ### 课程

  - #### Design and Analysis of Algorithms

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

        P: a solution can be solved in polynomial time.（多项式时间）

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

  - #### Distributed Systems

    - **Why do we need Distributed System?**

      Functional Separation（功能分离）、Inherent distribution（固有的分布性）、Power imbalance and load variation（负载均衡）、Reliability（可靠性）、Economies（经济性）

    - **goal：**资源共享 (resource sharing)、协同计算 (collaborative computing)
    - **definition：**A distributed system is one in which components located at networked computers communicate and coordinate their actions only by passing messages.
    - **fundamental feature：**并发性 (concurrency)、无全局时钟 (non-global clock)、故障独立性 (independent failure)
    - **challenge：**异构性（Heterogeneity）、开放性（Openness）、安全性（Security）、可伸缩性（Scalability）、故障处理（Failure handling）、并发行（Concurrency）、透明性（Transparency）
    - **RPC：**Remote Procedure Call

  - #### English for Academic Communication and Presentation

    - **Five Authorities in Artificial Intelligence**
    
      Alan Mathison Turing、John von Neumann、Andrew Chi-Chih Yao、Leslie B. Lamport、James Gosling
    
    - **Five Top Journals in Artificial Intelligence**
    
      NAME:《Artificial Intelligence》, PRESS: Elsevier
    
      NAME:《IEEE Trans on Pattern Analysis and Machine Intelligence》, PRESS: IEEE
    
      NAME:《International Journal of Computer Vision》, PRESS: Springer
    
      NAME:《IEEE Transactions on Evolutionary Computation》, PRESS: IEEE
    
      NAME:《Journal of Machine Learning Research》, PRESS: MIT Press
    
    - **Five Top Conferences in Artificial Intelligence**
    
      NAME:《AAAI Conference on Artificial Intelligence》, PRESS: AAAI
    
      NAME:《Annual Conference on Neural Information Processing Systems》，PRESS: MIT Press
    
      NAME:《Annual Meeting of the Association for Computational Linguistics》,PRESS: ACL
    
      NAME:《International Conference on Machine Learning》, PRESS: ACM
    
      NAME:《International Joint Conference on Artificial Intelligence》, PRESS: Morgan Kaufmann

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
    
    整合MyBatis-Plus
    	1）导入依赖
    	2）配置
    		Ⅰ、配置数据源：导数数据库驱动；在application.yml中配置数据源相关信息
    		Ⅱ、配置Mybatis-Plus：使用MapperScan注解；告诉Mybatis-plus，sql映射文件位置



## Week 2_2022.9.5~2022.9.11

- ### 课程

  - #### Design and Analysis of Algorithms

    ​	review, consider more possibilities about The Stable Matching Problem

  - - 

  - #### Thrifles

    Daily nucleic acid test, ordered MacBook Pro arrived

    Think more about the future 

- ### 自研

  - #### 分布式组件SpringCloud Alibaba

    Spring Cloud 是一个服务治理平台，是若干个框架的集合，提供了全套的分布式系统解决方案。包含了：服务注册与发现、配置中心、服务网关、智能路由、负载均衡、断路器、监控跟踪、分布式消息队列等等。Spring Cloud Alibaba 致力于提供微服务开发的一站式解决方案。此项目包含开发分布式应用微服务的必需组件，方便开发者通过 Spring Cloud 编程模型轻松使用这些组件来开发分布式应用服务。

    - **GuliMall技术搭配方案**

      SpringCloud Alibaba - Nacos：注册中心（服务发现/注册）、配置中心（动态配置管理）

      SpringCloud - Ribbon：负载均衡

      SpringCloud - Feign：声明式HTTP客户端（调用远程服务）

      SpringCloud Alibaba - Sentinel：服务容错（限流、降级、熔断）

      SpringCloud - Gateway：API网关（webflux编程模式）

      SpringCloud - Sleuth：调用链监控

      SpringCloud Alibaba - Seata：分布式事务解决方案

    - **Nacos**

      Nacos默认级群启动，测试时需要修改为单机启动，使用命令startup.cmd -m standalone

      **注册中心：**

      ​	微服务注册到Nacos只需三步：引入依赖，各微服务的application.yml中配置Nacos Server 地址：spring.cloud.nacos.discovery.server-addr=127.0.0.1:8848，然后使用 @EnableDiscoveryClient 注解开启服务注册与发现功能。
      
      **配置中心：**
      
      ​	使用Nacos作为配置中心统一管理配置：修改 pom.xml 文件，引入 Nacos Config Starter；创建bootstrap.properties 配置文件，其中配置 Nacos Config 元数据；需要给配置中心默认添加一个数据集（默认规则：微服务名.properties）；给微服务名.properties添加任何配置；动态获取配置：在controller文件中添加注解@RefreshScope（动态获取并刷新配置），以及使用@value（“￥{配置项的名}”）获取配置数据，而nacos配置中心内容优先级高于项目本地的配置内容。对于高版本的springboot需要导入spring-cloud-starter-bootstrap依赖。
      
      ​	**配置中心进阶：**	
      
      ​		命名空间：用作配置隔离（一般每个微服务创建一个命名空间，只加载自己命名空间下的配置），默认新增的配置都在public空间下，开发、测试、生产环境可以用命名空间分割，properties每个空间有一份。注意须在bootstrap.properties配置具体使用哪个命名空间（默认为puhlic）：spring.cloud.nacos.config.namespace=xxx # 命名空间ID
      
      ​		配置集：一组相关或不相关配置项的集合
      
      ​		配置集ID：类似于配置文件名，即Data ID
      
      ​		配置分组：默认所有的配置集都属于DEFAULT_GROUP，可在bootstrap.properties指定配置分组。每个微服务创建自己的命名空间，然后使用配置分组区分环境（dev/test/prod）
      
      ​		加载多配置集：把一个冗长的application.yml配置文件拆分，将其内容都分类别抽离出去。微服务任何配置信息，任何配置文件都可以放在配置中心里，只需要在bootstrap.properties说明加载配置中心里哪些配置文件即可；以前SpringBoot任何方法从配置文件中获取值，都能使用；配置中心有的有优先使用配置中心里的。
      
      **网关：**
      
      ​	网关是请求浏览的入口，常用功能包括路由转发，权限校验，限流控制等，网关动态地管理每个微服务的地址，他能从注册中心中实时地感知某个服务上线还是下线。
      
      ​	**三大核心概念：**
      
      ​		**Ⅰ、路由Route:** The basic building block of the gateway. It is defined by an ID, a destination URI, a collection of predicates, and a collection of filters. A route is matched if the aggregate predicate is true.发一个请求给网关，网关要将请求路由到指定的服务。路由有id，目的地uri，断言的集合，匹配了断言就能到达指定位置。
      ​		**Ⅱ、断言Predicate:** This is a Java 8 Function Predicate. The input type is a Spring Framework ServerWebExchange. This lets you match on anything from the HTTP request, such as headers or parameters. java里的断言函数，匹配请求里的任何信息，包括请求头等。
      
      ​		**Ⅲ、过滤器Filter:** These are instances of GatewayFilter that have been constructed with a specific factory. Here, you can modify requests and responses before or after sending the downstream request.过滤器请求和响应都可以被修改。
      
      ​		**总结：**客户端发请求给服务端，中间有网关。先交给映射器，如果能处理就交给handler处理，然后交给一系列filer，然后给指定的服务，再返回回来给客户端。
      
      ​	**网关使用：**
      
      ​		1）开启服务注册发现@EnableDiscoveryClient
      
      ​		2）在applicaion.properties配置nacos注册中心地址

  - #### ECMAScript6（ES6）

    ECMAScript是浏览器脚本语言的规范，JS是规范的具体实现

    - let&const
    
      var 声明的变量往往会越域， let 声明的变量有严格局部作用域
    
      var 可以声明多次， let 只能声明一次
    
      var 会变量提升， let 不存在变量提升
    
      const声明常量，const声明之后不允许改变，一但声明必须初始化，否则会报错
    
    - 解构&字符串
    
      数组解构：let arr = [...], const [x,y,z] = arr, console.log(x,y,z)
    
      对象解构：const objectname=  {...}, const {key...} = objectname
    
      字符串拓展：str.startwith/endwith/include/includes
    
      字符串模板：用反引号括住即可（·content·）
    
      字符串插入变量和表达式。变量名写在 ${} 中，${} 中可以放入 JavaScript 表达式
    
    - 函数优化
    
      原来想要函数默认值得这么写b = b || 1; 现在可以直接写成function add(a, b = 1)
    
      函数不定参数function fun(...values)
    
      箭头函数：var print = obj => console.log(obj); print("hello");   在一个对象里，箭头函数this不能使用，要想获取必须是对象.属性
    
    - 对象优化
    
      可以获取map的键值对等Object.keys()、values、entries
    
      Object.assgn(target,source1,source2) 合并
    
      const person = { age, name } //声明对象简写
    
      //  拷贝对象（深拷贝） let p1 = { name: "Amy", age: 15 }       let someone = { ...p1 }
    
    - map、reduce
    
      map()：接收一个函数，将原数组中的所有元素用这个函数处理后放入新数组返回。
    
      reduce() 为数组中的每一个元素依次执行回调函数，不包括数组中被删除或从未被赋值的元素
    
      arr.reduce(callback,[initialValue])其参数意义
           1、previousValue （上一次调用回调返回的值，或者是提供的初始值（initialValue））
          2、currentValue （数组中当前被处理的元素）
          3、index （当前元素在数组中的索引）
          4、array （调用 reduce 的数组）*/
    
    - promise异步编排
    
      嵌套ajax的时候很繁琐。解决方案：1、把Ajax封装到Promise中，let p = new Promise(resolve, reject){...}。2、在Ajax中成功使用resolve(data)，交给then处理。3、失败使用reject(err)，交给catch处理p.then().catch()
    
    - 模块化
    
      模块化就是把代码进行拆分，方便重复利用。类似于java中的导包，而JS换了个概念，是导模块。模块功能主要有两个命令构成 export 和import：export用于规定模块的对外接口、import用于导入其他模块提供的功能。export不仅可以导出对象，一切JS变量都可以导出。比如：基本类型变量、函数、数组、对象。
    
    
    


## Week 3_2022.9.12~2022.9.18

- ### 课程

  - #### Linear algebra

    - 行列式

    - 矩阵
    - 向量
    - 方程组求解
    - 特征值和特征向量
    - 二次型

  - #### TEMP

- ### 自研

  - #### VUE

  - #### TEMP







组会：

我目前在学习一个开源的微服务的架构分布式项目

了解redis服务、docker容器的基本概念与基本操作，并将其部署到了我自己的阿里云轻量级服务器上

学会使用开源的代码生成器，根据数据库信息，采用逆向工程，生成后端SpirngBoot框架下的基本CRUD代码

学习大型项目的分布式方案SpringCloud Alibaba，目前正在学习其中的注册中心/配置中心Nacos组件、以及调用远程服务的Feign组件

学习了JavaScript语言的新规范ECMAScript 6.0



