# 梯度下降 Gradient Descent

- [理解梯度下降法](https://zhuanlan.zhihu.com/p/36902908)
- [梯度下降法、牛顿法和拟牛顿法](https://zhuanlan.zhihu.com/p/37524275)
- [深度学习优化算法SGD、Momentum、Adagrad等](https://zhuanlan.zhihu.com/p/32230623)



# MLE & MAE & Bayes

- [一文搞懂极大似然估计](https://zhuanlan.zhihu.com/p/26614750)

- [频率派和贝叶斯派的核心差异](https://www.zhihu.com/question/268906722/answer/2159781976)
- [从最大似然到EM算法浅解](https://blog.csdn.net/zouxy09/article/details/8537620)
- [先验概率 & 后验概率](https://zhuanlan.zhihu.com/p/38567891)
- [最大似然估计MLE与贝叶斯估计](https://blog.csdn.net/bitcarmanlee/article/details/52201858)



# 凸优化 Convex Optimization

- [从零开始学习凸优化](https://www.zhihu.com/question/68418633/answer/3130746614)

- [理解凸优化](https://zhuanlan.zhihu.com/p/37108430)

- 凸优化综述：[上](https://zhuanlan.zhihu.com/p/72660217)，[下](https://zhuanlan.zhihu.com/p/73028673)

- [多目标优化之帕累托最优](https://zhuanlan.zhihu.com/p/54691447)

- [浅谈最优化问题的KKT条件](https://zhuanlan.zhihu.com/p/26514613)

- 关键概念总结

  - 仿射集：是一个平移后的线性空间

  - 凸集：对随机向量的期望运算封闭

  - 凸函数：首先是定义在凸集上的，其次期望的函数值小于等于函数值的期望

  - 凸优化：可行域是凸集，且目标函数是凸函数

    

# 拉格朗日对偶性 Lagrange duality

## 原始问题

假设 $f(x),c_i(x),h_j(x)$ 是定义在 $\mathcal{R}^n$ 上的连续可微函数，考虑约束最优化问题：
$$
\begin{aligned}
& \underset{x\in\mathcal{R}^n}{\min} f(x) \\
& \begin{array}{r@{\quad}r@{}l@{\quad}l}
s.t. & c_i(x)\leq 0, \quad i = 1,2,\cdots,k \\
     & h_j(x)=0, \quad j = 1,2,\cdots,l
\end{array}
\end{aligned}
$$
则此约束优化问题为原始最优化问题或原始问题。

首先，引入广义拉格朗日函数 generalized Lagrange function：
$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^k\alpha_ic_i(x)+\sum_{j=1}^l\beta_jh_j(x)
$$
其中，$x=(x^{(1)},x^{(2)},\cdots,x^{(n)})^T\in \mathcal{R}^n$，而 $\alpha_i,\beta_j$ 是拉格朗日乘子，$\alpha_i\geq 0$，考虑 $x$ 的函数：
$$
\theta_P(x)=\underset{\alpha,\beta;\alpha_i\geq0}{\max}L(x,\alpha,\beta)
$$
下标 $P$ 表示原始问题

假设给定某个 $x$，如果 $x$ 违反原始问题的约束条件，即存在某个 $i$ 使得 $c_i(x)>0$ 或者存在某个 $j$ 使得 $h_j(x)\neq0$，那么就有
$$
\theta_P(x)=\underset{\alpha,\beta;\alpha_i\geq0}{\max}
\left[f(x)+\sum_{i=1}^k\alpha_ic_i(x)+\sum_{j=1}^l\beta_jh_j(x) \right]
=+\infty
$$
 因为若某个 $i$ 使约束 $c_i(x)>0$，则可令 $\alpha\rightarrow +\infty$；若某个 $j$ 使 $h_j(x)\neq0$，则可令 $\beta_j$ 使得 $\beta_jh_j(x)\rightarrow+\infty$，而将其余各 $\alpha_i,\beta_j$ 均取为0

相反地，如果 $x$ 满足约束条件，则 $\theta_P(x)=f(x)$，因此
$$
\theta_P(x)=
\begin{cases}
f(x), & x \text{满足原始问题约束} \\
+\infty, & \text{其他}
\end{cases}
$$
所以如果考虑极小化问题：
$$
\underset{x}{\min}\theta_P(x)=\underset{x}{\min}\underset{\alpha,\beta;\alpha_i\geq0}{\max}L(x,\alpha,\beta)
$$
它与原始问题是等价的，即它们具有相同解。问题 $\underset{x}{\min}\underset{\alpha,\beta;\alpha_i\geq0}{\max}L(x,\alpha,\beta)$ 称为广义拉格朗日函数的极小极大问题。这样一来，就把原始最优化问题表示为拉格朗日函数的极小极大问题，不妨定义原始问题的最优值 $p^*=\underset{x}{\min}\theta_P(x)$ 为原始问题的值

## 对偶问题

定义
$$
\theta_D(\alpha,\beta)=\underset{x}{\min}L(x,\alpha,\beta)
$$
再考虑极大化 $\theta_D(\alpha,\beta)=\underset{x}{\min}L(x,\alpha,\beta)$，即
$$
\underset{\alpha,\beta;\alpha_i\geq0}{\max}\theta_D(\alpha,\beta)=\underset{\alpha,\beta;\alpha_i\geq0}{\max}\underset{x}{\min}L(x,\alpha,\beta)
$$
问题 $\underset{\alpha,\beta;\alpha_i\geq0}{\max}\underset{x}{\min}L(x,\alpha,\beta)$ 称为广义拉格朗日函数的极大极小问题

可以将广义拉格朗日函数的极大极小问题表示为约束最优化问题
$$
\begin{aligned}
& \underset{\alpha,\beta}{\max}\theta_D(\alpha,\beta)=\underset{\alpha,\beta}{\max}\underset{x}{\min}L(x,\alpha,\beta) \\
& \begin{array}{r@{\quad}r@{}l@{\quad}l}
s.t. & \alpha_i\geq 0, \quad i = 1,2,\cdots,k
\end{array}
\end{aligned}
$$
称为原始问题的对偶问题，定义对偶问题的最优值：$d^*=\underset{\alpha,\beta;\alpha_i\geq0}{\max}\theta_D(\alpha,\beta)$，称为对偶问题的值

## 原始问题和对偶问题的关系

定理1：若原始问题和对偶问题都有最优值，则
$$
d^*=\underset{\alpha,\beta;\alpha_i\geq0}{\max}\underset{x}{\min}L(x,\alpha,\beta) \leq \underset{x}{\min}\underset{\alpha,\beta;\alpha_i\geq0}{\max}L(x,\alpha,\beta) = p^*
$$
推论：设 $x^*$ 和 $\alpha^*,\beta^*$ 分别是原始问题和对偶问题的可行解，并且 $d^*=p^*$，则  $x^*$ 和 $\alpha^*,\beta^*$ 分别是原始问题和对偶问题的最优解

定理2：考虑原始问题和对偶问题。假设函数 $f(x)$ 和 $c_i(x)$ 是凸函数，$h_j(x)$ 是仿射函数，并且不等式约束 $c_i(x)$ 是严格执行的，即存在 $x$，对所有 $i$ 有 $c_i(x)<0$，则存在 $x^*,\alpha^*,\beta^*$，使 $x^*$ 是原始问题的解， $\alpha^*,\beta^*$ 是对偶问题的解，并且 $p^*=d^*=L(x^*,\alpha^*,\beta^*)$

定理3：对原始问题和对偶问题，假设函数 $f(x)$ 和 $c_i(x)$ 是凸函数，$h_j(x)$ 是仿射函数，并且不等式约束 $c_i(x)$ 是严格执行的，则 $x^*$ 和 $\alpha^*,\beta^*$ 分别是原始问题和对偶问题的解的充分必要条件是  $x^*,\alpha^*,\beta^*$ 满足下面的 Karush-Kuhn-Tucker(KTT) 条件：
$$
\begin{align}
& \nabla_x L(x^*,\alpha^*,\beta^*)=0 \\
& \alpha^*_ic_i(x^*) = 0, \quad i = 1,2,\cdots,k \\
& c_i(x^*) \leq 0, \quad i = 1,2,\cdots,k \\
& \alpha^* \geq 0, \quad i = 1,2,\cdots,k \\
& h_j(x^*)=0, \quad j = 1,2,\cdots,l \\
\end{align}
$$
其中，第二条称为 KTT 的对偶互补条件，由此条件可知：若 $\alpha^*>0$，则 $c_i(x^*)=0$



# 特征工程

## 特征选择

- [过滤法、包装法、嵌入法](https://www.cnblogs.com/pinard/p/9032759.html) 
- [Kaggle中的代码实战](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)

## 特征表达

- [缺失值、特殊特征、离散特征、连续特征](https://www.cnblogs.com/pinard/p/9061549.html)
- [连续特征离散化的好处](http://note.youdao.com/noteshare?id=024fa3dbabf4b5a07eb72c8021e60f62)
- [什么样的模型对缺失值更敏感？](https://blog.csdn.net/zhang15953709913/article/details/88717220)

## 特征预处理

- [标准化、归一化、异常特征清洗、不平衡数据](https://www.cnblogs.com/pinard/p/9093890.html)
- [不平衡数据的处理方法](https://blog.csdn.net/zhang15953709913/article/details/84635540)



# 博弈论 Game Theory

- [斯塔克尔伯格博弈](https://www.zhihu.com/question/475143505/answer/2638019081)，[推荐系统中的斯塔克尔伯格博弈](https://zhuanlan.zhihu.com/p/380135679)
- [平均场博弈](https://zhuanlan.zhihu.com/p/265578530)，[深度学习方法求解平均场博弈论问题](https://zhuanlan.zhihu.com/p/419182257)
- [纳什均衡](https://zhuanlan.zhihu.com/p/593411677)
- [帕累托最优](https://blog.csdn.net/qq_42364307/article/details/115096393)，[通俗地解释帕累托最优](https://www.zhihu.com/question/22570835/answer/21816685)




# CUDA

- [CUDA入门教程推荐](https://zhuanlan.zhihu.com/p/346910129)
- [CUDA-Python入门教程](https://blog.csdn.net/qq_42596142/article/details/103157468)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [UESTC Course：GPU Parallel Programming](https://i.study.uestc.edu.cn/GPUPP/menu/home)
- [谭升的博客：CUDA](https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89)



# Code 

- [数据量太大导致Dataloader加载很慢](https://www.zhihu.com/question/356829360/answer/3008169314)

- [深度学习炼丹技巧](https://zhuanlan.zhihu.com/p/518189935)

- Tricks in Deep Learning：[参考1](https://www.zhihu.com/question/540433389/answer/2549775065)，[参考2](https://www.zhihu.com/question/540433389/answer/2576569581)，[参考3](https://www.zhihu.com/question/540433389/answer/2551517886)

- [PyTorch常用代码段合集](https://zhuanlan.zhihu.com/p/497640563)

- [Annotated  Deep Learning Paper Implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)：非常好！

- [Machine Learning in Numpy](https://github.com/ddbourgin/numpy-ml)：值得一看

  

# 具体模型

- Attention：[Attention 图解](https://zhuanlan.zhihu.com/p/342235515)，[浅谈Attention机制(Self-Attention，QKV矩阵)](https://zhuanlan.zhihu.com/p/575643771)，[Transformer讲解——张俊林](https://zhuanlan.zhihu.com/p/37601161)，[Transformer讲解——李沐](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.0.0&vd_source=aed4f12da37d96d8f25730419892c4a9)，[图解 Transformers 的数学原理](https://zhuanlan.zhihu.com/p/654051912)

- Stable Diffusion：[参考1](https://zhuanlan.zhihu.com/p/583124756)，[参考2](https://huggingface.co/blog/annotated-diffusion)，[参考3](https://mp.weixin.qq.com/s/nU0_GrhWQv_gir--ZUBmOg)，[参考4](https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda)，[参考5](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

- BERT：[BERT 论文逐段精读——李沐](https://www.bilibili.com/video/BV1PL411M7eQ?vd_source=973b1421e263dea47bcea4ee868b05a3)

- ViT：[ViT论文逐段精读](https://www.bilibili.com/video/BV15P4y137jb?vd_source=973b1421e263dea47bcea4ee868b05a3)，[ViLT 论文精读](https://www.bilibili.com/video/BV14r4y1j74y?vd_source=973b1421e263dea47bcea4ee868b05a3)

- CLIP：[CLIP 论文逐段精读](https://www.bilibili.com/video/BV1SL4y1s7LQ?vd_source=973b1421e263dea47bcea4ee868b05a3)，[CLIP改进工作串讲(上)](https://www.bilibili.com/video/BV1FV4y1p7Lm?vd_source=973b1421e263dea47bcea4ee868b05a3)，[CLIP改进工作串讲(下)](https://www.bilibili.com/video/BV1gg411U7n4?vd_source=973b1421e263dea47bcea4ee868b05a3)

- Multimodal：[多模态串讲(上)](https://www.bilibili.com/video/BV1Vd4y1v77v?vd_source=973b1421e263dea47bcea4ee868b05a3)，[多模态串讲(下)](https://www.bilibili.com/video/BV1fA411Z772?vd_source=973b1421e263dea47bcea4ee868b05a3)

- PEFT：[参考1](https://zhuanlan.zhihu.com/p/491407745)，[参考2](https://zhuanlan.zhihu.com/p/551616933)

- BLIP-2：

  

# 面试

- [大模型面试八股文](https://zhuanlan.zhihu.com/p/643560888)
- [NLP/AI面试全记录](https://zhuanlan.zhihu.com/p/57153934)
- [牛客网面经总结](https://www.nowcoder.com/discuss/165930)，[答案](https://www.nowcoder.com/ta/review-ml?query=&asc=true&order=&tagQuery=&page=1)
- [海量数据判重](https://www.nowcoder.com/discuss/153978)
- [常考智力题/逻辑题](https://github.com/wangyuGithub01/Machine_Learning_Resources/blob/master/pdf/IQ.md)
- [常考概率题](https://github.com/wangyuGithub01/Machine_Learning_Resources/blob/master/pdf/statistic.md)
- [商汤研究院基础视觉组(大模型专题)](https://mp.weixin.qq.com/s?__biz=MzkwNDE4Nzg1MQ==&mid=2247493130&idx=1&sn=777abbae93825972dcf9f42afc2d4c6e&chksm=c0887caef7fff5b804dcd3e58f7abff36701a300b4c09a3de09d46640dfb7547e9ff8b5bd0b5&mpshare=1&scene=1&srcid=0827SraNLKVodOjvLDBGhX1H&sharer_sharetime=1693123853559&sharer_shareid=6224ae01f0b343747219af88e94201f4&from=singlemessage&isappinstalled=0&clicktime=1693797129&enterid=1693797129&ascene=1&devicetype=iOS16.6&version=18002831&nettype=WIFI&lang=zh_CN&countrycode=CN&fontScale=100&exportkey=n_ChQIAhIQLXWe3YrOonRVPbuDLDTfkRLjAQIE97dBBAEAAAAAAIoNOKCKUT4AAAAOpnltbLcz9gKNyK89dVj0rY46vSSDnwOGNMHBMXI28op6Yew4La3ZjEWzzIX3qE10SaNyayyZdueWq8kYN%2FG4XokYwlRKpwt2g0nIRhM%2Brp6dn1LKvzbFFtgLuoGJMFw%2Fh2CTd1YeZwg%2FkKzBVI2psWzfFwi5lJJNvEaqkFgBwJVzJOuY7LkQ5SSfkF8mTFYwSy5BQwTKomFgsyek2SNFhNpiCA02HvfwF6KZxe9yd4%2B2aKuSLEIAw9sPPNlJ1x1NjP9lP%2B8Wl%2BbcOO7L&pass_ticket=aSPwmN2aoMEAuxUvIaZKTdY76A6olWQBX9%2FTWISIjXpmcxGUfU926aeQtLarpdx0&wx_header=3)



# 书籍/笔记/专栏

- 《机器学习方法》李航：原《统计学习方法》
- [《动手学深度学习》](https://zh.d2l.ai/)：深度学习教科书，pytorch代码实现
- [《动手学强化学习》](https://hrl.boyuai.com/chapter/intro/)：强化学习快速入门，代码实现
- [Explanation to key concepts in ML](https://github.com/dair-ai/ML-Papers-Explained)
- [Start Here with Machine Learning](https://machinelearningmastery.com/start-here/)
- [《Easy RL》](https://datawhalechina.github.io/easy-rl/#/)：强化学习更详细的算法解释
- [南瓜书PumpkinBook](https://datawhalechina.github.io/pumpkin-book/#/)：西瓜书推导公式的细节详述
- [《推荐系统实战》](https://github.com/wangyuGithub01/E-book/blob/master/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AE%9E%E8%B7%B5.pdf)
- [华校专学习总结笔记](http://huaxiaozhuan.com/)：数学、统计学习、深度学习、工具
- [王喆的机器学习专栏](https://zhuanlan.zhihu.com/wangzhenotes)：推荐系统、计算广告等机器学习领域前沿知识
- [荐道馆](https://www.zhihu.com/column/learningdeep)：推荐算法领域
- [美团技术团队](https://tech.meituan.com/tags/%E7%AE%97%E6%B3%95.html)：美团的技术博客，新技术与实际应用相结合
- [张俊林——深度学习前沿笔记](https://zhuanlan.zhihu.com/c_188941548)：LLM为主
- [计算广告论文、学习资料、业界分享](https://github.com/wzhe06/Ad-papers)
- [Annotated PyTorch Paper Implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)：用 pytorch 框架实现各种深度学习模型，并且带有讲解







