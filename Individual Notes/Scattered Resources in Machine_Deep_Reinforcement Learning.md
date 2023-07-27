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

- [理解凸优化](https://zhuanlan.zhihu.com/p/37108430)
- [多目标优化之帕累托最优](https://zhuanlan.zhihu.com/p/54691447)
- [浅谈最优化问题的KKT条件](https://zhuanlan.zhihu.com/p/26514613)



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

## 特征预处理

- [标准化、归一化、异常特征清洗、不平衡数据](https://www.cnblogs.com/pinard/p/9093890.html)
- [不平衡数据的处理方法](https://blog.csdn.net/zhang15953709913/article/details/84635540)

## 特征表达

- [缺失值、特殊特征、离散特征、连续特征](https://www.cnblogs.com/pinard/p/9061549.html)
- [连续特征离散化的好处](http://note.youdao.com/noteshare?id=024fa3dbabf4b5a07eb72c8021e60f62)
- [什么样的模型对缺失值更敏感？](https://blog.csdn.net/zhang15953709913/article/details/88717220)

## 特征选择

- [过滤法、包装法、嵌入法](https://www.cnblogs.com/pinard/p/9032759.html) 
- [Kaggle中的代码实战](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)



# Entropy

https://zhuanlan.zhihu.com/p/74075915

https://www.zhihu.com/question/391900914/answer/2488135320

## Summary

**目前分类损失函数多用交叉熵，而不是KL散度**

首先损失函数的功能是通过样本来计算模型分布与目标分布间的差异，在分布差异计算中，KL散度是最合适的。但在实际中，某一事件的标签是已知不变的（例如设置猫的label为1，那么所有关于猫的样本都要标记为1），即目标分布的熵为常数。

而根据KL公式可以看到，**KL散度 - 目标分布熵 = 交叉熵（这里的“-”表示裁剪）**。所以不用计算KL散度，只需要计算交叉熵就可以得到模型分布与目标分布的损失值。

已知模型分布与目标分布差异可用交叉熵代替KL散度的条件是目标分布为常数。如果目标分布是有变化的（如同为猫的样本，不同的样本，其值也会有差异），那么就不能使用交叉熵，例如蒸馏模型的损失函数就是KL散度，因为蒸馏模型的目标分布也是一个模型，该模型针对同类别的不同样本，会给出不同的预测值（如两张猫的图片a和b，目标模型对a预测为猫的值是0.6，对b预测为猫的值是0.8）。

**交叉熵：**其用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小。这也是为什么在机器学习中的分类算法中，我们总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明我们算法所算出的非真实分布越接近真实分布。

**KL散度（相对熵）：**衡量不同策略之间的差异，所以我们使用KL散度来做模型分布的拟合损失。

## Self-Information 自信息

任何事件都会承载着一定的信息量，包括已经发生的事件和未发生的事件，只是它们承载的信息量会有所不同。

如昨天下雨这个已知事件，因为已经发生，既定事实，那么它的信息量就为0。如明天会下雨这个事件，因为未有发生，那么这个事件的信息量就大。从上面例子可以看出信息量是一个与事件发生概率相关的概念，而且可以得出，事件发生的概率越小，其信息量越大。

已知某个事件的信息量是与它发生的概率有关，那么可以通过如下公式计算信息量：

假设 $X$ 是一个离散型随机变量，其取值集合为 $K$，概率分布函数 $p(x)=P(K=x),x\in K$，则定义事件 $K=x_0$ 的信息量为 $I(x_0)=-log(p(x_0))$

## Entropy 熵

已知：当一个事件发生的概率为 $p(x)$，那么它的信息量是 $-log(p(x))$		

如果我们把这个事件的所有可能性罗列出来，就可以求得该事件信息量的期望，而信息量的期望就是熵，所以熵的公式表达如下：

假设事件 $X$ 共有 n 种可能，发生 $x_i$ 的概率为 $P(x_i)$ ，那么该事件的熵为 $H(X)=-\sum_{i=1}^n p(x_i)log(p(x_i))$
$$
离散情况：H(X)=-\sum_{i=1}^n p(x_i)log(p(x_i))\\
连续情况：H(x)=-\int^{+\infty}_{-\infty}{p(x)log(p(x))dx}\\
期望形式：H(x)=E_{x～p(x)}[-log(p(x))]
$$
如果式中的log以2为底的话，我们可以将这个式子解释为：要花费至少多少位的编码来表示此概率分布。从此式也可以看出，信息熵的本质是一种期望

然而有一类比较特殊的问题，比如投掷硬币只有两种可能，字朝上或花朝上。买彩票只有两种可能，中奖或不中奖，称之为0-1分布问题（二项分布的特例），对于这类问题，熵的计算方法可以简化为如下算式：

$H(X)=-\sum_{i=1}^n p(x_i)log(p(x_i))=-p(x)log(p(x))-[1-p(x)]log[1-p(x)]$

## Kullback-Leibler Divergence kL散度(相对熵)

如果对同一个随机变量 $x$ 有两个单独的概率分布 $p(x) 和 q(x)$，不妨将 $p(x)$ 看成是真实的分布，$q(x)$ 看成是估计的分布

KL散度，是指当估计分布 $q(x)$ 被用于近似真实 $p(x)$ 时的信息损失，也就是说，$q(x)$ 能在多大程度上表达 $p(x)$ 所包含的信息，KL散度越大，表达效果越差
$$
离散情况：D_{KL}(p\;||\;q)=\sum_{i=1}^np(x_i)log(\frac{p(x_i)}{q(x_i)}) \\
连续情况：D_{KL}(p\;||\;q)=\int^{+\infty}_{-\infty}{p(x)log(\frac{p(x)}{q(x)})dx}\\
期望形式：E_{x～p(x)}[log(\frac{p(x)}{q(x)})]
$$
当我们使用一个较简单、常见的分布(如均匀分布、二项分布等)来拟合我们观察到的一个较为复杂的分布时，由于拟合出的分布与观察到的分布并不一致，会有信息损失的情况出现。KL散度就是为了度量这种损失而被提出的。

因为对数函数是凸函数，所以KL散度的值为非负数。

有时会将KL散度称为KL距离，但它并不满足距离的性质：KL散度不是对称的；KL散度不满足三角不等式。

## Cross Entropy 交叉熵

$$
\begin{aligned}
D_{KL}(p\;||\;q)
& =\sum_{i=1}^np(x_i)log(\frac{p(x_i)}{q(x_i)}) \\
& = \sum_{i=1}^np(x_i)log(p(x_i)) \; - \; \sum_{i=1}^np(x_i)log(q(x_i)) \\
& = -H(p(x)) \; + \; [-\sum_{i=1}^np(x_i)log(q(x_i))]
\end{aligned}
$$

将 KL散度公式变形可得两个分式，等式的前一部分恰巧就是p的熵，等式的后一部分，就是交叉熵
$$
离散情况：H(p,q)= -\sum_{i=1}^np(x_i)log(q(x_i))\\
连续情况：H(p,q)= -\int^{+\infty}_{-\infty}{p(x)log(q(x))dx} \\
期望形式：H(p,q)= E_{x～p(x)}[-log(q(x))]
$$
在机器学习中，我们需要评估label和predicts之间的差距，使用KL散度刚刚好，即 $D_{KL}(y|| \hat{y})$，而其前一部分是熵，保持不变，故在优化过程中，只需要关注交叉熵即可。所以一般在机器学习中直接使用 Cross Entropy Loss 来评估模型



# Code

- [数据量太大导致Dataloader加载很慢](https://www.zhihu.com/question/356829360/answer/3008169314)

- [深度学习炼丹技巧](https://zhuanlan.zhihu.com/p/518189935)

- [PyTorch常用代码段合集](https://zhuanlan.zhihu.com/p/497640563)

  

# 具体模型

- Attention：[参考1](https://zhuanlan.zhihu.com/p/342235515)，[参考2](https://zhuanlan.zhihu.com/p/575643771)，[Transformer讲解——张俊林](https://zhuanlan.zhihu.com/p/37601161)，[Transformer讲解——李沐](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.0.0&vd_source=aed4f12da37d96d8f25730419892c4a9)
- Stable Diffusion：[参考1](https://zhuanlan.zhihu.com/p/583124756)，[参考2](https://huggingface.co/blog/annotated-diffusion)，[参考3](https://mp.weixin.qq.com/s/nU0_GrhWQv_gir--ZUBmOg)，[参考4](https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda)，[参考5](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)



# 面试

- [大模型面试八股文](https://zhuanlan.zhihu.com/p/643560888)
- [NLP/AI面试全记录](https://zhuanlan.zhihu.com/p/57153934)
- [牛客网面经总结](https://www.nowcoder.com/discuss/165930)
- [海量数据判重](https://www.nowcoder.com/discuss/153978)
- [常考智力题/逻辑题](https://github.com/wangyuGithub01/Machine_Learning_Resources/blob/master/pdf/IQ.md)
- [常考概率题](https://github.com/wangyuGithub01/Machine_Learning_Resources/blob/master/pdf/statistic.md)



# 推荐书籍/笔记/专栏

- 《机器学习方法》李航：原《统计学习方法》
- [《动手学深度学习》](https://zh.d2l.ai/)：深度学习教科书，pytorch代码实现
- [《动手学强化学习》](https://hrl.boyuai.com/chapter/intro/)：强化学习快速入门，代码实现
- [《Easy RL》](https://datawhalechina.github.io/easy-rl/#/)：强化学习更详细的算法解释
- [南瓜书PumpkinBook](https://datawhalechina.github.io/pumpkin-book/#/)：西瓜书推导公式的细节详述
- [《推荐系统实战》](https://github.com/wangyuGithub01/E-book/blob/master/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AE%9E%E8%B7%B5.pdf)
- [华校专学习总结笔记](http://huaxiaozhuan.com/)：数学、统计学习、深度学习、工具
- [王喆的机器学习专栏](https://zhuanlan.zhihu.com/wangzhenotes)：推荐系统、计算广告等机器学习领域前沿知识
- [荐道馆](https://www.zhihu.com/column/learningdeep)：推荐算法领域
- [美团技术团队](https://tech.meituan.com/tags/%E7%AE%97%E6%B3%95.html)：美团的技术博客，新技术与实际应用相结合
- [张俊林——深度学习前沿笔记](https://zhuanlan.zhihu.com/c_188941548)：LLM为主
- [计算广告论文、学习资料、业界分享](https://github.com/wzhe06/Ad-papers)







