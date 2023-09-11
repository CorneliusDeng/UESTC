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

- BLIP-2：

  

# Research thinking in the era of large models

在模型越来越大的时代背景下，如何利用有限的资源做出一些科研工作？

Inspired by [Yi Zhu](https://bryanyzhu.github.io/)

## 研究方向

1. Efficient(PEFT) 

   提升训练效率，这里以PEFT(parameter efficient fine tuning)为例

2. Existing stuff(pretrained model)、New directions

   使用别人的预训练模型，新的研究方向

3. plug-and-play

   做一些即插即用的模块，例如模型的模块、目标函数、新损失函数、数据增强方法等等

4. Dataset,evaluation and survey

   构建数据集、发表分析为主的文章或者综述论文

## Efficient(PEFT) 

https://huggingface.co/blog/peft

### PEFT 方法一：adapter

最早来自于这篇论文：

论文地址：http://proceedings.mlr.press/v97/houlsby19a.html
论文标题：Parameter-Efficient Transfer Learning for NLP
标题翻译：用于NLP的参数高效转移学习

Adapter层的结构，如下图右边所示：下采样FC层+非线性激活层+上采样FC层，加上残差连接

这里PEFT的方法是指，如下图左边所示，在Transformer中加入了两个adapter，进行微调时，原来的Transformer的参数都是锁住的，只有adapter层的参数在学习

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/PEFT%202.png" style="zoom:67%;" />

adapter层参数量和大模型相比非常少，例如在175B的GPT3中使用LoRA，需要训练的参数只要万分之一，因此训练成本大幅降低

### PEFT 方法二：prompt tuning

论文地址：https://link.springer.com/article/10.1007/s11263-022-01653-1
论文标题：Learning to Prompt for Vision-Language Models

prompt tuning是指可以任意调整提示词，这样的调整对最后的性能会有很大的影响，能否得到想要的结果，取决于有没有选择一个好的提示词。例如下图所示，不同的提示词对准确率的影响很大。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/PEFT%203.png)

上图是如何通过提示给图片分类的？将类别名称CLASS给模型，看哪个文字和图片的相似度最高。

- Prompt分为两种：

  - Hard Prompt：人工设置的提示词，不能修改也无法学习。设置这些需要一定的先验知识，但我们并不会总有这样的先验知识
  - Soft Prompt：将提示词设置为一个可学习的向量。如下图所示 ，将文本端(text encoder)的输入CLASS设置为learnable context，模型优化的是这个context部分。这样既可以节省很多计算量 ，也可以避免在下游任务时手动设置提示词

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/PEFT%204.png)

将可学习的Prompt方法用到纯视觉任务中，做法如下图所示

论文地址：https://arxiv.org/abs/2203.12119
论文标题：Visual Prompt Tuning

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/PEFT%205.png)

图中蓝色部分是原来训练好的模型，红色是需要微调的prompt，加入Prompt tuning有两种方式：
VPT: Deep，在每一层的输入输出都加入prompt。
VPT: Shallow，在输入端加入prompt

综述文章，近期PEFT方法总结，从统一的观点进行归纳：
论文地址：https://openreview.net/forum?id=0RDcd5Axok

### Example

通过论文AIM为例讲述如何进行PEFT，即在硬件资源有限时对大模型进行高效微调

- 论文地址：https://arxiv.org/abs/2302.03024
- 论文标题：AIM: Adapting Image Models for Efficient Video Action Recognition
- 标题翻译：调整图像模型以实现高效的视频动作识别

**思考：已经训练好的图像模型是否需要继续微调？**

1. clip已经证明了即使ZeroShot(模型不变，直接在各个数据集上进行推理)，它的效果也很好。即一个训练很好的图片模型从中提取视觉特征是有泛化性、有效的
2. 继续微调会导致灾难性遗忘。如果使用少量数据在大模型上微调，可能会直接过拟合，或者大模型的很多特征丢失

**结论：预训练的图像模型不需要继续微调**

传统模型和论文改进的微调方法对比图：
<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/PEFT%201.png" style="zoom: 67%;" />

因此，论文的做法是，尝试将模型参数锁住，在上面加一些时序处理模块、目标函数等修改周边的方式(即PEFT)让图片模型能够做视频理解的任务，不需要重新训练视频模型，省时省力。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/PEFT%206.png)

如上图所示，AIM模型就是在图b的ViT模型中加入图a的Adapter，共有图c、d、e三种方式：

1、Spatial Adaptation，只在S-MSA层后面加入Adapter，即不增加视频理解能力，只加一些学习的参数，实验是否能从图像学习的特征迁移到视频数据集中，并稍微解决 domain gap 问题

2、Temporal Adaptation，复用一个MSA层，在两个MSA层后面都加入Adapter，即让模型从Spatial和Temporal两个方向上进行学习，从而有时序建模的能力

3、Joint Adaptation，在Temporal Adaptation的基础上，在MLP边上也加入Adapter，即让三个Adapter各司其职，使得优化问题更简单一些

注：MSA是多头自注意力(MultiHead Self-Attention，S-MSA和T-MSA共享权重，但维度不同。

## Existing stuff(pretrained model)

有两点：

1、巧妙使用别人的预训练模型，从而达到去做FewShot，ZeroShot，或者最多Fine Tuning的实验。

2、新的研究方向。

通过这篇论文讲述这两点是如何运用的：
论文地址：https://arxiv.org/abs/2207.05027
论文标题：Unsupervised Semantic Segmentation with Self-supervised Object-centric Representations

从标题就可以看出这两点技巧：
1、这里的Self-supervised是指使用了预训练好的DINO、DeepUSPS、BASNet等网络
2、这里做的方向是Object-centric Learning，属于蓬勃发展的题目，玩家不多、数据集不大

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/Existing%20stuff.png)

上图展示了如何使用几个预训练好的模型，在无监督的情况下找到新的物体，步骤如下：

1、通过预训练模型**DeepUSPS**找到一些显著性物体的Mask。

例如，图片中的篮球可以得到一个圆形的Mask

2、根据Mask将图片中的对应物体抠出来，并调整大小为224*224。

例如，将图片中的篮球抠出来并放大

3、然后将步骤2得到的图片通过预训练模型**DINO**返回一个1024*1024的特征(global representation)。

4、将所有的特征进行聚类**Clustering**，这样就可以通过无监督学习得到这些物体的分类ID。

注：聚类只能将相同的物体分类到一起，但并不知道具体是什么物体。

5、将图片和对应的分类ID去训练一个语义分割网络(**Semantic segmentation network**)。

注：这里相当于一个有监督的学习，标签来自于步骤4

6、一张图片可能有多个物体，所以加一个**Self-training**，多做几个轮回。

这样就可以从图片中找到物体了。

## plug-and-play

做一些通用的、即插即用的模块，在一个设定的范围内，加入了这样的模块后，能够有一个统一的涨点，并且能给出合适的分析，就非常有说服力了。通过MixGen论文讲述如何加入模块：
论文地址：https://arxiv.org/abs/2206.08358
论文标题：MixGen: A New Multi-Modal Data Augmentation

文本的模型都很大，图片的模型相对来说小一些，但是自注意力的参数是可以共享的，所以尝试用文本大模型来蒸馏图片小模型
注：模型蒸馏：使用训练集训练出来一个完整复杂的teacher模型，然后设计一个小规模的student模型，再固定teacher模型的权重参数，然后使用训练集和teacher模型的输出同时对student模型进行训练，此时就需要设计一系列loss，让student模型在蒸馏学习的过程中逐渐向teacher模型的表现特性靠拢，使得student模型的预测精度逐渐逼近teacher模型。

**为什么之前图片模型不做数据增强？**
1、图片模型训练时已经用了很多图片了，不需要再做数据增强。
2、或者做了数据增强，但是将其中的Color Jittering和Random Filp去掉了，因为这两个对图片的变化会导致图片和文本不匹配。

例如：图片有白色的狗和绿色的树，只对图片做Color Jittering会导致颜色变化，图片中不再是白色的狗，但是文本依然是白色的狗，这样文本和图片就不匹配了。

论文的做法：既然目标是尽可能保留更多信息，这里的做法很简单粗暴，就是直接将两个句子拼接在一起，这样就可以做到不丢失信息的情况下得到新的训练样本。

例如下图，将两个图片通过数据增强得到第三个图片，同时将两个图片的文本进行拼接得到第三个图片的文本。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Something%20Summary/plug-and-play.png)

审稿人的建设性提议：在下游任务只有少量数据时进行数据增强。

## Dataset,evaluation and survey

构建数据集、发表分析为主的文章或者综述论文，这里举了两篇论文为例。

以数据集为主的big detection，将三个数据集整合到一起：
论文地址：https://arxiv.org/abs/2203.13249

视频动作检测的综述论文：
论文地址：https://arxiv.org/abs/2012.06567



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







