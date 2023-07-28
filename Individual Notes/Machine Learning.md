# 简介 Introduce

- **机器学习定义**
  - Arthur Samuel(1959): Field of study that gives computers the ability to learn without being explicitly programmed. 在没有明确设置的情况下，使计算机具有学习能力的研究领域。
  
  -  Tom Mitchell(1998): Well-posed Learning Problem:A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 计算机程序从经验E中学习解决某一任务T，进行某一性能度量P，通过P测定在T上的表现因经验E而提高。例如，在人机玩跳棋游戏中，经验E是程序与自己下几万次跳棋；任务T是玩跳棋；性能度量P是与新对手玩跳棋时赢的概率。
  
- **机器学习分类**
  - 监督学习(Supervised Learning): 教计算机如何去完成任务。它的训练数据是有标签的，训练目标是能够给新数据（测试数据）以正确的标签。
    - 回归Regression
  
    - 分类Classification
  
  - 无监督学习(Unsupervised Learning)：让计算机自己进行学习。它的训练数据是无标签的，训练目标是能对观察值进行分类或者区分等。
  
  - 强化学习(Reinforcement Learning)：智能体以“试错”的方式进行学习，通过与环境进行交互获得的奖赏指导行为，目标是使智能体获得最大的奖赏。
- **机器学习算法**
  - 监督学习算法：线性回归、Logistic回归、神经网络、支持向量机等。
  - 无监督学习算法：聚类、降维、异常检测算法等。
  - 特殊算法：推荐算法等。



# 线性回归 Linear Regression

线性回归是利用数理统计中回归分析来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。
回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。
如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。

## 单变量线性回归 Univariate linear regression

单变量线性回归(Univariate linear regression)又称一元线性回归(Linear regression with one variable)

- **符号标记**

  𝑚 代表训练集中实例的数量
  𝑥 代表特征/输入变量
  𝑦 代表目标变量/输出变量
  (𝑥, 𝑦) 代表训练集中的实例
  (𝑥(𝑖), 𝑦(𝑖)) 代表第𝑖 个观察实例
  ℎ 代表学习算法的解决方案或函数也称为假设（hypothesis）

- **监督算法学习工作流程**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/procedure%20of%20supervised%20learning.png)

- **线性回归模型表示**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/modal%20of%20linear%20regression.png)

  其中θ是模型参数，x是输入变量/特征，y是输出/目标变量

- **代价函数 Cost Function**

  ​	代价函数也被称作平方误差函数，有时也被称为平方误差代价函数。我们之所以要求出误差的平方和，是因为误差平方代价函数，对于大多数问题，特别是回归问题，都是一个合理的选择。还有其他的代价函数也能很好地发挥作用，但是平方误差代价函数可能是解决回归问题最常用的手段。

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/cost%20function.png)

- **梯度下降 Gradient Descent**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Gradient%20descent.png)

  - **梯度下降的缺点**
    - 只能知道导数方向，不知道与最优点的距离；
    - 不能保证全局最优性。

- **线性回归的梯度下降**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/GradientDescentForLinearRegression.png)

## 多变量线性回归 Linear Regression with Multiple Variables 

- **多维特征 Multiple Features**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Multiple%20Features.png)

- **多变量梯度下降 Gradient Descent for Multiple Variables**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Gradient%20Descent%20for%20Multiple%20Variables.png)

- **特征缩放 Feature scaling**

  特征缩放(Feature scaling)是为了确保特征在一个相近的范围内, 使得算法更快收敛。可以使用均值归一化的方法实现特征缩放。
  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Feature%20scaling.png)

- **学习率**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/learn%20rate.png)

- **特征和多项回归**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/feature%20and%20mutilregression.png)

- **正规方程 Normal Equation**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Normal%20Equation.png)



# 逻辑回归 Logistic Regression

Logistic回归是一种广义的线性回归分析模型。它是一种分类方法，可以适用于二分类问题，也可以适用于多分类问题，但是二分类的更为常用，也更加容易解释。实际中最为常用的就是二分类的logistic回归，常用于数据挖掘，疾病自动诊断，经济预测等领域。

用于二分类问题。其基本思想为：
a. 寻找合适的假设函数，即分类函数，用以预测输入数据的判断结果；
b. 构造代价函数，即损失函数，用以表示预测的输出结果与训练数据的实际类别之间的偏差；
c. 最小化代价函数，从而获取最优的模型参数。

## 分类问题 Classification

我们讨论的是要预测的变量y是一个离散情况下的分类问题。
分类问题中，我们尝试预测的是结果是否属于某一个类。分类问题的例子有：判断一封电子邮件是否是垃圾邮件; 判断一次金融交易是否是欺计；判断一个肿瘤是恶性的还是良性的。
我们预测的变量 y ∈ { 0 , 1 }，其中 0 表示负类 (Negative class)，1表示正类 (Positive class) 。
Logistic回归算法是一种分类算法，它适用于标签取值离散的情况，它的输出值永远在0到1之间。
不推荐将线性回归用于分类问题，线性回归模型的预测值可超越[0,1]范围。

## 假设表示 Hypothesis Representation

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Hypothesis%20Representation.png)

## 决策边界 Decision Boundary

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Decision%20Boundary.png)

## 代价函数 Cost Function

对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将ℎ𝜃(𝑥)带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数(non-convexfunction)，这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Logical%20Regression%20Cost%20Function.png)

## 梯度下降 Gradient Descent

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Logical%20Regression%20Gradient%20Descent.png)

## 高级优化 Advanced Optimization

一些更高级的优化算法有：共轭梯度法、BFGS 和L-BFGS 等。

优点：一个是通常不需要手动选择学习率，它们有一个智能内循环（线性搜索算法），可以自动尝试不同的学习速率α并自动选择一个好的学习速率，它们甚至可以为每次迭代选择不同的学习速率，那么我们就不需要自己选择。还有一个是它们经常快于梯度下降算法。

缺点：过于复杂。

## 多类别分类 Multiclass Classification

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Multiclass%20Classification.png)



# 正则化 Regularization

机器学习中的正则化是一种为了减小测试误差的行为。我们在搭建机器学习模型时，最终目的是让模型在面对新数据的时候，可以有很好的表现。当用比较复杂的模型（比如神经网络）去拟合数据时，很容易出现过拟合现象，这会导致模型的泛化能力下降，这时候我们就需要使用正则化技术去降低模型的复杂度，从而改变模型的拟合度。

## 过拟合的问题 The Problem of Overfitting

过拟合Overfit也可以叫做高方差high-variance，与之相反的概念是欠拟合underfit或高偏差high-bias

过拟合的问题就是指我们有非常多的特征，通过学习得到的模型能够非常好地适应训练集（代价函数可能几乎为0），但是推广到新的数据集上效果会非常的差。正则化可以改善或者减少过度拟合的问题。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/The%20Problem%20of%20Overfitting.png)

如果我们发现了过拟合问题，应该如何解决？
1.获取更多数据；
2.丢弃一些不能帮助我们正确预测的特征。可以是手工选择保留哪些特征，或者使用一些模型选择的算法来帮忙（例如PCA）；
3.正则化：留所有的特征，但是减少参数的大小。

## 代价函数 Cost Function

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Regularization%20Cost%20Function.png)

## 正则化线性回归 Regularized Linear Regression

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Regularized%20Linear%20Regression.png)

## 正则化逻辑回归 Regularized Logistic Regression

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Regularized%20Logistic%20Regression.png)



# 神经网络 Neural Networks

神经网络最初是一个生物学的概念，一般是指大脑神经元、触点、细胞等组成的网络，用于产生意识，帮助生物思考和行动，后来人工智能受神经网络的启发，发展出了人工神经网络。

人工神经网络（Artificial Neural Networks，简写为ANNs）也简称为神经网络（NNs）或称连接模型（Connection Model），它是一种模仿动物神经网络行为特征进行分布式并行信息处理的算法数学模型。这种网络依靠系统的复杂程度，通过调整内部大量节点之间相互连接的关系，从而达到处理信息的目的。神经网络的分支和演进算法很多种，从著名的卷积神经网络CNN，循环神经网络RNN，再到对抗神经网络GAN等等。

神经网络Neural Networks也被称为深度学习算法Deep Learning Algorithms或者决策树Decision Trees.

## 非线性假设 Non-linear Hypotheses

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Non-linear%20Hypotheses.png)

## 神经元和大脑 Neurons and the Brain

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Neurons%20and%20the%20Brain.png)

## 模型表示 Model Representation

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Model%20Representation%201.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Model%20Representation%202.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Model%20Representation%203.png)

## 代价函数 Cost Function

首先引入一些标记方法：假设神经网络的训练样本有m个，每个包含一组输入x和一组输出y，L表示神经网络层数，Sl表示第l层的单元数，即神经元的数量。神经网络的分类有两种情况：二元分类（Binary classification）和多类别分类（Multi-class classification）。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Neural%20Network%20Cost%20function%201.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Neural%20Network%20Cost%20function%202.png)



# 应用机器学习的建议 Advice for Applying Machine Learning

## 评估假设函数 Evaluating a Hypothesis

当我们确定学习算法的参数时，考虑的是选择参数来使训练误差最小化。有人认为得到一个非常小的训练误差一定是一件好事，但我们已经知道，仅仅因为这个假设函数具有很小的训练误差并不能说明它一定是一个好的假设函数。而且过拟合假设函数推广到新的训练集上是不适用的，所以仅靠具有很小的训练误差就说一个假设函数是好的假设函数这种说法是错误的。

那么，如何判断一个假设函数是否过拟合呢？
对于预测房价这个简单的例子，我们可以对假设函数进行绘图，然后观察图形趋势；但对于有很多特征变量的情况，想要通过画出假设函数的图形来进行观察，就会变得很难甚至不可能实现。 因此，我们需要另一种方法来评估我们的假设函数是否过拟合。
为了检验算法是否过拟合，我们将数据分成训练集和测试集，通常用70%的数据作为训练集，用剩下30%的数据作为测试集。很重要的一点是训练集和测试集均要含有各种类型的数据，通常我们要对数据进行“洗牌”，然后再分成训练集和测试集。所以说如果这组数据有某种规律或顺序的话，那么最好是随机选择70%的数据作为训练集，30%的数据作为测试集。

测试集评估在通过训练集让我们的模型学习得出其参数后，对测试集运用该模型，我们有两种方式计算误差：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Evaluating%20a%20Hypothesis.png)

## 模型选择和训练、验证、测试集 Model Selection and Training/Validation/Test Sets

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Training%20Validation%20Test%20Sets.png)

## 诊断偏差/方差 Diagnosing Bias/Variance

偏差(bias)大↔欠拟合(underfit)，方差(variance)大↔过拟合(overfit)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/high%20bias%20and%20variance.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Diagnosing%20Bias%20Variance.png)
对于训练集，当d较小时，模型拟合程度更低，误差较大；随着d的增长，拟合程度提高，误差减小。
对于交叉验证集，当d较小时，模型拟合程度低，误差较大；但是随着d的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始过拟合训练数据集的时候。

训练集误差和交叉验证集误差近似时：偏差/欠拟合
交叉验证集误差远大于训练集误差时：方差/过拟合

## 正则化和偏差/方差 Regularization and Bias/Variance

 ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Regularization%20and%20Bias%20Variance.png)

## 学习曲线 Learning Curves

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Learning%20Curves.png)

## 决定下一步如何改进 Deciding What to Do Next Revisited

假设我们已经使用正则化线性回归实现了模型的预测，但是当我们在一系列新的数据集上测试我们的假设函数时发现存在着很大的误差，改进的思路：
获得更多的训练样本——解决高方差
尝试减少特征的数量——解决高方差
尝试获得更多的特征——解决高偏差
尝试增加多项式特征——解决高偏差
尝试减少正则化程度λ——解决高偏差
尝试增加正则化程度λ——解决高方差

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Neural%20networks%20and%20overfitting.png)
使用较小的神经网络，类似于参数较少的情况，容易导致高偏差和欠拟合，但计算代价较小；使用较大的神经网络，类似于参数较多的情况，容易导致高方差和过拟合，虽然计算代价比较大，但是可以通过正则化手段来调整而更加适应数据。
通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果要好。
对于神经网络中的隐藏层的层数的选择，通常从一层开始逐渐增加层数，为了更好地作选择，可以把数据分为训练集、交叉验证集和测试集，针对不同隐藏层层数的神经网络训练神经网络， 然后选择交叉验证集代价最小的神经网络。

## 斜偏类的误差度量 Error Metrics for Skewed Classes

设定某个实数来评价我们的学习算法并衡量它的表现，有了算法的评估和误差度量值后，要注意的是使用一个合适的误差度量值有时会对于我们的学习算法造成非常微妙的影响，这就是偏斜类的问题。

在癌症分类例子中，我们训练Logistic回归模型（y = 1为癌症，y = 0为其他），假设使用测试集来检验这个分类模型，发现它只有1%的错误，因此我们99%会做出正确的诊断，这看起来是一个不错的结果。但假设我们发现在测试集中只有0.5%的患者真正患了癌症，那么1%的错误率就不显得那么好了，这种情况发生在训练集中有非常多的同一种类的样本且只有很少或没有其他类的样本，把这种情况称为偏斜类。

偏斜类：一个类中的样本数与另一个类的样本数相比多很多，通过总是预测y = 0或y = 1，算法可能表现得非常好，因此使用分类误差或者分类精确度来作为评估度量会产生问题。
如果我们有一个偏斜类，用分类精度并不能很好的衡量算法。因为我们可能会获得一个很高的精确度、非常低的错误率，但是我们并不知道我们是否真的提升了分类模型的质量。就像总是预测y = 0并不是一个好的分类模型，但是会将我们的误差降低至更低水平。所以当我们遇到偏斜类问题时，希望有一个不同的误差度量值或不同的评估度量值，例如查准率(Precision)和查全率(Recalll)。

查准率（Precision）和查全率（Recall） 我们将算法预测的结果分成四种情况：
1、正确肯定（True Positive,TP）：预测为真，实际为真
2、正确否定（True Negative,TN）：预测为假，实际为假
3、错误肯定（False Positive,FP）：预测为真，实际为假
4、错误否定（False Negative,FN）：预测为假，实际为真
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Error%20Metrics%20for%20Skewed%20Classes.png)

查准率=TP/(TP+FP)，即真正预测准确的数量 / 预测是准确的数量。例，在所有我们预测有恶性肿瘤的病人中，实际上有恶性肿瘤的病人的百分比，越高越好。
查全率=TP/(TP+FN)，即真正预测准确的数量 / 所有真正准确的数量。例，在所有实际上有恶性肿瘤的病人中，成功预测有恶性肿瘤的病人的百分比，越高越好。

高查准率和高查全率才可以表示一个模型是好模型
查准率和查全率是一对矛盾的指标，一般说，当查准率高的时候，查全率一般很低；查全率高时，查准率一般很低。

比如在西瓜书中的经典例子：若我们希望选出的西瓜中好瓜尽可能多，即查准率高，则只挑选最优把握的西瓜，算法挑选出来的西瓜（TP+FP）会减少，相对挑选出的西瓜确实是好瓜（TP）也相应减少，但是分母（TP+FP）减少的更快，所以查准率变大；在查全率公式中，分母（所有好瓜的总数）是不会变的，分子（TP）在减小，所以查全率变小。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Precision%20and%20Recall.png)



# 支持向量机 Support Vector Machines

可以学习的推文 https://mp.weixin.qq.com/s/171zIYlnxEoqpcZfJJRbGw

## 优化目标 Optimization Objective

与Logistic回归和神经网络相比，支持向量机（SVM）在学习复杂的非线性方程时提供了一种更为清晰、更加强大的方式。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Logistic.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/SVM.png)

## 大边界 Large Margin

有时人们会把支持向量机叫做大间距分类器（Large margin classifiers）
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Large%20Margin.png)

## 核函数 Kernels

对于下图的非线性数据集，可以通过构造一个复杂的多项式模型来解决无法用直线进行分隔的分类问题。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Kernels%201.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Kernels%202.png)

## 使用支持向量机 Using SVM

从Logistic回归模型，我们得到了支持向量机模型，在两者之间，我们应该如何选择呢？下面是一些普遍使用的准则，n为特征数，m为训练样本数。

如果n相较于m足够大的话，即训练集数据量不够支持我们训练一个复杂的非线性模型，我们选用Logistic回归模型或者不带核函数的支持向量机。
如果n较小，m大小中等，例如n在 1-1000 之间，而m在10-10000之间，使用高斯核函数的支持向量机。
如果n较小，而m较大，例如n在1-1000之间，而m大于50000，则使用支持向量机会非常慢，解决方案是手动地创建拥有更多的特征变量，然后使用Logistic回归或不带核函数的支持向量机。

值得一提的是，神经网络在以上三种情况下都可能会有较好的表现，但是训练神经网络可能非常慢，选择支持向量机的原因主要在于它的代价函数是凸函数，不存在局部最小值。

在遇到机器学习问题时，有时会不确定该用哪种算法，但是通常更加重要的是有多少数据，有多熟练，是否擅长做误差分析和排除学习算法，指出如何设定新的特征变量和找出其他能决定你学习算法的变量等方面，通常这些方面会比我们具体使用Logistic回归还是SVM算法更加重要。



# 决策树 Decision Tree

## 熵与信息增益 Entropy and Information Gain

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Entropy%20and%20Information%20Gain%201.jpg)

设属性A将S划分成m份，根据A划分的子集的熵或期望信息由下式给出：

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Entropy%20and%20Information%20Gain%202.jpg)

## Algorithm ID3

基本的 ID3 算法通过自顶向下构造决策树来进行学习。构造过程是从“哪一个属性将在树的根结点被测试？”这个问题开始的。然后为根结点属性的每个可能值产生一个分支，并把训练样例排列到适当的分支（也就是，样例的该属性值对应的分支）之下。然后重复整个过程，用每个分支结点关联的训练样例来选取在该点被测试的最佳属性。同时此贪婪搜索从不回溯重新考虑先前的选择。

故ID3算法主要围绕3个问题的解决来进行：如何选择最优属性、结点数据如何拆分、子树何时停止增长。

在为树节点选择测试属性时，需要选择最有助于分类实例的属性（也即特征）。ID3定义了一个统计属性“信息增益”，用来衡量给定属性区分当前训练样例集的能力，在其增长树的每一步使用该信息增益标准从侯选属性集中选择属性。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Algorithm%20ID3%201.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Algorithm%20ID3%202.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Algorithm%20ID3%20Procedure.png)

- 算法优点
  - ID3算法使用信息增益作为结点选择依据，从信息论的角度来进行学习，原理明晰、可解释性强
  - ID3算法操作简单，学习泛化能力强，考虑到ID3算法的归纳偏置：较短的树比较长的树优先，信息增益高的特征更靠近根节点的树优先，我们可以看到决策树所习得的规则是简单且易泛化到新数据的（符合奥卡姆剃刀原则）
  - 对于样例集中不充足属性的数据，可以有多种有效的方式进行填充，包括此次所用的“最通常值法”、“比例分配法”、“调换特征与目标属性角色法”等等。
  - ID3 算法在搜索的每一步都使用当前的所有训练样例，以统计为基础决定怎样精化当前的假设。这与那些基于单独的训练样例递增作出决定的方法（例如，Find-S或候选消除法）不同。使用所有样例的统计属性（例如，信息增益）的一个优点是大大减小了对个别训练样例错误的敏感性。
- 算法缺点
  - ID3算法只能处理分类属性的数据，不适宜连续类型的数据
  - 不能判断有多少个其他的决策树也是与现有的训练数据一致的
  - ID3算法很容易出现过度拟合训练数据的问题（特别是当训练数据集合小的时候）。因为训练样例仅仅是所有可能实例的一个样本，向树增加分支可能提高在训练样例上的性能，但却降低在训练实例外的其他实例上的性能。因此，通常需要后修剪决策树来防止过度拟合训练集，一般来说，这可以通过划分一个验证集来观测修剪。

## Algorithm C4.5

C4.5是一系列用在机器学习和数据挖掘的分类问题中的算法。它的目标是监督学习：给定一个数据集，其中的每一个元组都能用一组属性值来描述，每一个元组属于一个互斥的类别中的某一类。C4.5的目标是通过学习，找到一个从属性值到类别的映射关系，并且这个映射能用于对新的类别未知的实体进行分类。

- 改进表现

  - ID3算法在选择根节点和各内部节点中的分支属性时，采用信息增益作为评价标准。信息增益的缺点是倾向于选择取值较多的属性，而这类属性并不一定是最优的属性

    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Algorithm%20C4.5%201.png)

  - 在决策树构造过程中进行剪枝，因为某些具有很少元素的结点可能会使构造的决策树过适应（Overfitting），如果不考虑这些结点可能会更好

  - 能够处理离散型和连续型的属性类型，即将连续型的属性进行离散化处理

  - 能够处理具有缺失属性值的训练数据

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Algorithm%20C4.5%202.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Algorithm%20C4.5%203.png)



# 聚类 Clustering

## 无监督学习 Unsupervised Learning

在典型的监督学习中，我们有一个有标签的训练集，目标是找到能够区分正样本和负样本的决策边界。与此不同的是，在无监督学习中，我们需要将一系列无标签的训练数据输入到一个算法中，然后通过实现算法为我们找到训练数据的内在结构。将无标签数据集划分成一系列点集（称为簇）。能够划分这些点集的算法，就被称为聚类算法。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Unsupervised%20Learning.png)

## K-均值算法 K-Means Algorithm

K-均值算法是一种迭代求解的聚类分析算法，算法接受一个未标记的数据集，然后将数据聚类成不同的组。其步骤是，预将数据分为K组，则随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。

K-均值算法会做两件事：1. 簇分配；2. 移动聚类中心。

假设有一个无标签的数据集，想将其分为两个簇，执行K-均值算法。如下图所示，首先随机生成两点，这两点称为聚类中心，然后根距离移动聚类中心，直至中心点不再变化为止。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/K-means.png)

## 优化目标 Optimization Objective

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Optimization%20Objective.png)

## 随机初始化 Random Initialization

如何初始化K-均值聚类算法？如何使算法避开局部最优？

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Random%20Initialization.png)

## 选择聚类数 Choosing the Number of Clusters

最好的选择聚类数的方法，通常是根据不同的问题，人工进行选择，选择能最好服务于聚类目的的数量。

肘部法则（Elbow method）：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Choosing%20the%20Number%20of%20Clusters.png)

大部分时候，聚类数量K仍是通过手动、人工输入或者经验来决定，一种可以尝试的方法是使用“肘部原则”，但不能期望它每次都有效果。选择聚类数量更好的思路是去问自己运行K-均值聚类算法的目的是什么，然后再想聚类数目K取哪一个值能更好的服务于后续的目的。

# 降维 Dimensionality Reduction

## 数据压缩 Data Compression

数据压缩不仅能对数据进行压缩，使得数据占用较少的内存或硬盘空间，还能让我们对学习算法进行加速。

假使我们要采用两种不同的仪器来测量一些东西的尺寸，其中一个仪器测量结果x1的单位是厘米，另一个仪器测量的结果x2是英寸，我们希望将测量的结果作为我们机器学习的特征，如下图所示。现在的问题的是，两种仪器对同一个东西测量的结果不完全相等（由于误差、精度等原因），但将两者都作为特征又有些重复，因而我们希望将这个二维的数据降至一维来减少这种冗余。

## 数据可视化 Data Visualization

在许多机器学习问题中，如果我们能将数据可视化，便能寻找到一个更好的解决方案，降维可以帮助我们。

假设我们有关于许多不同国家的数据，每一个特征向量都有50个特征（如GDP、人均GDP、平均寿命等），如下表所示。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Data%20Visualization.png)

如果要将上面这个50维的数据直接进行可视化是不可能的，但使用降维的方法先将其降至2维，我们便可以将其可视化了。但这样做的问题在于，降维的算法只负责减少维数，新产生的特征的意义就必须由我们自己去发现了。

## 主成分分析问题规划 Principal Component Analysis Problem Formulation

主成分分析（Principal Component Analysis）是常用的降维算法。在PCA中，我们要做的是找到一个低维平面，当我们将所有数据都投影到该平面上时，希望投影误差（Projection error）能尽可能地小。在应用PCA之前，常规的做法是先进性均值归一化和特征归一化，使得特征量均值为0，并且其数值在可比较的范围之内。

如下图所示的数据，其在红线所代表的向量上的投影误差总的来说要比绿线所代表的向量上的投影误差要小得多，效果更好。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Principal%20Component%20Analysis%20Problem%20Formulation%201.png)
在本例将2维降至1维的情况中，PCA要做的是去找到一个数据投影后能够最小化投影误差的方向向量。对于将 n 维降至 k 维的情况，PCA要做的是找到 k 个方向向量来对数据进行投影来最小化投影误差。

主成分分析与线性回归是两种不同的算法。如下图所示，左边的是线性回归的误差（垂直于横轴投影），右边是主成分分析的误差（垂直于方向向量投影）。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Principal%20Component%20Analysis%20Problem%20Formulation%202.png)

主成分分析与线性回归的区别在于：
1、线性回归的误差距离是某个点与假设得到的预测值之间的距离；PCA的误差距离是某个点与方向向量之间的正交距离，即最短距离。
2、主成分分析最小化的是投影误差；线性回归尝试的是最小化预测误差。
3、线性回归的目的是预测结果，而主成分分析不作任何预测。

PCA将 n 个特征降维到 k 个，可以用来进行数据压缩，如果100维的向量最后可以用10维来表示，那么压缩率为90%。同样图像处理领域的KL变换使用PCA做图像压缩。但PCA要保证降维后，还要保证数据的特性损失最小。
PCA技术的一大好处是对数据进行降维的处理。我们可以对新求出的“主元”向量的重要性进行排序，根据需要取前面最重要的部分，将后面的维数省去，可以达到降维从而简化模型或是对数据进行压缩的效果，同时最大程度的保持了原有数据的信息。
PCA技术的一个很大的优点是完全无参数限制。在PCA的计算过程中完全不需要人为的设定参数或是根据任何经验模型对计算进行干预，最后的结果只与数据相关，与用户是独立的。但这一点同时也可以看作是缺点。如果用户对观测对象有一定的先验知识，掌握了数据的一些特征，却无法通过参数化等方法对处理过程进行干预，可能会得不到预期的效果，效率也不高。

## 主成分分析算法 Principal Component Analysis Algorithm

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Principal%20Component%20Analysis%20Algorithm.png)

## 重建压缩表示 Reconstruction from Compressed Representation

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Reconstruction%20from%20Compressed%20Representation.png)

## 主成分数量选择 Choosing The Number Of Principal Components

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Choosing%20The%20Number%20Of%20Principal%20Components.png)

## 主成分分析的应用建议 Advice for Applying PCA

采用PCA算法对监督学习算法进行加速：假使我们正在针对一张 100×100像素的图片进行某个计算机视觉的机器学习，即总共有10000 个特征。但因为数据量大，会使得学习算法运行的非常慢。采用PCA算法可以降低数据的维数从而使得算法运行更加高速。

PCA算法的应用：
1、压缩：减少存储数据所需的存储器或硬盘空间；加速学习算法。
2、可视化。

PCA算法的错误使用：
一个常见的错误使用PCA算法的情况是，将其用于防止过拟合（减少了特征的数量）。减少数据维度来防止过拟合的方法不是解决过拟合问题的好方法，不如尝试正则化处理。原因在于主成分分析只是近似地丢弃掉一些特征，它并不考虑任何与结果变量有关的信息，因此可能会丢失非常重要的特征。然而当我们进行正则化处理时，会考虑到结果变量，不会丢掉重要的数据。
另一个常见的错误是，默认地将主成分分析作为学习过程中的一部分，这虽然在很多时候都有效果，但最好还是从所有原始特征开始，只在有必要的时候（算法运行太慢或者占用太多内存）才考虑采用主成分分析。

# 异常检测 Anomaly Detection

异常检测是机器学习算法的一个常见应用，这种算法的一个有趣之处在于：它虽然主要用于非监督学习，但从某些角度来看，又类似于一些监督学习问题。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Anomaly%20Detection.png)

## 高斯分布 Gaussian Distribution

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Gaussian%20Distribution.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Gaussian%20Distribution%20Algorithm.png)

## 开发和评价异常检测系统 Developing and Evaluating an Anomaly Detection System

实数评估的重要性：当我们为某个应用开发一个学习算法时，需要进行一系列的选择（比如，选择特征等）。如果我们有某种方法，通过返回一个实数来评估我们的算法，那么对这些选择做出决定往往会更容易的多。

异常检测算法是一个非监督学习算法，意味着我们无法根据结果变量y的值来告诉我们数据是否真的是异常的。我们需要另一种方法来帮助检验算法是否有效。

当我们开发一个异常检测系统时，我们从带标记（异常或正常）的数据着手，从其中选择一部分正常数据用于构建训练集，然后用剩下的正常数据和异常数据混合的数据构成交叉检验集和测试集。

对异常检测系统具体的评价方法如下：
1、根据训练集数据，我们估计特征的平均值和方差并构建p(x)函数；
2、对交叉检验集，我们尝试使用不同的ε值作为阀值，并预测数据是否异常，根据F1值或者查准率与查全率的比例来选择ε.
3、选出ε后，针对测试集进行预测，计算异常检验系统的F1值，或者查准率与查全率之比。

## 异常检测与监督学习对比 Anomaly Detection vs Supervised Learning

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Anomaly%20Detection%20vs.%20Supervised%20Learning.png)

什么情况下，能让我们把某个学习问题当做是一个异常检测，或者是一个监督学习的问题？
对于一个学习问题，如果正样本的数量很少，甚至有时候是0，也就是说出现了太多没见过的不同的异常类型，那么对于这些问题，通常应该使用的算法就是异常检测算法；而如果正负样本数量都很多的话，则可以使用监督学习算法。例如，如果网络有很多诈骗用户，则可以变为监督学习；如果只有少量诈骗用户，则为异常检测。

## 选择特征 Choosing What Features to Use

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Choosing%20What%20Features%20to%20Use.png)

如何得到异常检测算法的特征？进行误差分析。

一个常见的问题是一些异常的数据可能也会有较高的p(x)值，因而被算法认为是正常的。这种情况下误差分析能够帮助我们，我们可以分析那些被算法错误预测为正常的数据，观察能否找出一些问题。我们可能能从问题中发现我们需要增加一些新的特征，增加这些新特征后获得的新算法能够帮助我们更好地进行异常检测。

我们通常可以通过将一些相关的特征进行组合，来获得一些新的更好的特征（异常数据的该特征值异常地大或小）。

## 多元高斯分布 Multivariate Gaussian Distribution

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Multivariate%20Gaussian%20Distribution%201.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Multivariate%20Gaussian%20Distribution%202.png)



# 推荐系统 Recommender Systems

## 基于内容的推荐系统 Content Based Recommendations

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Content%20Based%20Recommendations.png)

## 协同过滤 Collaborative Filtering

在之前的基于内容的推荐系统中，对于每一部电影, 我们都掌握了可用的特征，使用这些特征训练出了每一个用户的参数。相反地，如果我们拥有用户的参数，我们可以学习得出电影的特征。但是，如果当我们既没有用户的参数，也没有电影的特征，这两种方法就都不可行了。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Collaborative%20Filtering%201.png)

协同过滤算法指的是当我们执行算法时，要观察大量的用户，观察这些用户的实际行为来协调地得到对个用户对电影更佳的评分值。因为如果每个用户都对一部分电影做出了评价，那么每个用户都在帮助算法学习出更合适的特征，然后这些学习出的特征又可以被用来更好地预测其他用户的评分。协同的意思是每位用户都在帮助算法更好地进行特征学习，这就是协同过滤。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Collaborative%20Filtering%202.png)

## 均值归一化 Mean Normalization

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Mean%20Normalization.png)

## 基于内容的过滤 Content Based Filtering

Collaborative Filtering: 
Recommend items to you based on rating of users who gave similar ratings as you.

Content-based Filtering:
Recommend items to you based on features of user and item to find good match.



# 强化学习 Reinforcement Learning

## 强化学习的任务 The Mission of RL

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/RL_Introduce.png)

如图，人与环境交互的一种模型化表示，在每个时间点，大脑 agent 会从可以选择的动作集合A中选择一个动作 at 执行。环境则根据 agent 的动作给 agent 反馈一个 reward rt，同时 agent 进入一个新的状态。

知道了整个过程，任务的目标就出来了，那就是要能获取尽可能多的Reward。Reward越多，就表示执行得越好。每个时间片，agent 根据当前的状态来确定下一步的动作。也就是说我们需要一个state找出一个action，使得 reward 最大，从 state 到 action 的过程就称之为一个策略Policy，一般用 π 表示。

A policy is a function π(s): a mapping from states to actions, that tells you what action a to take in a given state s.
Find a policy π that tells you what action (a = π(s)) to take in every state (s) so as to maximize the return.

强化学习的任务就是找到一个最优的策略Policy从而使Reward最多。

我们一开始并不知道最优的策略是什么，因此往往从随机的策略开始，使用随机的策略进行试验，就可以得到一系列的状态样本，强化学习的算法就是需要根据这些样本来改进Policy，从而使得得到的样本中的Reward更好。由于这种让Reward越来越好的特性，所以这种算法就叫做强化学习Reinforcement Learning。

## 马尔可夫决策过程 Markov Decision Process

强化学习的问题都可以模型化为MDP(马尔可夫决策过程)的问题，MDP 实际上是对环境的建模；MDP 与常见的 Markov chains 的区别是加入了action 和 rewards 的概念。

一个基本的 MDP 可以用一个五元组 (S,A,P,R,γ) 表示，其中
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/MDP.png)

对于MDP，未来只取决于当前状态而不取决于任何事物。

因此，MDP 的核心问题就是找到一个策略 π(s) 来决定在状态 s 下选择哪个动作，这种情况下MDP就变成了一个 Markov chain，且此时的目标与强化学习的目标是一致的。

## 回报与价值函数 Return and Value Function

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Return%20and%20Value%20Function.png)

## Bellman Equation

采用上面获取最优策略的第 2 种方法时，我们需要估算 Value Function，只要能够计算出价值函数，那么最优决策也就得到了。因此，问题就变成了如何计算Value Function？

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Bellman%20Equation.png)



# The concept of Entropy

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
\begin{align}
& 离散情况：H(X)=-\sum_{i=1}^n p(x_i)log(p(x_i))\\
& 连续情况：H(x)=-\int^{+\infty}_{-\infty}{p(x)log(p(x))dx}\\
& 期望形式：H(x)=E_{x～p(x)}[-log(p(x))]
\end{align}
$$
如果式中的log以2为底的话，我们可以将这个式子解释为：要花费至少多少位的编码来表示此概率分布。从此式也可以看出，信息熵的本质是一种期望

然而有一类比较特殊的问题，比如投掷硬币只有两种可能，字朝上或花朝上。买彩票只有两种可能，中奖或不中奖，称之为0-1分布问题（二项分布的特例），对于这类问题，熵的计算方法可以简化为如下算式：

$H(X)=-\sum_{i=1}^n p(x_i)log(p(x_i))=-p(x)log(p(x))-[1-p(x)]log[1-p(x)]$

## Kullback-Leibler Divergence kL散度(相对熵)

如果对同一个随机变量 $x$ 有两个单独的概率分布 $p(x) 和 q(x)$，不妨将 $p(x)$ 看成是真实的分布，$q(x)$ 看成是估计的分布

KL散度，是指当估计分布 $q(x)$ 被用于近似真实 $p(x)$ 时的信息损失，也就是说，$q(x)$ 能在多大程度上表达 $p(x)$ 所包含的信息，KL散度越大，表达效果越差
$$
\begin{align}
& 离散情况：D_{KL}(p\;||\;q)=\sum_{i=1}^np(x_i)log(\frac{p(x_i)}{q(x_i)}) \\
& 连续情况：D_{KL}(p\;||\;q)=\int^{+\infty}_{-\infty}{p(x)log(\frac{p(x)}{q(x)})dx}\\
& 期望形式：E_{x～p(x)}[log(\frac{p(x)}{q(x)})]
\end{align}
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



# Loss functions and Probability theory

贝叶斯公式： $P(\theta|x)=\frac{P(x|\theta)P(\theta)}{P(x)})$

$P(\theta)$ 是先验概率，$P(x|\theta)$ 是似然概率，$P(\theta|x)$ 是后验概率

机器学习中，有频率派和贝叶斯派

- 频率派和贝叶斯派是两种不同的看待世界的方法论

  - 频率派把模型参数看成**未知的定量**，用极大似然法MLE（一种参数点估计方法）求解参数，往往最后变成**最优化**问题，这一分支又被称为统计学习

    极大似然法 MLE: $\theta=arg\,max\,log\,P(x|\theta)$

    补充：arg 是自变量 argument 的缩写。arg min 就是使后面这个式子达到最小值时的变量的取值；arg max 就是使后面这个式子达到最大值时的变量的取值

  - 贝叶斯派把模型参数看成**未知的变量（概率分布）**，用最大化后验概率MAP求解参数

    最大后验法 MAP: $\theta=arg\,max\,log\,P(\theta|x)=arg\,max\,log\,P(x|\theta)P(\theta)$

可以看到两者**最大的区别在于对参数的认知。**频率派认为参数是常量，数据是变量；贝叶斯派则认为参数是变量，不可能求出固定的参数，数据是常量。

## 最小二乘法

回归任务可以化为下式，其中 $y$ 是真实的连续值，$f^w(x)$ 是预测的连续值，$\varepsilon$  则是噪声

$y=f^w(x)+\varepsilon$，$f^w(x)=w^Tx$

我们假设噪声 $\varepsilon$ 符合正态分布，即 $\varepsilon\sim N(0,\delta^2)$

因此当我们给定 $w$ 和 $x$ 时，$y|w,x\sim N(w^Tx,\delta^2)$

得 $P(y|x,w)=\frac{1}{\sqrt{2\pi}\delta}exp(-\frac{(y-w^Tx)^2}{2\delta^2})$

下面就用频率派的思想，极大似然法MLE
$$
\begin{align}
L(w)  
& = log(Y|X,w) \\
& = log\prod(y_i|x_i,w) \\
& = \sum log(y_i|x_i,w) \\
& = \sum(log\frac{1}{\sqrt{2\pi}\delta}-\frac{(y_i-w^Tx_i)^2}{2\delta^2})
\end{align}
$$
那么我们要求 $w$，转换为最优化问题

$w=argmaxL(w)=argmax\sum-\frac{(y_i-w^Tx_i)^2}{2\delta^2}=argmin\sum(y_i-w^Tx_i)^2$

至此我们证明了，最小二乘法就是噪声符合正态分布的极大似然法的数学形式。从概率角度给出了最小二乘法的理论支撑。我们发现频率派，往往转换为极大似然法问题，也就是最优化求极值问题，这也被称为统计学习，像决策树，支持向量机都有最优化思想，都属于这一分支。

## 交叉熵

我们知道交叉熵用在分类任务上。以二分类为例，假设符合伯努利分布，则

$P(y|x)=g(x)^y\ast(1-g(x))^{1-y}$

$y$ 就是真实的类别，取值为0或1，$g(x)=sigmoid(w^Tx)$ 表示为1类的概率

用极大似然法

$L(w)=\sum logP(y_i|w,x_i)=\sum y_ilog(g(x_i))+(1-y_i)log(1-g(x_i))$

$w=argmaxL(w)=argmin\sum-y_ilog(g(x_i))-(1-y_i)log(1-g((x_i))$

这就是交叉熵的数学形式

## L2正则化、L1正则化

L2正则化，又被称为岭回归Ridge regression，是避免过拟合的有效手段

以回归任务为例： $y=f^w(x)+\varepsilon$，$f^w(x)=w^Tx$

我们假设噪声 $\varepsilon$ 符合正态分布，即 $\varepsilon\sim N(0,\delta_0^2)$

把 $x$ 看成常量，当给定 $w$ 时，$y|w\sim N(w^Tx,\delta_0^2)$

得 $P(y|w)=\frac{1}{\sqrt{2\pi}\delta_0}exp(-\frac{(y-w^Tx)^2}{2\delta_0^2})$

并且我们引入先验，假设参数 $w$ 符合正态分布，即 $w\sim N(0,\delta_1^2)$，因此

$P(w)=\frac{1}{\sqrt{2\pi}\delta_1}exp(-\frac{||w||^2}{2\delta_1^2})$

利用最大后验法MAP：
$$
\begin{align}
L(w)  
& = argmaxP(w|y) \\
& = argmax\frac{P(y|w)P(w)}{P(y)} \\
& = argmaxlogP(y|w)P(w) \\
& = argmaxlog(\frac{1}{\sqrt{2\pi}\delta_0}\frac{1}{\sqrt{2\pi}\delta_1})-\frac{(y-w^Tx)^2}{2\delta_0^2}-\frac{||w||^2}{2\delta_1^2} \\
& = argmin\frac{(y-w^Tx)^2}{2\delta_0^2}+\frac{||w||^2}{2\delta_1^2} \\
& = argmin(y-w^Tx)^2+\frac{\delta_0^2}{\delta_1^2}||w||^2
\end{align}
$$
一顿操作后，发现L2正则化就是假设参数符合正态分布的最大后验法的数学形式

同理可得L1正则化是假设参数符合拉普拉斯分布的最大后验法。

我们现在可以从概率角度解释正则化到底在干什么了。正则化就是引入了先验知识，我们知道世界上大多数事件是服从正态分布的，像身高、体重、成绩等等。因此我们假设参数也符合正态分布。引入先验知识有什么好处呢，我们现在抛一枚硬币，50次中有30次都是正面向上，问你抛这枚硬币的概率分布，这时你想起你人生中遇到的大多数硬币都是均匀的，尽管数据显示不均匀，你还是会认为这枚硬币是均匀的。如果你是这么想的，那你就引入了先验知识。因此引入先验知识在数据不足的时候有很大好处。

- 频率派认为模型参数是客观存在的，它就在那里。因此把参数看成常量，如果有一个全知全能神，就能告诉你参数值是多少，当数据量成千上万时，我们可以不断逼近那个真实的参数。
- 贝叶斯派认为认为一切概率都是主观的，因此把参数看成变量，不存在客观存在的概率。
