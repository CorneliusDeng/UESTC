# 简介

- **机器学习定义**
  - Arthur Samuel(1959): Field of study that gives computers the ability to learn without being explicitly programmed. 在没有明确设置的情况下，使计算机具有学习能力的研究领域。
  
  - Tom Mitchell(1998): Well-posed Learning Problem:A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 计算机程序从经验E中学习解决某一任务T，进行某一性能度量P，通过P测定在T上的表现因经验E而提高。例如，在人机玩跳棋游戏中，经验E是程序与自己下几万次跳棋；任务T是玩跳棋；性能度量P是与新对手玩跳棋时赢的概率。
  
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

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Training%3AValidation%3ATest%20Sets.png)

## 诊断偏差/方差 Diagnosing Bias/Variance

偏差(bias)大↔欠拟合(underfit)，方差(variance)大↔过拟合(overfit)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/high%20bias%20and%20variance.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Diagnosing%20Bias%20Variance.png)
对于训练集，当d较小时，模型拟合程度更低，误差较大；随着d的增长，拟合程度提高，误差减小。
对于交叉验证集，当d较小时，模型拟合程度低，误差较大；但是随着d的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始过拟合训练数据集的时候。

训练集误差和交叉验证集误差近似时：偏差/欠拟合
交叉验证集误差远大于训练集误差时：方差/过拟合

## 正则化和偏差/方差 Regularization and Bias/Variance

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Regularization%20and%20Bias%3AVariance.png) 

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







