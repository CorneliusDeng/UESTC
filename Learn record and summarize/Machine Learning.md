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

## 商和信息增益 Entropy and Information Gain

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

## K-均值算法 K-Means Algorithm

## 优化目标 Optimization Objective

## 随机初始化 Random Initialization

## 选择聚类数 Choosing the Number of Clusters







# 降维 Dimensionality Reduction







# 异常检测 Anomaly Detection







# 推荐系统 Recommender Systems

