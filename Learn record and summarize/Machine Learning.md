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



