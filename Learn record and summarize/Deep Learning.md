# 简介 Introduce

神经网络最初是一个生物学的概念，一般是指大脑神经元、触点、细胞等组成的网络，用于产生意识，帮助生物思考和行动，后来人工智能受神经网络的启发，发展出了人工神经网络。

人工神经网络（Artificial Neural Networks，简写为ANNs）也简称为神经网络（NNs）或称连接模型（Connection Model），它是一种模仿动物神经网络行为特征进行分布式并行信息处理的算法数学模型。这种网络依靠系统的复杂程度，通过调整内部大量节点之间相互连接的关系，从而达到处理信息的目的。神经网络的分支和演进算法很多种，从著名的卷积神经网络CNN，循环神经网络RNN，再到对抗神经网络GAN等等。

神经网络Neural Networks也被称为深度学习算法Deep Learning Algorithms或者决策树Decision Trees.

目前为止，由神经网络模型创造的价值基本上都是基于监督学习（Supervised Learning）的。监督学习与非监督学习本质区别就是是否已知训练样本的输出y。在实际应用中，机器学习解决的大部分问题都属于监督式学习，神经网络模型也大都属于监督式学习。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Supervised%20Learning.png)

对于一般的监督式学习（房价预测和线上广告问题），我们只要使用标准的神经网络模型就可以了。而对于图像识别处理问题，我们则要使用卷积神经网络（Convolution Neural Network），即CNN。而对于处理类似语音这样的序列信号时，则要使用循环神经网络（Recurrent Neural Network），即RNN。还有其它的例如自动驾驶这样的复杂问题则需要更加复杂的混合神经网络模型。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/NN%20Modal.png)

数据类型一般分为两种：Structured Data和Unstructured Data。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Structured%20Unstructured%20Data.png)
Structured Data通常指的是有实际意义的数据，例如房价预测中的size，#bedrooms，price等，在线广告中的User Age，Ad ID等，这些数据都具有实际的物理意义，比较容易理解。而Unstructured Data通常指的是比较抽象的数据，例如Audio，Image或者Text。以前，计算机对于Unstructured Data比较难以处理，而人类对Unstructured Data却能够处理的比较好，例如我们第一眼很容易就识别出一张图片里是否有猫，但对于计算机来说并不那么简单。现在，由于深度学习和神经网络的发展，计算机在处理Unstructured Data方面效果越来越好，甚至在某些方面优于人类。总的来说，神经网络与深度学习无论对Structured Data还是Unstructured Data都能处理得越来越好，并逐渐创造出巨大的实用价值。

构建一个深度学习的流程是首先产生Idea，然后将Idea转化为Code，最后进行Experiment。接着根据结果修改Idea，继续这种Idea->Code->Experiment的循环，直到最终训练得到表现不错的深度学习网络模型。



# 神经网络基础知识 Basics Knowledge of Neural Network

## 逻辑回归 Logistic Regression

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Logistic%20Regression.png)

## 逻辑回归代价函数 Logistic Regression Cost Function

逻辑回归中，w和b都是未知参数，需要反复训练优化得到。因此，我们需要定义一个cost function，包含了参数w和b。通过优化cost function，当cost function取值最小时，得到对应的w和b。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Logistic%20Regression%20Cost%20Function%201.png)

Loss function是衡量错误大小的，它越小越好
当y=1时，如果越接近1，表示预测效果越好；如果越接近0，表示预测效果越差。
当y=0时，如果越接近0，表示预测效果越好；如果越接近1，表示预测效果越差。

Loss function是针对单个样本的。那对于m个样本，我们定义Cost function，Cost function是m个样本的Loss function的平均值，反映了m个样本的预测输出与真实样本输出y的平均接近程度。

Cost function可表示为：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Logistic%20Regression%20Cost%20Function%202.png)

Cost function是关于待求系数w和b的函数。我们的目标就是迭代计算出最佳的w和b值，最小化Cost function，让Cost function尽可能地接近于零。

逻辑回归问题可以看成是一个简单的神经网络，只包含一个神经元。

**Detailed explanation of logistic regression cost function**

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Explanation%20of%20logistic%20regression%20cost%20function.png)

## 梯度下降 Gradient Descent

由于J(w,b)是convex function，梯度下降算法是先随机选择一组参数w和b值，然后每次迭代的过程中分别沿着w和b的梯度（偏导数）的反方向前进一小步，不断修正w和b。每次迭代更新w和b后，都能让J(w,b)更接近全局最小值。

梯度下降的过程如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Gradient%20Descent.png)

α是学习因子（learning rate），表示梯度下降的步长。它越大，w和b每次更新的“步伐”更大一些；它越小，w和b每次更新的“步伐”更小一些。

## 计算图 Computation Graph

整个神经网络的训练过程实际上包含了两个过程：正向传播（Forward Propagation）和反向传播（Back Propagation）。正向传播是从输入到输出，由神经网络计算得到预测输出的过程；反向传播是从输出到输入，对参数w和b计算梯度的过程。下面，我们用计算图（Computation graph）的形式来理解这两个过程。

假如Cost function为J(a,b,c)=3(a+bc)，包含a,b,c三个变量。我们用u表示bc，v表示a+u，则J=3v。它的计算图可以写成如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Computation%20Graph%201.png)
令a=5，b=3，c=2，则u=bc=6，v=a+u=11，J=3v=33。计算图中，这种从左到右，从输入到输出的过程就对应着神经网络或者逻辑回归中输入与权重经过运算计算得到Cost function的正向过程。

反向传播（Back Propagation），即计算输出对输入的偏导数。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Computation%20Graph%202.png)

## 逻辑回归中的梯度下降 Logistic Regression Gradient Descent

对单个样本而言，逻辑回归Loss function表达式如下:
$$
Z=w^Tx+b
$$

$$
\widehat{y}=a=\sigma(z)
$$

$$
L(a,y)=-(ylog(a)+(1-y)log(1-a))
$$

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Logistic%20Regression%20Gradient%20Descent%202.png)

计算该逻辑回归的反向传播过程，即由Loss function计算参数w和b的偏导数。推导过程如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Logistic%20Regression%20Gradient%20Descent%203.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Logistic%20Regression%20Gradient%20Descent%204.png)

如果有m个样本，其Cost function表达式如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Gradient%20descent%20on%20m%20examples.png)

这样，每次迭代中w和b的梯度有m个训练样本计算平均值得到。其算法流程如下所示：

```python
J=0; dw1=0; dw2=0; db=0;
for i = 1 to m
    z(i) = wx(i)+b;
    a(i) = sigmoid(z(i));
    J += -[y(i)log(a(i))+(1-y(i)）log(1-a(i));
    dz(i) = a(i)-y(i);
    dw1 += x1(i)dz(i);
    dw2 += x2(i)dz(i);
    db += dz(i);
J /= m;
dw1 /= m;
dw2 /= m;
db /= m;
```

经过每次迭代后，根据梯度下降算法，w和b都进行更新，这样经过n次迭代后，整个梯度下降算法就完成了。

在上述的梯度下降算法中，利用for循环对每个样本进行dw1，dw2和db的累加计算最后再求平均数的。在深度学习中，样本数量m通常很大，使用for循环会让神经网络程序运行得很慢。所以，我们应该尽量避免使用for循环操作，而使用矩阵运算，能够大大提高程序运行速度。

## 向量化 Vectorization

深度学习算法中，数据量很大，在程序中应该尽量减少使用loop循环语句，而可以使用向量运算来提高程序运行速度。
向量化（Vectorization）就是利用矩阵运算的思想，大大提高运算速度。

为了加快深度学习神经网络运算速度，可以使用比CPU运算能力更强大的GPU。事实上，GPU和CPU都有并行指令（parallelization instructions），称为Single Instruction Multiple Data（SIMD）。SIMD是单指令多数据流，能够复制多个操作数，并把它们打包在大型寄存器的一组指令集。SIMD能够大大提高程序运行速度，例如python的numpy库中的内建函数（built-in function）就是使用了SIMD指令。相比而言，GPU的SIMD要比CPU更强大一些。

在python的numpy库中，我们通常使用np.dot()函数来进行矩阵运算。
我们将向量化的思想使用在逻辑回归算法上，尽可能减少for循环，而只使用矩阵运算。值得注意的是，算法最顶层的迭代训练的for循环是不能替换的，而每次迭代过程对J，dw，b的计算是可以直接使用矩阵运算。

整个训练样本构成的输入矩阵X的维度是（，m），权重矩阵w的维度是（，1），b是一个常数值，而整个训练样本构成的输出矩阵Y的维度为（1，m）。利用向量化的思想，所有m个样本的线性输出Z可以用矩阵表示：
$$
Z=w^TX+b
$$
在python的numpy库中可以表示为:

```python
Z = np.dot(w.T,X) + b
A = sigmoid(Z)
```

其中，w.T表示w的转置。这里在 Python 中有一个巧妙的地方，这里b是一个实数，或者你可以说是一个 1 × 1 矩阵，只是一个普通的实数。但是当你将这个向量加上这个实数时，Python 自动把这个实数b扩展成一个 1 × m 的行向量。所以这种情况下的操作似乎有点不可思议，它在Python中被称作广播(brosdcasting)。
这样，我们就能够使用向量化矩阵运算代替for循环，对所有m个样本同时运算，大大提高了运算速度。

## 向量化逻辑回归的梯度输出 Vectorizing Logistic Regression’s Gradient Output

对于所有m个样本，dZ的维度是（1，m），可表示为：dZ=A−Y
db可表示为：
$$
db=\frac{1}{m}\sum_{i=1}^mdz^{(i)}
$$
对应的程序为

```Python
db = 1/m*np.sum(dZ)
```

dw可表示为：
$$
dw=\frac{1}{m}X·dZ^T
$$
对应的程序为：

```python
dw = 1/m*np.dot(X,dZ.T)
```

这样，我们把整个逻辑回归中的for循环尽可能用矩阵运算代替，对于单次迭代，梯度下降算法流程如下所示：

```python
Z = np.dot(w.T,X) + b
A = sigmoid(Z)
dZ = A-Y
dw = 1/m*np.dot(X,dZ.T)
db = 1/m*np.sum(dZ)

w = w - alpha*dw
b = b - alpha*db
```

其中，alpha是学习因子，决定w和b的更新速度。上述代码只是对单次训练更新而言的，外层还需要一个for循环，表示迭代次数。

## Broadcasting in Python 

- python中的广播机制可由下面四条表示：
  - 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐
  - 输出数组的shape是输入数组shape的各个轴上的最大值
  - 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
  - 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值
- 简而言之，就是python中可以对不同维度的矩阵进行四则混合运算，但至少保证有一个维度是相同的。
- 在python程序中为了保证矩阵运算正确，可以使用reshape()函数来对矩阵设定所需的维度

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Broadcasting%20example.png)

# 浅层神经网络 Shallow neural networks

## 神经网络概述 Neural Networks Overview

神经网络的结构与逻辑回归类似，只是神经网络的层数比逻辑回归多一层，多出来的中间那层称为隐藏层或中间层。这样从计算上来说，神经网络的正向传播和反向传播过程只是比逻辑回归多了一次重复的计算。

正向传播过程分成两层，第一层是输入层到隐藏层，用上标[1]来表示：
$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$

$$
a^{[1]}=σ(z^{[1]})
$$

第二层是隐藏层到输出层，用上标[2]来表示：
$$
z^{[2]}=W^{[2]}x+b^{[2]}
$$

$$
a^{[2]}=σ(z^{[2]})
$$

在写法上值得注意的是，方括号上标[i]表示当前所处的层数；圆括号上标(i)表示第i个样本。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Networks%20Overview.png)

# 深层神经网络 Deep neural networks

