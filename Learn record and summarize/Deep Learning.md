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

## 神经网络的表示 Neural Network Representation

如下图所示，单隐藏层神经网络就是典型的浅层（shallow）神经网络
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Network%20Representation%201.png)

结构上，从左到右，可以分成三层：输入层（Input layer），隐藏层（Hidden layer）和输出层（Output layer）。输入层和输出层，顾名思义，对应着训练样本的输入和输出，隐藏层是抽象的非线性的中间层，这也是其被命名为隐藏层的原因。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Network%20Representation%202.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Network%20Representation%203.png)

最后输出层将产生某个数值𝑎，它只是一个单独的实数，所以的y^值将取为𝑎^[2]。这与逻辑回归很相似，在逻辑回归中，我们有𝑦^直接等于𝑎，在逻辑回归中我们只有一个输出层，所以我们没有用带方括号的上标。但是在神经网络中，我们将使用这种带上标的形式来明
确地指出这些值来自于哪一层，有趣的是在约定俗成的符号传统中，上图所示例子，只能叫做一个两层的神经网络。原因是当我们计算网络的层数时，输入层是不算入总层数内，所以隐藏层是第一层，输出层是第二层。第二个惯例是我们将输入层称为第零层，所以在技术上，这仍然是一个三层的神经网络，因为这里有输入层、隐藏层，还有输出层。但是在传统的符号使用中，人们将这个神经网络称为一个两层的神经网络，因为输入层不被看作一个标准的层。

关于隐藏层对应的权重和常数项，w的维度是(4,3)，这里的4对应着隐藏层神经元个数，3对应着输入层x特征向量包含元素个数。常数项的维度是(4,1)，这里的4同样对应着隐藏层神经元个数。关于输出层对应的权重和常数项，w的维度是(1,4)，这里的1对应着输出层神经元个数，4对应着输出层神经元个数。常数项的维度是(1,1)，因为输出只有一个神经元。
**总结：第i层的权重维度的行等于i层神经元的个数，列等于i-1层神经元的个数；第i层常数项维度的行等于i层神经元的个数，列始终为1.**

## 计算神经网络的输出 Computing Neural Network's output

关于神经网络是怎么计算的，从逻辑回归开始，如下图所示，用圆圈表示神经网络的计算单元，逻辑回归的计算有两个步骤，首先按步骤计算出𝑧，然后在第二步中以 sigmoid 函数为激活函数计算𝑧（得出𝑎），一个神经网络只是这样子做了好多次重复计算。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Computing%20a%20Neural%20Network%E2%80%99s%20Output%201.png)

对于两层神经网络，从输入层到隐藏层对应一次逻辑回归运算，从隐藏层到输出层也对应一次逻辑回归运算。每层计算时，要注意对应的上标和下标，一般我们记上标方括号表示layer，下标表示第几个神经元。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Computing%20a%20Neural%20Network%E2%80%99s%20Output%202.png)

为了提高程序运算速度，我们引入向量化和矩阵运算的思想，将上述表达式转换成矩阵运算的形式
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Computing%20a%20Neural%20Network%E2%80%99s%20Output%203.png)

逻辑回归是将各个训练样本组合成矩阵，对矩阵的各列进行计算。神经网络是通过对逻辑回归中的等式简单的变形，让神经网络计算出输出值，这种计算是所有的训练样本同时进行的。如果有多个训练样本，不过是对单个样本计算的重复。

## 激活函数 Activation Function

神经网络隐藏层和输出层都需要激活函数(activation function)，除了使用Sigmoid函数作为激活函数，还有其它激活函数可供使用，不同的激活函数有各自的优点。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Activation%20Function.png)

不同激活函数形状不同，a的取值范围也有差异。
首先我们来比较sigmoid函数和tanh函数。对于隐藏层的激活函数，一般来说，tanh函数要比sigmoid函数表现更好一些。因为tanh函数的取值范围在[-1,+1]之间，隐藏层的输出被限定在[-1,+1]之间，可以看成是在0值附近分布，均值为0。这样从隐藏层到输出层，数据起到了归一化（均值为0）的效果。因此，隐藏层的激活函数，tanh比sigmoid更好一些。而对于输出层的激活函数，因为二分类问题的输出取值为{0,+1}，所以一般会选择sigmoid作为激活函数。观察sigmoid函数和tanh函数，我们发现有这样一个问题，就是当|z|很大的时候，激活函数的斜率（梯度）很小。因此，在这个区域内，梯度下降算法会运行得比较慢。在实际应用中，应尽量避免使z落在这个区域，使|z|尽可能限定在零值附近，从而提高梯度下降算法运算速度。
为了弥补sigmoid函数和tanh函数的这个缺陷，就出现了ReLU激活函数。ReLU激活函数在z大于零时梯度始终为1；在z小于零时梯度始终为0；z等于零时的梯度可以当成1也可以当成0，实际应用中并不影响。对于隐藏层，选择ReLU作为激活函数能够保证z大于零时梯度始终为1，从而提高神经网络梯度下降算法运算速度。但当z小于零时，存在梯度为0的缺点，实际应用中，这个缺点影响不是很大。为了弥补这个缺点，出现了Leaky ReLU激活函数，能够保证z小于零是梯度不为0。

最后总结一下，如果是分类问题，输出层的激活函数一般会选择sigmoid函数。但是隐藏层的激活函数通常不会选择sigmoid函数，tanh函数的表现会比sigmoid函数好一些。实际应用中，通常会会选择使用ReLU或者Leaky ReLU函数，保证梯度下降速度不会太小。
其实，具体选择哪个函数作为激活函数没有一个固定的准确的答案，应该要根据具体实际问题进行验证（validation）。

上述的四种激活函数都是非线性(non-linear)的。那是否可以使用线性激活函数呢？答案是不行！
如果使用线性的激活函数，经过推导我们发现输出结果仍是输入变量x的线性组合。这表明，使用神经网络与直接使用线性模型的效果并没有什么两样。即便是包含多层隐藏层的神经网络，如果使用线性函数作为激活函数，最终的输出仍然是输入x的线性模型。这样的话神经网络就没有任何作用了。因此，隐藏层的激活函数必须要是非线性的。
另外，如果所有的隐藏层全部使用线性激活函数，只有输出层使用非线性激活函数，那么整个神经网络的结构就类似于一个简单的逻辑回归模型，而失去了神经网络模型本身的优势和价值。
值得一提的是，如果是预测问题而不是分类问题，输出y是连续的情况下，输出层的激活函数可以使用线性函数。如果输出y恒为正值，则也可以使用ReLU激活函数，具体情况，具体分析。

## 激活函数的导数 Derivatives of activation functions

在梯度下降反向计算过程中少不了计算激活函数的导数即梯度。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Derivatives%20of%20activation%20functions.png)

## 神经网络的梯度下降 Gradient descent for neural networks

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Gradient%20descent%20for%20neural%20networks.png)

上述是反向传播的步骤，注：这些都是针对所有样本进行过向量化，Y是 1 ∗ m 的矩阵；这里np.sum是python的numpy命令，axis=1表示水平相加求和，keepdims是防止python输出那些古怪的秩数 (n , )  ，加上这个确保矩阵db^[2] 这个向量输出的维度为(n,1) 这样标准的形式。
还有一种防止python输出奇怪的秩数，需要显式地调用reshape把np.sum输出结果写成矩阵形式。

## 对反向传播的理解 Backpropagation intuition

在逻辑回归中，引入了计算图来推导正向传播和反向传播，其过程如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Backpropagation%201.png)

由于多了一个隐藏层，神经网络的计算图要比逻辑回归的复杂一些。对于单个训练样本，正向过程很容易，反向过程可以根据梯度计算方法逐一推导。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Backpropagation%202.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Backpropagation%203.png)

浅层神经网络（包含一个隐藏层），m个训练样本的正向传播过程和反向传播过程分别包含了6个表达式，其向量化矩阵形式如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Backpropagation%204.png)

##  随机初始化 Random Initialization

对于逻辑回归，把权重初始化为0 当然也是可以的。但是对于一个神经网络，如果把权重或者参数都初始化为 0，那么梯度下降将不会起作用。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Random%20Initialization.png)

这样使得隐藏层第一个神经元的输出等于第二个神经元的输出。因此，这样的结果是隐藏层两个神经元对应的权重行向量和每次迭代更新都会得到完全相同的结果，始终等于，完全对称。这样隐藏层设置多个神经元就没有任何意义了。值得一提的是，参数b可以全部初始化为零，并不会影响神经网络训练效果。

我们把这种权重W全部初始化为零带来的问题称为symmetry breaking problem。解决方法也很简单，就是将W进行随机初始化（b可初始化为零）。python里可以使用如下语句进行W和b的初始化：

```python
W_1 = np.random.randn((2,2))*0.01
b_1 = np.zero((2,1))
W_2 = np.random.randn((1,2))*0.01
b_2 = 0
```

这里我们将W_1和W_2乘以0.01的目的是尽量使得权重W初始化比较小的值。之所以让W比较小，是因为如果使用sigmoid函数或者tanh函数作为激活函数的话，W比较小，得到的|z|也比较小（靠近零点），而零点区域的梯度比较大，这样能大大提高梯度下降算法的更新速度，尽快找到全局最优解。如果W较大，得到的|z|也比较大，附近曲线平缓，梯度较小，训练过程会慢很多。

当然，如果激活函数是ReLU或者Leaky ReLU函数，则不需要考虑这个问题。但是，如果输出层是sigmoid函数，则对应的权重W最好初始化到比较小的值



# 深层神经网络 Deep neural networks

## Deep L-layer neural network

深层神经网络其实就是包含更多的隐藏层神经网络。如下图所示，分别列举了逻辑回归、1个隐藏层的神经网络、2个隐藏层的神经网络和5个隐藏层的神经网络它们的模型结构。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Deep%20L-layer%20neural%20network%201.png)

命名规则上，一般只参考隐藏层个数和输出层。例如，上图中的逻辑回归又叫1 layer NN，1个隐藏层的神经网络叫做2 layer NN，2个隐藏层的神经网络叫做3 layer NN，以此类推。如果是L-layer NN，则包含了L-1个隐藏层，最后的L层是输出层。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Deep%20L-layer%20neural%20network%202.png)

## 深层网络中的前向传播 Forward propagation in a Deep Network

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Forward%20propagation%20in%20a%20Deep%20Network%200.png)

对于单个样本，推导深层神经网络的正向传播过程：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Forward%20propagation%20in%20a%20Deep%20Network%201.png)

如果有m个训练样本，其向量化矩阵形式为：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Forward%20propagation%20in%20a%20Deep%20Network%202.png)

## 矩阵维数 Matrix Dimensions

当实现深度神经网络的时候，检查代码是否有错的方法之一是计算一遍算法中矩阵的维数

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/matrix%20dimensions%201.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/matrix%20dimensions%202.png)

## 搭建神经网络块 Building blocks of deep neural networks

用流程块图来解释神经网络正向传播和反向传播过程。如下图所示，对于第l层来说，正向传播过程中存在输入、输出、参数、缓存变量，反向传播过程中存在输入、输出、参数
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Building%20blocks%20of%20deep%20neural%20networks%201.png)

对于神经网络所有层，整体的流程块图正向传播过程和反向传播过程如下所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Building%20blocks%20of%20deep%20neural%20networks%202.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/The%20connection%20between%20neural%20network%20and%20brain.jpg)

## 参数和超参数 Parameters and Hyperparameters

想要你的深度神经网络起很好的效果，需要规划好参数和超参数

超参数是例如学习速率，训练迭代次数N，隐藏层数L，隐藏层神经元个数，激活函数等，之所以叫做超参数的原因是这些数字实际上控制了最后的参数w和b的值

如何设置最优的超参数是一个比较困难的、需要经验知识的问题。通常的做法是选择超参数一定范围内的值，分别代入神经网络进行训练，测试cost function随着迭代次数增加的变化，根据结果选择cost function最小时对应的超参数值



# 深度学习的实践层面 Practical aspects of Deep Learning

## 训练/验证/测试集 Train / Dev / Test sets

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Train%20Dev%20Test%20sets.png)

在构建一个神经网络的时候，我们需要设置许多参数，例如神经网络的层数、每个隐藏层包含的神经元个数、学习因子（学习速率）、激活函数的选择等等。实际上很难在第一次设置的时候就选择到这些最佳的参数，而是需要通过不断地迭代更新来获得。这个循环迭代的过程是这样的：我们先有个想法Idea，先选择初始的参数值，构建神经网络模型结构；然后通过代码Code的形式，实现这个神经网络；最后，通过实验Experiment验证这些参数对应的神经网络的表现性能。根据验证结果，我们对参数进行适当的调整优化，再进行下一次的Idea->Code->Experiment循环。通过很多次的循环，不断调整参数，选定最佳的参数值，从而让神经网络性能最优化。

应用深度学习是一个反复迭代的过程，需要通过反复多次的循环训练得到最优化参数。决定整个训练过程快慢的关键在于单次循环所花费的时间，单次循环越快，训练过程越快。而设置合适的Train/Dev/Test sets数量，能有效提高训练效率。

选择最佳的训练集（Training sets）、验证集（Development sets）、测试集（Test sets）对神经网络的性能影响非常重要。一般地，我们将所有的样本数据分成三个部分：Train/Dev/Test sets。Train sets用来训练算法模型；Dev sets用来验证不同算法的表现情况，从中选择最好的算法模型；Test sets用来测试最好算法的实际表现，作为该算法的无偏估计。

之前人们通常设置Train sets和Test sets的数量比例为70%和30%。如果有Dev sets，则设置比例为60%、20%、20%，分别对应Train/Dev/Test sets。这种比例分配在样本数量不是很大的情况下，例如100,1000,10000，是比较科学的。但是如果数据量很大的时候，例如100万，这种比例分配就不太合适了。科学的做法是要将Dev sets和Test sets的比例设置得很低。因为Dev sets的目标是用来比较验证不同算法的优劣，从而选择更好的算法模型就行了。因此，通常不需要所有样本的20%这么多的数据来进行验证。对于100万的样本，往往只需要10000个样本来做验证就够了。Test sets也是一样，目标是测试已选算法的实际表现，无偏估计。对于100万的样本，往往也只需要10000个样本就够了。因此，对于大数据样本，Train/Dev/Test sets的比例通常可以设置为98%/1%/1%，或者99%/0.5%/0.5%。样本数据量越大，相应的Dev/Test sets的比例可以设置的越低一些。

最后提一点的是如果没有Test sets也是没有问题的。Test sets的目标主要是进行无偏估计。我们可以通过Train sets训练不同的算法模型，然后分别在Dev sets上进行验证，根据结果选择最好的算法模型。这样也是可以的，不需要再进行无偏估计了。如果只有Train sets和Dev sets，通常也有人把这里的Dev sets称为Test sets，我们要注意加以区别

## 偏差和方差 Bias and Variance

偏差（Bias）和方差（Variance）是机器学习领域非常重要的两个概念和需要解决的问题。在传统的机器学习算法中，Bias和Variance是对立的，分别对应着欠拟合和过拟合，我们常常需要在Bias和Variance之间进行权衡。而在深度学习中，我们可以同时减小Bias和Variance，构建最佳神经网络模型。

如下图所示，显示了二维平面上，high bias，just right，high variance的例子。可见，high bias对应着欠拟合，而high variance对应着过拟合。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Bias%20and%20Variance%201.png)

上图这个例子中输入特征是二维的，high bias和high variance可以直接从图中分类线看出来。而对于输入特征是高维的情况，如何来判断是否出现了high bias或者high variance呢？

例如猫识别问题，输入是一幅图像，其特征维度很大。这种情况下，我们可以通过两个数值Train set error和Dev set error来理解bias和variance。假设Train set error为1%，而Dev set error为11%，即该算法模型对训练样本的识别很好，但是对验证集的识别却不太好。这说明了该模型对训练样本可能存在过拟合，模型泛化能力不强，导致验证集识别率低。这恰恰是high variance的表现。假设Train set error为15%，而Dev set error为16%，虽然二者error接近，即该算法模型对训练样本和验证集的识别都不是太好。这说明了该模型对训练样本存在欠拟合。这恰恰是high bias的表现。假设Train set error为15%，而Dev set error为30%，说明了该模型既存在high bias也存在high variance（深度学习中最坏的情况）。再假设Train set error为0.5%，而Dev set error为1%，即low bias和low variance，是最好的情况。值得一提的是，以上的这些假设都是建立在base error是0的基础上，即人类都能正确识别所有猫类图片。base error不同，相应的Train set error和Dev set error会有所变化，但没有相对变化。

一般来说，Train set error体现了是否出现bias，Dev set error体现了是否出现variance（正确地说，应该是Dev set error与Train set error的相对差值）

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Bias%20and%20Variance%202.png)
上图所示模型既存在high bias也存在high variance，可以理解成某段区域是欠拟合的，某段区域是过拟合的。

## 机器学习基础 Basic Recipe for Machine Learning

机器学习中基本的一个诀窍就是避免出现high bias和high variance。首先，减少high bias的方法通常是增加神经网络的隐藏层个数、神经元个数，训练时间延长，选择其它更复杂的NN模型等。在base error不高的情况下，一般都能通过这些方式有效降低和避免high bias，至少在训练集上表现良好。其次，减少high variance的方法通常是增加训练样本数据，进行正则化Regularization，选择其他更复杂的NN模型等。

注意：
第一，解决high bias和high variance的方法是不同的。实际应用中通过Train set error和Dev set error判断是否出现了high bias或者high variance，然后再选择针对性的方法解决问题。
第二，Bias和Variance的折中tradeoff。传统机器学习算法中，Bias和Variance通常是对立的，减小Bias会增加Variance，减小Variance会增加Bias。而在现在的深度学习中，通过使用更复杂的神经网络和海量的训练样本，一般能够同时有效减小Bias和Variance。这也是深度学习之所以如此强大的原因之一。

## 正则化 Regularization

如果出现了过拟合high variance，则需要采用正则化regularization来解决。虽然扩大训练样本数量也是减小high variance的一种方法，但是通常获得更多训练样本的成本太高，比较困难。所以，更可行有效的办法就是使用regularization

对于Logistic regression，采用L2 regularization（向量参数𝑤 的欧几里德范数(2 范数)的平方），其表达式为：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%201.png)

为什么只对w进行正则化而不对b进行正则化呢？其实也可以对b进行正则化。但是一般w的维度很大，而b只是一个常数。相比较来说，参数很大程度上由w决定，改变b值对整体模型影响较小。所以，一般为了简便，就忽略对b的正则化了。

另外一个正则化方法：L1 regularization（向量参数𝑤 的1 范数）
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%202.png)

与L2 regularization相比，L1 regularization得到的w更加稀疏，即很多w为零值。其优点是节约存储空间，因为大部分w为0。然而，实际上L1 regularization在解决high variance方面比L2 regularization并不更具优势。而且，L1的在微分求导方面比较复杂。所以，一般L2 regularization更加常用。
L1、L2 regularization中的就是正则化参数（超参数的一种）。可以设置为不同的值，在Dev set中进行验证，选择最佳的。

在深度学习模型中，L2 regularization的表达式为：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%203.png)
上图中的𝑤矩阵范数被称作“弗罗贝尼乌斯范数”，用下标𝐹标注，我们不称之为“矩阵𝐿2范数”，而称它为“弗罗贝尼乌斯范数“

一个矩阵的Frobenius范数就是计算所有元素平方和再开方，如下所示
$$
||A||_F=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^n{|a_{ij}|^2}}
$$
由于加入了正则化项，梯度下降算法中反向传播的计算表达式需要做如下修改：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%204.png)

L2 regularization也被称做“权重衰减”(weight decay)。这是因为，由于加上了正则项，有个增量，在更新的时候，会多减去这个增量，使得比没有正则项的值要小一些。不断迭代更新，不断地减小。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%205.png)

假如我们选择了非常复杂的神经网络模型，在未使用正则化的情况下，我们得到的分类超平面可能是过拟合情况。但是，如果使用L2 regularization，当λ很大时，权重矩阵w近似为零，意味着该神经网络模型中的某些神经元实际的作用很小，可以忽略。从效果上来看，其实是将某些神经元给忽略掉了。这样原本过于复杂的神经网络模型就变得不那么复杂了，而变得非常简单化了。如下图所示，整个简化的神经网络模型变成了一个逻辑回归模型，可是深度却很大。问题就从high variance变成了high bias了。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%206.png)
因此，选择合适大小的值，就能够同时避免high bias和high variance，得到最佳模型。

还有另外一个直观的例子来解释为什么正则化能够避免发生过拟合。假设激活函数是tanh函数。tanh函数的特点是在z接近零的区域，函数近似是线性的，而当|z|很大的时候，函数非线性且变化缓慢。
用g(z)表示tanh(z)，如果正则化参数 λ 很大，激活函数的参数会相对较小，因为代价函数中的参数变大了。如果𝑊很小，相对来说，𝑧也会很小，则此时的分布在tanh函数的近似线性区域。那么这个神经元起的作用就相当于是linear regression。如果每个神经元对应的权重都比较小，那么整个神经网络模型相当于是多个linear regression的组合，即可看成一个linear network。得到的分类超平面就会比较简单，不会出现过拟合现象。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Regularization%207.png)

## 随机失活正则化 Dropout Regularization

除了L2 regularization之外，还有另外一种防止过拟合的有效方法：Dropout（随机失活）

Dropout是指在深度学习网络的训练过程中，对于每层的神经元，按照一定的概率将其暂时从网络中丢弃。也就是说，每次训练时，每一层都有部分神经元不工作，起到简化复杂网络模型的效果，从而避免发生过拟合。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Dropout%20Regularization%201.png)

Dropout有不同的实现方法，一种常用的方法是Inverted dropout（反向随机失活）。
用一个三层（𝑙 = 3）网络来举例说明。

```Python
# 假设对于第𝑙层神经元，设定保留神经元比例概率keep_prob=0.8，即该层有20%的神经元停止工作。为dropout向量，设置为随机vector，其中80%的元素为1，20%的元素为0。在python中可以使用如下语句生成dropout vector：
dl = np.random.rand(al.shape[0],al.shape[1])< keep_prob
# 然后，第𝑙 层经过dropout，随机删减20%的神经元，只保留80%的神经元，其输出为：
al = np.multiply(al,dl)
# 最后，还要对进行scale up处理，即：
al /= keep_prob
```

之所以要对进行scale up是为了保证在经过dropout后，作为下一层神经元的输入值尽量保持不变。假设第层有50个神经元，经过dropout后，有10个神经元停止工作，这样只有40神经元有作用。那么得到的只相当于原来的80%。scale up后，能够尽可能保持的期望值相比之前没有大的变化。

Inverted dropout的另外一个好处就是在对该dropout后的神经网络进行测试时能够减少scaling问题。因为在训练时，使用scale up保证的期望值没有大的变化，测试时就不需要再对样本数据进行类似的尺度伸缩操作了。

对于m个样本，单次迭代训练时，随机删除掉隐藏层一定数量的神经元；然后，在删除后的剩下的神经元上正向和反向更新权重w和常数项b；接着，下一次迭代中，再恢复之前删除的神经元，重新随机删除一定数量的神经元，进行正向和反向更新w和b。不断重复上述过程，直至迭代训练完成。

值得注意的是，使用dropout训练结束后，在测试和实际应用模型时，不需要进行dropout和随机删减神经元，所有的神经元都在工作。

Dropout通过每次迭代训练时，随机选择不同的神经元，相当于每次都在不同的神经网络上进行训练。除此之外，还可以从权重w的角度来解释为什么dropout能够有效防止过拟合。对于某个神经元来说，某次训练时，它的某些输入在dropout的作用被过滤了。而在下一次训练时，又有不同的某些输入被过滤。经过多次训练后，某些输入被过滤，某些输入被保留。这样，该神经元就不会受某个输入非常大的影响，影响被均匀化了。也就是说，对应的权重w不会很大。这从从效果上来说，与L2 regularization是类似的，都是对权重w进行“惩罚”，减小了w的值。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Dropout%20Regularization%202.png)

在使用dropout的时候，有几点需要注意。首先，不同隐藏层的dropout系数keep_prob可以不同。一般来说，神经元越多的隐藏层，keep_out可以设置得小一些.，例如0.5；神经元越少的隐藏层，keep_out可以设置的大一些，例如0.8，甚至是1。另外，实际应用中，不建议对输入层进行dropout，如果输入层维度很大，例如图片，那么可以设置dropout，但keep_out应设置的大一些，例如0.8，0.9。总体来说，就是越容易出现overfitting的隐藏层，其keep_prob就设置的相对小一些。没有准确固定的做法，通常可以根据validation进行选择。

使用dropout的时候，可以通过绘制cost function来进行debug，看看dropout是否正确执行。一般做法是，将所有层的keep_prob全设置为1，再绘制cost function，即涵盖所有神经元，看J是否单调下降。下一次迭代训练时，再将keep_prob设置为其它值。

## 其他正则化方法 Other Regularization Methods

除了L2 regularization和dropout regularization之外，还有其它减少过拟合的方法。

一种方法是增加训练样本数量。但是通常成本较高，难以获得额外的训练样本。但是，我们可以对已有的训练样本进行一些处理来“制造”出更多的样本，称为data augmentation。例如图片识别问题中，可以对已有的图片进行水平翻转、垂直翻转、任意角度旋转、缩放或扩大等等。如下图所示，这些处理都能“制造”出新的训练样本。虽然这些是基于原有样本的，但是对增大训练样本数量还是有很有帮助的，不需要增加额外成本，却能起到防止过拟合的效果。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Other%20Regularization%20Methods%201.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Other%20Regularization%20Methods%202.png)

还有另外一种防止过拟合的方法：early stopping。一个神经网络模型随着迭代训练次数增加，train set error一般是单调减小的，而dev set error 先减小，之后又增大。也就是说训练次数过多时，模型会对训练样本拟合的越来越好，但是对验证集拟合效果逐渐变差，即发生了过拟合。因此，迭代训练次数不是越多越好，可以通过train set error和dev set error随着迭代次数的变化趋势，选择合适的迭代次数，即early stopping。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Other%20Regularization%20Methods%203.png)

然而，Early stopping有其自身缺点。通常来说，机器学习训练模型有两个目标：一是优化cost function，尽量减小J；二是防止过拟合。这两个目标彼此对立的，即减小J的同时可能会造成过拟合，反之亦然。我们把这二者之间的关系称为正交化orthogonalization。在深度学习中，我们可以同时减小Bias和Variance，构建最佳神经网络模型。但是，Early stopping的做法通过减少得带训练次数来防止过拟合，这样J就不会足够小。也就是说，early stopping将上述两个目标融合在一起，同时优化，但可能没有“分而治之”的效果好。
与early stopping相比，L2 regularization可以实现“分而治之”的效果：迭代训练足够多，减小J，而且也能有效防止过拟合。而L2 regularization的缺点之一是最优的正则化参数的选择比较复杂。对这一点来说，early stopping比较简单。
总的来说，L2 regularization更加常用一些。

## 归一化输入 Normalizing Inputs

在训练神经网络时，标准化输入可以提高训练的速度。标准化输入就是对训练数据集进行归一化的操作。
归一化需要两个步骤：零均值、归一化方差。
即将原始数据减去其均值后，再除以其方差。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Normalizing%20inputs%201.png)

以二维平面为例，下图展示了其归一化过程：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Normalizing%20inputs%202.png)

值得注意的是，由于训练集进行了标准化处理，那么对于测试集或在实际应用时，应该使用同样的和对其进行标准化处理。这样保证了训练集合测试集的标准化操作一致。

之所以要对输入进行标准化操作，主要是为了让所有输入归一化同样的尺度上，方便进行梯度下降算法时能够更快更准确地找到全局最优解。假如输入特征是二维的，且x1的范围是[1,1000]，x2的范围是[0,1]。如果不进行标准化处理，x1与x2之间分布极不平衡，训练得到的w1和w2也会在数量级上差别很大。这样导致的结果是cost function与w和b的关系可能是一个非常细长的椭圆形碗。对其进行梯度下降算法时，由于w1和w2数值差异很大，只能选择很小的学习因子，来避免J发生振荡。一旦较大，必然发生振荡，J不再单调下降。如下左图所示。然而，如果进行了标准化操作，x1与x2分布均匀，w1和w2数值差别不大，得到的cost function与w和b的关系是类似圆形碗。对其进行梯度下降算法时，可以选择相对大一些，且J一般不会发生振荡，保证了J是单调下降的。如下右图所示。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Normalizing%20inputs%203.png)

另外一种情况，如果输入特征之间的范围本来就比较接近，那么不进行标准化操作也是没有太大影响的。但是，标准化处理在大多数场合下还是值得推荐的。

## 梯度消失/梯度爆炸 Vanishing / Exploding gradients

在神经网络尤其是深度神经网络中存在可能存在这样一个问题：梯度消失和梯度爆炸。意思是当训练一个层数非常多的神经网络时，计算得到的梯度可能非常小或非常大，甚至是指数级别的减小或增大。这样会让训练过程变得非常困难。

举个例子来说明，假设一个多层的每层只包含两个神经元的深度神经网络模型，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Vanishing%20and%20Exploding%20gradients.png)

为了简化复杂度，便于分析，我们令各层的激活函数为线性函数，即g(z)=z，且忽略各层常数项b的影响，令b全部为零。那么，该网络的预测输出为：
$$
\widehat{Y} =W^{[l]}W^{[l−1]}W^{[l−2]}⋯W^{[3]}W^{[2]}W^{[1]}X
$$
如果各层权重的元素都稍大于1，例如1.5，则Layer越大，Y-hat越大，且呈指数型增长，称之为数值爆炸。相反，如果各层权重的元素都稍小于1，例如0.5，网络层数Layer越多，Y-hat越小，且呈指数型减小，称之为数值消失。
也就是说，如果各层权重都大于1或者都小于1，那么各层激活函数的输出将随着层数的增加，呈指数型增大或减小。当层数很大时，出现数值爆炸或消失。同样，这种情况也会引起梯度呈现同样的指数型增大或减小的变化。L非常大时，例如L=150，则梯度会非常大或非常小，引起每次更新的步进长度过大或者过小，这让训练过程十分困难。

改善Vanishing and Exploding gradients这类问题的方法是对权重w进行一些初始化处理
深度神经网络模型中，以单个神经元为例，它有n个输入特征，暂时忽略b，其输出为：
$$
z=w_1x_1+w_2x_2+⋯+w_nx_n，a=g(z)
$$
为了让z不会过大或者过小，思路是让w与n有关，且n越大，w应该越小才好。如果把很多𝑤𝑖𝑥i相加，希望每项值更小，最合理的方法就是设置𝑤𝑖=1/n，n表示神经元的输入特征数量。

```python
# 如果激活函数是tanh,此处的n[l-1]表示第l-1层神经元数量，即第l层的输入数量，这被称为 Xavier 初始化
w[l] = np.random.randn(n[l],n[l-1])*np.sqrt(1/n[l-1])
# 如果激活函数是ReLU，权重w的初始化一般令其方差为
w[l] = np.random.randn(n[l],n[l-1])*np.sqrt(2/n[l-1]) 
# Yoshua Bengio提出了另外一种初始化w的方法，令其方差为：
w[l] = np.random.randn(n[l],n[l-1])*np.sqrt(2/n[l-1]*n[l]) 
```

##  梯度的数值逼近和梯度检查 Numerical approximation of gradients and Gradient checking

Back Propagation神经网络有一项重要的测试是梯度检查(gradient checking)，其目的是检查验证反向传播过程中梯度下降算法是否正确。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Numerical%20approximation%20of%20gradients.png)

其中ε趋近于无穷小，而这样计算得出的导数值与直接计算𝑔(𝜃)相比的误差非常小，使得我们确信，𝑔(𝜃)可能是𝑓导数的正确实现。

上文介绍了如何近似求出梯度值，下文将展开如何进行梯度检查(Gradient checking)，来验证训练过程中是否出现bugs：

假设网络中含有下列参数，𝑊[1]和𝑏[1]……𝑊[𝑙]和𝑏[𝑙]，为了执行梯度检验，首先要做的就是，把所有参数转换成一个巨大的向量数据，需要先把所有𝑊矩阵转换成向量，接着做连接运算，得到一个巨型向量𝜃，该向量表示为参数𝜃，代价函数𝐽是所有𝑊和𝑏的函数，现在就得到了一个𝜃的代价函数𝐽（即𝐽(𝜃)）。接着，得到与𝑊和𝑏顺序相同的数据，同样可以把反向传播中的𝑑𝑊[1]和𝑑𝑏[1]……𝑑𝑊[𝑙]和𝑑𝑏[𝑙]转换成一个新的向量，用它们来初始化大向量𝑑𝜃，它与𝜃具有相同维度。

首先，我们清楚𝐽是超参数𝜃的一个函数，𝐽函数可以展开为𝐽(𝜃1, 𝜃2, 𝜃3,… … )，接着利用对每个 d𝜃_i 计算近似梯度，其值与反向传播算法得到的相比较，检查是否一致。例如，对于第i个元素，近似梯度为：
$$
dθ_{approx}[i]=\frac{J(θ_1,θ_2,⋯,θ_i+ε,⋯)−J(θ_1,θ_2,⋯,θ_i−ε,⋯)}{2ε}
$$
计算完所有的近似梯度后，可以计算与反向传播得到的 d𝜃 的欧氏（Euclidean）距离来比较二者的相似度
$$
\frac{||d𝜃_{approx}-d𝜃||_2}{||d𝜃_{approx}||_2+||d𝜃||_2}
$$
一般来说，如果欧氏距离较小，10^-7或更小，表明二者接近，即反向梯度计算是正确的，没有bugs。如果欧氏距离较大，10^-5，则表明梯度计算可能出现问题，需要再次检查是否有bugs存在。如果欧氏距离很大，10^-3，则表明二者差别很大，梯度下降计算过程有bugs，需要仔细检查。

在进行梯度检查的过程中有几点需要注意的地方：

不要在整个训练过程中都进行梯度检查，仅仅作为debug使用。

如果梯度检查出现错误，找到对应出错的梯度，检查其推导是否出现错误。

注意不要忽略正则化项，计算近似梯度的时候要包括进去。

梯度检查时关闭dropout，检查完毕后再打开dropout。

随机初始化时运行梯度检查，经过一些训练后再进行梯度检查（不常用）

# 优化算法  Optimization Algorithms

##  Mini-batch 梯度下降 Mini-batch Gradient Descent 

神经网络训练过程是对所有m个样本，称为batch，通过向量化计算方式，同时进行的。如果m很大，例如达到百万数量级，训练速度往往会很慢，因为每次迭代都要对所有样本进行进行求和运算和矩阵运算，这种梯度下降算法被称为Batch Gradient Descent。

向量化能够有效地对所有 m 个样本进行计算，允许处理整个训练集，而无需某个明确的公式。用一个巨大的矩阵X表示训练样本，其维度是(𝑛_𝑥, 𝑚)，结果Y也是如此，其维度是(1, 𝑚)
$$
X=[x^{(1)}x^{(2)}...x^{(m)}]，Y=[y^{(1)}y^{(2)}...y^{(m)}]
$$
为了解决这一问题，我们可以把m个训练样本分成若干个子集，称为mini-batches，这样每个子集包含的数据量就小了，然后每次在单一子集上进行神经网络训练，速度就会大大提高，这种梯度下降算法被称为Mini-batch Gradient Descent。

假设总的训练样本个数m=5000000，每个mini-batch只有1000个样本，那么一共存在5000个mini-batch
$$
不妨将训练样本x^{(1)}到x^{(1000)}取出记作X^{\{1\}},训练样本x^{(1001)}到x^{(2000)}取出记作X^{\{2\}},X的维数为（n_x,1000）
$$
对Y也进行相同处理，相同表示
$$
那么存在X^{\{1\}}～X^{\{5000\}}，Y^{\{1\}}～Y^{\{5000\}},Y的维数为（1,1000）
$$
符号总结：
上角小括号(𝑖)表示训练集里的值，所以𝑥^(𝑖)是第𝑖个训练样本
上角中括号[𝑙]来表示神经网络的层数，𝑧^[𝑙]表示神经网络中第𝑙层的𝑧值
我们现在引入了{𝑡}来代表不同的 mini-batch，所以存在X^{t}，Y^{t}

Mini-batches Gradient Descent的实现过程是先将总的训练样本分成T个子集（mini-batches），然后对每个mini-batch进行神经网络训练，包括Forward Propagation，Compute Cost Function，Backward Propagation，循环至T个mini-batch都训练完毕。
经过T次循环之后，所有m个训练样本都进行了梯度下降计算。这个过程，我们称之为经历了一个epoch。对于Batch Gradient Descent而言，一个epoch只进行一次梯度下降算法；而Mini-Batches Gradient Descent，一个epoch会进行T次梯度下降算法。

值得一提的是，对于Mini-Batches Gradient Descent，可以进行多次epoch训练。而且，每次epoch，最好是将总体训练数据重新打乱，重新分成T组mini-batches，这样有利于训练出最佳的神经网络模型。

Batch gradient descent和Mini-batch gradient descent的cost曲线如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Mini-batch%20gradient%20descent%201.png)

对于一般的神经网络模型，使用Batch gradient descent，随着迭代次数增加，cost是不断减小的。然而，使用Mini-batch gradient descent，随着在不同的mini-batch上迭代训练，其cost不是单调下降，而是受类似noise的影响，出现振荡。但整体的趋势是下降的，最终也能得到较低的cost值。之所以出现细微振荡的原因是不同的mini-batch之间是有差异的，例如可能第一个子集是好的子集，而第二个子集包含了一些噪声noise，出现细微振荡是正常的。

如何选择每个mini-batch的大小：有两个极端：如果mini-batch size=m，即为Batch gradient descent，只包含一个子集；如果mini-batch size=1，即为Stochastic gradient descent，每个样本就是一个子集，共有m个子集。

比较一下Batch gradient descent和Stochastic gradient descent的梯度下降曲线。如下图所示，蓝色的线代表Batch gradient descent，紫色的线代表Stachastic gradient descent。Batch gradient descent会比较平稳地接近全局最小值，但是因为使用了所有m个样本，每次前进的速度有些慢。Stachastic gradient descent每次前进速度很快，但是路线曲折，有较大的振荡，最终会在最小值附近来回波动，难以真正达到最小值处。而且在数值处理上就不能使用向量化的方法来提高运算速度。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Mini-batch%20gradient%20descent%202.png)

实际使用中，mini-batch size不能设置得太大（Batch gradient descent），也不能设置得太小（Stachastic gradient descent）。这样，相当于结合了Batch gradient descent和Stachastic gradient descent各自的优点，既能使用向量化优化算法，又能叫快速地找到最小值。mini-batch gradient descent的梯度下降曲线如下图绿色所示，每次前进速度较快，且振荡较小，基本能接近全局最小值。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Mini-batch%20gradient%20descent%203.png)

一般来说，如果总体样本数量m不太大时（小于2000个样本），可以直接使用Batch gradient descent。如果总体样本数量m很大时，需要将样本分成许多mini-batches，推荐常用的mini-batch size为64,128,256,512，这些都是2的幂，之所以这样设置的原因是计算机存储数据一般是2的幂，这样设置可以提高运算速度。

## 指数加权平均数 Exponentially Weighted Averages

举个例子，记录半年内伦敦市的气温变化，并在二维平面上绘制出来，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Exponentially%20weighted%20averages%201.png)
看上去，温度数据似乎有noise，而且抖动较大。如果我们希望看到半年内气温的整体变化趋势，可以通过移动平均（moving average)的方法来对每天气温进行平滑处理。

例如我们可以设V_0 = 0，当成第0天的气温值。
后续每天，需要使用 0.9 的加权数乘以之前的数值加上当日温度的 0.1 倍。
$$
例如：V_1=0.9V_0+0.1\theta_1
$$
即第t天与第t-1天的气温迭代关系为：
$$
V_t=0.9V_{t-1}+0.1\theta_t=0.9^t·V_0+0.9^{t-1}·0.1\theta_1+0.9^{t-2}·0.1\theta_2+……+0.9·0.1\theta_{t-1}+0.1\theta_t
$$
经过移动平均处理得到的气温如下图红色曲线所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Exponentially%20weighted%20averages%202.png)

这种滑动平均算法称为指数加权平均（exponentially weighted average）。根据之前的推导公式，其一般形式为：
$$
V_t = \beta V_{t-1}+(1-\beta)\theta_t
$$
β的值决定了指数加权平均的天数，V_t可近似约等于为 1/(1-β)天的平均温度。即当β为0.9，表示将前10天进行指数加权平均；当β为0.98，表示将前50天进行指数加权平均。β值越大，则指数加权平均的天数越多，平均后的趋势线就越平缓，但是同时也会向右平移（可以理解为β很大时，相当于给前一天的值加了太多的权重，只有很少的权重给了当日的值，所以指数加权平均值适应地更缓慢一些）。下图绿色曲线和黄色曲线分别表示了β=0.98和β=0.5时，指数加权平均的结果。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Exponentially%20weighted%20averages%203.png)

那么对于上述例子，我们可以计算第100天的温度
$$
V_{100}=0.1\theta_{100}+0.1·0.9\theta_{99}+0.1·0.9^2\theta_{98}+0.1·0.9^3\theta_{97}+……
$$
我么可以构建一个指数衰减函数，从 0.1 开始，到0.1 × 0.9，到0.1 × (0.9)^2，以此类推。那么显然，计算V_100是选取每日温度，将其与指数衰减函数相乘，然后求和。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Exponentially%20Weighted%20Averages%204.png)

那么到底平均了多少天的温度？实际上 0.9^10 大约为0.35，这大约是 1/e，那么即10天之后，曲线的高度下降到 1/3，相当于在峰值的 1/e。又因此当β= 0.9的时候，我们说仿佛你在计算一个指数加权平均数，只关注了过去 10天的温度，因为 10 天后，权重下降到不到当日权重的三分之一。
$$
根据高等数学，我们知道：\beta^{\frac{1}{1-\beta}}=\frac{1}{e}，或者说 (1-\frac{1}{\beta})^{\beta}=\frac{1}{e}
$$
指数加权平均的偏差修正(Bias correction in exponentially weighted averages)可以让平均数运算更加准确。当β的值越靠近于1，其曲线起点越低，也就意味着初始阶段的估计不准确。
$$
解决办法：估测初期，不使用V_t，而是使用 \frac{V_t}{1-\beta^t}，t就是现在的天数
$$

$$
具体的例子：当t=2，\beta=0.98时，对第二天的估测变为\frac{V_2}{1-0.98^2}=\frac{0.98·0.02·\theta_1+0.02\theta_2}{1-0.98^2}
$$

很明显地，随着t的增大，β^t接近于0，这时候偏差修正几乎没有作用。

## 动量梯度下降法 Gradient Descent with Momentum

动量梯度下降算法，其速度要比传统的梯度下降算法快很多。做法是在每次训练时，对梯度进行指数加权平均处理，然后用得到的梯度值更新权重W和常数项b。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Gradient%20Descent%20with%20Momentum.png)
原始的梯度下降算法如上图蓝色折线所示。在梯度下降过程中，梯度下降的振荡较大，尤其对于W、b之间数值范围差别较大的情况。此时每一点处的梯度只与当前方向有关，产生类似折线的效果，前进缓慢。而如果对梯度进行指数加权平均，这样使当前梯度不仅与当前方向有关，还与之前的方向有关，这样处理让梯度前进方向更加平滑，减少振荡，能够更快地到达最小值处。

权重W和常数项b的指数加权平均表达式如下
$$
V_{dW}=\beta·V_{dW}+(1-\beta)·dW,V_{db}=\beta·V_{db}+(1-\beta)·db
$$
然后重新赋值权重
$$
W:=W-\alpha V_{dW},b:=b-\alpha V_{db}
$$
从动量的角度来看，Momentunm项(V_dW)可以看成速度V，微分项(dW)可以看成是加速度a，β可以看成是一些摩擦力。指数加权平均实际上是计算当前的速度，当前速度由之前的速度和现在的加速度共同影响。而β，又能限制速度过大。也就是说，当前的速度是渐变的，而不是瞬变的，是动量的过程。这保证了梯度下降的平稳性和准确性，减少振荡，较快地达到最小值处。

## RMSprop

RMSprop 的算法，全称是 root mean square prop 算法，它也可以加速梯度下降。

每次迭代训练过程中，其权重W和常数项b的更新表达式如下
$$
S_{dW}=\beta S_{dW}+(1-\beta)dW^2,S_{db}=\beta S_{db}+(1-\beta)db^2
$$

$$
W:=W-\alpha \frac{d_W}{\sqrt{S_{dW}}+\varepsilon},b:=b-\alpha \frac{d_b}{\sqrt{S_{db}}+\varepsilon},
$$

其中平方是针对整个符号的平方，即 (dW)^2，还有一点需要注意的是为了避免RMSprop算法中分母为零，通常可以在分母增加一个极小的常数ε，可以取10^-8

下面简单解释一下RMSprop算法的原理，以下图为例，为了便于分析，令水平方向为W的方向，垂直方向为b的方向。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/RMSprop%202.png)
从图中可以看出，梯度下降（蓝色折线）在垂直方向（b）上振荡较大，在水平方向（W)上振荡较小，表示在b方向上梯度较大，而在W方向上梯度较小。

我们希望学习速度快，而在垂直方向，也就是图中的b方向，我们希望减缓垂直方向上的摆动，所以有了𝑆𝑑𝑊和𝑆𝑑𝑏，我们希望𝑆𝑑𝑊会
相对较小，所以我们要除以一个较小的数，而希望𝑆𝑑𝑏又较大，所以这里我们要除以较大的数字，这样就可以减缓纵轴上的变化。观察图示微分，垂直方向的要比水平方向的大得多，所以斜率在𝑏方向特别大，所以这些微分中，𝑑𝑏较大，𝑑𝑊较小，因为函数的倾斜程度，在
纵轴上，也就是 b 方向上要大于在横轴上，也就是𝑊方向上。𝑑𝑏的平方较大，所以𝑆𝑑𝑏也会较大，而相比之下，𝑑𝑊会小一些，亦或𝑑𝑊平方会小一些，因此𝑆𝑑𝑊会小一些，结果就是纵轴上的更新要被一个较大的数相除，就能消除摆动，而水平方向的更新则被较小的数相除。即加快了W方向的速度，减小了b方向的速度，减小振荡，实现快速梯度下降算法，其梯度下降过程如绿色折线所示。总得来说，就是如果哪个方向振荡大，就减小该方向的更新速度，从而减小振荡。
当然，也可以用一个更大的学习率，加快学习，就无需在垂直方向上偏离。

##  Adam优化算法 Adam Optimization Algorithm

Adam(Adaptive Moment Estimation) 优化算法基本上就是将 Momentum 和 RMSprop 结合在一起.

其算法流程为：
$$
初始化，令V_{dW}=0,S_{dW}=0,V_{db}=0,S_{db}=0
$$
在第t次迭代中，计算微分，用当前的 mini-batch 计算𝑑𝑊，𝑑𝑏，一般会用 mini-batch 梯度下降法，接下来计算 Momentum 指数加权平均数（Momentum使用的超参数记为β1，RMSprop使用的超参数记为β2）
$$
V_{dW}=\beta_1·V_{dW}+(1-\beta_1)·dW,V_{db}=\beta_1·V_{db}+(1-\beta_1)·db
$$
接着用 RMSprop 进行更新
$$
S_{dW}=\beta_2 S_{dW}+(1-\beta_2)dW^2,S_{db}=\beta_2 S_{db}+(1-\beta_2)db^2
$$
一般使用 Adam 算法的时候，要计算偏差修正
$$
V^{corrected}_{dW}=\frac{V_{dw}}{1-\beta_1^t},V^{corrected}_{db}=\frac{V_{db}}{1-\beta_1^t}
$$

$$
S^{corrected}_{dW}=\frac{S_{dw}}{1-\beta_2^t},S^{corrected}_{db}=\frac{S_{db}}{1-\beta_2^t}
$$

最后更新权重（如果只是用Momentum，使用V𝑑𝑊或者修正后的V𝑑𝑊，但现在加入了RMSprop的部分，所以要除以修正后

𝑆𝑑𝑊的平方根加上𝜀）
$$
W:=W-\alpha \frac{V^{corrected}_{dW}}{\sqrt{S^{corrected}_{dW}}+\varepsilon},b:=b-\alpha \frac{V^{corrected}_{db}}{\sqrt{S^{corrected}_{db}}+\varepsilon},
$$
所以 Adam 算法结合了 Momentum 和 RMSprop 梯度下降法，并且是一种极其常用的学习算法，被证明能有效适用于不同神经网络，适用于广泛的结构。

本算法中有很多超参数，超参数学习率a很重要，也经常需要调试，可以尝试一系列值，然后看哪个有效。𝛽1常用的缺省值为 0.9，这是 dW 的移动平均数，也就是𝑑𝑊的加权平均数，这是 Momentum 涉及的项。至于超参数𝛽2，Adam 论文作者，也就是 Adam 算法的发明者，推荐使用 0.999，这是在计算(𝑑𝑊)^2以及(𝑑𝑏)^2的移动加权平均值，关于𝜀的选择其实没那么重要，Adam 论文的作者建议𝜀10^−8，但并不需要设置它，因为它并不会影响算法表现。但是在使用 Adam 的时候，人们往往使用缺省值即可，𝛽1，𝛽2和𝜀都是如此。

## 学习率衰减 Learning Rate  Decay 

Learning rate decay就是随着迭代次数增加，学习因子逐渐减小。下面用图示的方式来解释这样做的好处。下图中，蓝色折线表示使用恒定的学习因子，由于每次训练相同，步进长度不变，在接近最优值处的振荡也大，在最优值附近较大范围内振荡，与最优值距离就比较远。绿色折线表示使用不断减小的，随着训练次数增加，逐渐减小，步进长度减小，使得能够在最优值处较小范围内微弱振荡，不断逼近最优值。相比较恒定的来说，learning rate decay更接近最优值。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Learning%20Rate%20%20Decay%201.png)
$$
\alpha=\frac{1}{1+decay\_rate*epoch}\alpha_0
$$
其中，deacy_rate是衰减率，epoch是训练完所有样本的次数，a0为初始学习率。随着epoch增加，会不断变小。

除了上面计算的公式之外，还有其它可供选择的计算公式
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Learning%20Rate%20%20Decay%202.png)
其中，k为可调参数，t为mini-bach number。

除此之外，还可以设置为关于t的离散值，随着t增加，呈阶梯式减小。当然，也可以根据训练情况灵活调整当前的值，但会比较耗时间。

## 局部最优化问题 The problem of local optima

在使用梯度下降算法不断减小cost function时，可能会得到局部最优解（local optima）而不是全局最优解（global optima）。之前我们对局部最优解的理解是形如碗状的凹槽，如下图左边所示。但是在神经网络中，local optima的概念发生了变化。准确地来说，大部分梯度为零的“最优点”并不是这些凹槽处，而是形如右边所示的马鞍状，称为saddle point。也就是说，梯度为零并不能保证都是convex（极小值），也有可能是concave（极大值）。特别是在神经网络中参数很多的情况下，所有参数梯度为零的点很可能都是右边所示的马鞍状的saddle point，而不是左边那样的local optimum。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/The%20problem%20of%20local%20optima%201.png)

类似马鞍状的plateaus会降低神经网络学习速度。Plateaus是梯度接近于零的平缓区域，如下图所示。在plateaus上梯度很小，前进缓慢，到达saddle point需要很长时间。到达saddle point后，由于随机扰动，梯度一般能够沿着图中绿色箭头，离开saddle point，继续前进，只是在plateaus上花费了太多时间。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/The%20problem%20of%20local%20optima%202.png)

总的来说，关于local optima，有两点总结：只要选择合理的强大的神经网络，一般不太可能陷入local optima；Plateaus可能会使梯度下降变慢，降低学习速度。另外，上述总结的的动量梯度下降，RMSprop，Adam算法都能有效解决plateaus下降过慢的问题，大大提高神经网络的学习速度。

# 超参数调试 Hyperparameters Tuning

## 调试处理 Tuning Process

深度神经网络需要调试的超参数（Hyperparameters）较多，包括:
α ：学习因子
β：动量梯度下降因子
β1,β2,ε：Adam算法参数
#layers：神经网络层数
#hidden units：各隐藏层神经元个数
learning rate decay：学习因子下降参数
mini-batch size：批量训练样本包含的样本个数

超参数之间也有重要性差异。通常来说，学习因子α是最重要的超参数，也是需要重点调试的超参数。动量梯度下降因子β、各隐藏层神经元个数#hidden units和mini-batch size的重要性仅次于α。然后就是神经网络层数#layers和学习因子下降参数learning rate decay。最后，Adam算法的三个参数β1,β2,ε一般常设置为0.9，0.999和10^−8，不需要反复调试。当然，这里超参数重要性的排名并不是绝对的，具体情况，具体分析。

如何选择和调试超参数？传统的机器学习中，我们对每个参数等距离选取任意个数的点，然后，分别使用不同点对应的参数组合进行训练，最后根据验证集上的表现好坏，来选定最佳的参数。例如有两个待调试的参数，分别在每个参数上选取5个点，这样构成了5x5=25中参数组合，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Tuning%20Process%201.png)

这种做法在参数比较少的时候效果较好。但是在深度神经网络模型中，我们一般不采用这种均匀间隔取点的方法，比较好的做法是使用随机选择。也就是说，对于上面这个例子，我们随机选择25个点，作为待调试的超参数，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Tuning%20Process%202.png)

随机化选择参数的目的是为了尽可能地得到更多种参数组合。还是上面的例子，如果使用均匀采样的话，每个参数只有5种情况；而使用随机采样的话，每个参数有25种可能的情况，因此更有可能得到最佳的参数组合。

这种做法带来的另外一个好处就是对重要性不同的参数之间的选择效果更好。假设hyperparameter1为α，hyperparameter2为ε，显然二者的重要性是不一样的。如果使用第一种均匀采样的方法，ε的影响很小，相当于只选择了5个α值。而如果使用第二种随机采样的方法，ε和α都有可能选择25种不同值。这大大增加了α调试的个数，更有可能选择到最优值。其实，在实际应用中完全不知道哪个参数更加重要的情况下，随机采样的方式能有效解决这一问题，但是均匀采样做不到这点。

在经过随机采样之后，我们可能得到某些区域模型的表现较好。然而，为了得到更精确的最佳参数，我们应该继续对选定的区域进行由粗到细的采样（coarse to fine sampling scheme）。也就是放大表现较好的区域，再对此区域做更密集的随机采样。例如，对下图中右下角的方形区域再做25点的随机采样，以获得最佳参数。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Tuning%20Process%203.png)

## 为超参数选择合适的范围 Using an appropriate scale to pick hyperparameters

调试参数使用随机采样，对于某些超参数是可以进行尺度均匀采样的，但是某些超参数需要选择不同的合适尺度进行随机采样。

例如对于超参数#layers和#hidden units，都是正整数，是可以进行均匀随机采样的，即超参数每次变化的尺度都是一致的（如每次变化为1，犹如一个刻度尺一样，刻度是均匀的）。

但是，对于某些超参数，可能需要非均匀随机采样（即非均匀刻度尺）。例如超参数α，待调范围是[0.0001, 1]。如果使用均匀随机采样，那么有90%的采样点分布在[0.1, 1]之间，只有10%分布在[0.0001, 0.1]之间。这在实际应用中是不太好的，因为最佳的α值可能主要分布在[0.0001, 0.1]之间，而[0.1, 1]范围内α值效果并不好。因此我们更关注的是区间[0.0001, 0.1]，应该在这个区间内细分更多刻度。

通常的做法是将linear scale转换为log scale，将均匀尺度转化为非均匀尺度，然后再在log scale下进行均匀采样。这样，[0.0001, 0.001]，[0.001, 0.01]，[0.01, 0.1]，[0.1, 1]各个区间内随机采样的超参数个数基本一致，也就扩大了之前[0.0001, 0.1]区间内采样值个数。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/appropriate%20scale%20%201.png)

一般解法是，如果线性区间为[a, b]，令m=log(a)，n=log(b)，则对应的log区间为[m,n]。对log区间的[m,n]进行随机均匀采样，然后得到的采样值r，最后反推到线性区间，即10^r。10^r就是最终采样的超参数。相应的Python语句为：

```Python
m = np.log10(a)
n = np.log10(b)
r = np.random.rand()
r = m + (n-m)*r
r = np.power(10,r)
```

除了α之外，动量梯度因子β也是一样，在超参数调试的时候也需要进行非均匀采样。一般β的取值范围在[0.9, 0.999]之间，那么1−β的取值范围就在[0.001, 0.1]之间。那么直接对1−β在[0.001, 0.1]区间内进行log变换即可

在训练深度神经网络时，一种情况是受计算能力所限，我们只能对一个模型进行训练，调试不同的超参数，使得这个模型有最佳的表现。我们称之为Babysitting one model。另外一种情况是可以对多个模型同时进行训练，每个模型上调试不同的超参数，根据表现情况，选择最佳的模型。我们称之为Training many models in parallel。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/appropriate%20scale%20%202.png)

# Batch标准化 Batch Normalization

## Batch Normalization

Batch Normalization不仅可以让调试超参数更加简单，而且可以让神经网络模型更加“健壮”。也就是说较好模型可接受的超参数范围更大一些，包容性更强，使得更容易去训练一个深度神经网络。

前文提到，在训练神经网络时，标准化输入可以提高训练的速度。方法是对训练数据集进行归一化的操作，即将原始数据减去其均值后，再除以其方差。但是标准化输入只是对输入进行了处理，那么对于神经网络，又该如何对各隐藏层的输入进行标准化处理呢？

在神经网络中，第i层隐藏层的输入就是第i-1层隐藏层的输出。对各层激活函数进行标准化处理，从原理上来说可以提高和的训练速度和准确度。这种对各隐藏层的标准化处理就是Batch Normalization。值得注意的是，实际应用中，一般是对隐藏单元值z进行标准化处理而不是激活函数a本身，其实差别不是很大。

Batch Normalization对第 l 层隐藏层的输入做如下标准化处理，忽略上标方括号表示的层数 l：
$$
\mu=\frac{1}{m}\sum_{i=1}z^{(i)},\sigma^2=\frac{1}{m}\sum_{i=1}(z^{(i)}-\mu)^2,z_{norm}^{(i)}=\frac{z^{(i)-\mu}}{\sqrt {\sigma^2+\varepsilon}}
$$
现在已把这些𝑧值标准化，化为含平均值 0 和标准单位方差，所以𝑧的每一个分量都含有平均值 0 和方差 1，但是，大部分情况下并不希望所有的均值都为0，方差都为1，也不太合理。通常需要对进行进一步处理：
$$
\widetilde{z}^{(i)}=\gamma z_{norm}^{(i)}+\beta
$$
这里𝛾和𝛽是模型的学习参数，所以我们使用梯度下降或一些其它类似梯度下降的算法，比如 Momentum 或者 Nesterov，Adam，会更新𝛾和𝛽，正如更新神经网络的权重一样。通过对𝛾和𝛽合理设定，规范化过程，从根本来说，只是计算恒等函数，通过赋予𝛾和𝛽其它值，可以构造含其它平均值和方差的隐藏单元值。

值得注意的是，输入的标准化处理Normalizing inputs和隐藏层的标准化处理Batch Normalization是有区别的。Normalizing inputs使所有输入的均值为0，方差为1。而Batch Normalization可使各隐藏层输入的均值和方差为任意值。实际上，从激活函数的角度来说，如果各隐藏层的输入均值在靠近0的区域即处于激活函数的线性区域，这样不利于训练好的非线性神经网络，得到的模型效果也不会太好。这也解释了为什么需要用和来对作进一步处理。

对于L层神经网络，经过Batch Norm的作用，整体流程如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Batch%20Norm%201.png)
显然，Batch Norm是发生在计算隐藏层单元值z和激活函数a之间的

实际上，Batch Norm经常使用在mini-batch上，这也是其名称的由来。值得注意的是，因为Batch Norm对各隐藏层有去均值的操作，所以这里的常数项可以消去，其数值效果完全可以由𝛽来实现；因此，我们在使用Batch Norm的时候，可以忽略各隐藏层的常数项。
在使用梯度下降算法时，运行𝑡 = 1到 batch 数量的 for 循环，你会在 mini-batch 𝑋{𝑡}上应用正向 prop，每个隐藏层都应用正向 prop，用 Batch 归一化代替𝑧^[𝑙]为𝑧̃^[𝑙]。接下来，它确保在这个 mini-batch 中，𝑧值有归一化的均值和方差，归一化均值和方差后是𝑧̃^[𝑙]，然后，你用反向 prop 计算𝑑𝑤^[𝑙]和𝑑𝑏^[𝑙]，及所有 l 层所有的参数，𝑑𝛽^[𝑙]和𝑑𝛾^[𝑙]。尽管严格来说，因为你要去掉𝑏，这部分其实已经去掉了。最后，你更新这些参数：𝑤^[𝑙] = 𝑤^[𝑙] − αd𝑤^[𝑙]，和以前一样，𝛽^[𝑙] = 𝛽^[𝑙] − 𝛼𝑑𝛽^[𝑙]，对于𝛾也是如此𝛾^[𝑙] = 𝛾^[𝑙] − 𝛼𝑑𝛾^[𝑙].
除了使用梯度下降法更新mini-batch，也可以使用动量梯度下降、RMSprop或者Adam等优化算法更新由 Batch 归一化添加到算法中的𝛽 和𝛾 参数。

## Why does Batch Norm work？

我们可以把输入特征做均值为0，方差为1的规范化处理，来加快学习速度。而Batch Norm也是对隐藏层各神经元的输入做类似的规范化处理。总的来说，Batch Norm不仅能够提高神经网络训练速度，而且能让神经网络的权重W的更新更加“稳健”，尤其在深层神经网络中更加明显。比如神经网络很后面的W对前面的W包容性更强，即前面的W的变化对后面W造成的影响很小，整体网络更加健壮。

举个例子来说明，假如用一个浅层神经网络（类似逻辑回归）来训练识别猫的模型。如下图所示，提供的所有猫的训练样本都是黑猫。然后，用这个训练得到的模型来对各种颜色的猫样本进行测试，测试的结果可能并不好。其原因是训练样本不具有一般性（即不是所有的猫都是黑猫），这种训练样本（黑猫）和测试样本（猫）分布的变化称之为covariate shift。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Batch%20Norm%202.png)

对于这种情况，如果实际应用的样本与训练样本分布不同，即发生了covariate shift，则一般是要对模型重新进行训练的。在神经网络，尤其是深度神经网络中，covariate shift会导致模型预测效果变差，重新训练的模型各隐藏层的和均产生偏移、变化。而Batch Norm的作用恰恰是减小covariate shift的影响，让模型变得更加健壮，鲁棒性更强。Batch Norm减少了各层、之间的耦合性，让各层更加独立，实现自我训练学习的效果。也就是说，如果输入发生covariate shift，那么因为Batch Norm的作用，对个隐藏层输出进行均值和方差的归一化处理，和更加稳定，使得原来的模型也有不错的表现。针对上面这个黑猫的例子，如果我们使用深层神经网络，使用Batch Norm，那么该模型对花猫的识别能力应该也是不错的。

从另一个方面来说，Batch Norm也起到轻微的正则化（regularization）效果。具体表现在：
每个mini-batch都进行均值为0，方差为1的归一化操作
每个mini-batch中，对各个隐藏层的添加了随机噪声，效果类似于Dropout
mini-batch越小，正则化效果越明显

但是，Batch Norm的正则化效果比较微弱，正则化也不是Batch Norm的主要功能

# Softmax回归 Softmax Regression

## Softmax Regression

二分类问题，神经网络输出层只有一个神经元，只有两种可能的标记 0 或 1。存在一种 logistic 回归的一般形式，叫做 Softmax 回归，能使你识别某一分类时做出预测，或者说是多种分类中的一个，不只是识别两个分类。

对于多分类问题，用C表示种类个数，神经网络中输出层就有C个神经元，即指示类别的数字，从 0 到𝐶 − 1。其中，每个神经元的输出依次对应属于该类的概率。

Softmax回归模型输出层的激活函数如下所示：
$$
计算输出层的z变量：z^{[L]}=W^{[L]}a^{[L-1]}+b^{[L]}
$$

$$
计算z后使用Softmax激活函数：a_i^{[L]}=\frac{e^{z_i^{[L]}}}{\sum_{i=1}^Ce^{z_i^{[L]}}}
$$

输出层每个神经元的输出对应属于该类的概率，满足
$$
\sum_{i=1}^Ca_i^{[L]}=1
$$
设a^[L]=g^\[L](z^[L])，这一激活函数g的与众不同之处在于需要输入一个 C×1 维向量，然后输出一个 C×1 维向量。之前的激活函数都是接受单行数值输入，例如 Sigmoid 和 ReLu 激活函数，输入一个实数。Softmax 激活函数的特殊之处在于，因为需要将所有可能的输出归一化，就需要输入一个向量，最后输出一个向量。

下图为几个简单的线性多分类的例子图示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Softmax%20Regression.png)

## Training a Softmax classifier

Softmax classifier的训练过程与二元分类问题有所不同。

举例来说，假如C=4
$$
某样本预测输出\widehat{y}=
\begin{bmatrix}
    0.3\\ 0.2\\ 0.1\\ 0.4
\end{bmatrix},
某样本真实标签y=
\begin{bmatrix}
    0\\ 1\\ 0\\ 0
\end{bmatrix}
$$
明显地，预测为第四类，但实际为第二类，因此该预测效果不佳。

我们定义softmax classifier的loss function如下：
$$
L（\widehat{y},y）=-\sum_{i=1}^Cy_i·log\widehat{y_i}
$$
由上述例子可知，最后的损失函数L = -log hat(y_2)，这就意味着，如果学习算法试图将它变小，因为梯度下降法是用来减少训练集的
损失的，要使它变小的唯一方式就是使−log hat(y_2)变小，要想做到这一点，就需要使hat(y_2)尽可能大，因为这些是概率，它又不可能比 1 大。概括来讲，损失函数所做的就是它找到训练集中的真实类别，然后试图使该类别相应的概率尽可能地高，这其实就是最大似然估计的一种形式。

所有m个样本的cost function为：
$$
J(w^{[1]},b^{[1]}……)=\frac{1}{m}\sum_{i=1}^mL（\widehat{y}^{(i)},y^{(i)}）
$$
softmax classifier的反向传播过程仍然使用梯度下降算法，其推导过程与二元分类有一点不一样，因为只有输出层的激活函数不一样。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/softmax%20classifier.png)

可见的表达式与二元分类结果是一致的，虽然推导过程不太一样。然后就可以继续进行反向传播过程的梯度下降算法了，推导过程与二元分类神经网络完全一致。

# 卷积神经网络 Convolutional Neural Networks

机器视觉（Computer Vision）是深度学习应用的主要方向之一。一般的CV问题包括以下三类：Image Classification、Object Detection、Neural Style Transfer

使用传统神经网络处理机器视觉的一个主要问题是输入层维度很大。例如一张64x64x3的图片，神经网络输入层的维度为12288。如果图片尺寸较大，例如一张1000x1000x3的图片，神经网络输入层的维度将达到3百万，使得网络权重W非常庞大。这样会造成两个后果，一是神经网络结构复杂，数据量相对不够，容易出现过拟合；二是所需内存、计算量较大。解决这一问题的方法就是使用卷积神经网络（CNN）。

## 边缘检测 Edge Detection

对于CV问题，神经网络由浅层到深层，分别可以检测出图片的边缘特征 、局部特征（例如眼睛、鼻子等）、整体面部轮廓。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%201.png)

最常检测的图片边缘有两类：一是垂直边缘（vertical edges），二是水平边缘（horizontal edges）
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%202.png)

图片的边缘检测可以通过与相应滤波器进行卷积来实现。以垂直边缘检测为例，原始图片尺寸为6x6，滤波器filter尺寸为3x3，卷积后的图片尺寸为4x4。在6x6的矩阵上，每一行最多可以匹配4个3x3的滤波器，每一列也最多可以匹配4个3x3的滤波器，所以卷积后的图片尺寸是4x4。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%203.png)

∗表示卷积操作
python中，卷积用conv_forward()表示
tensorflow中，卷积用tf.nn.conv2d()表示
keras中，卷积用Conv2D()表示
卷积运算不是矩阵乘法，上图只显示了卷积后的第一个值和最后一个值，卷积运算的过程是每个元素与滤波器的对应元素相乘求和，例如卷积后的第一个值-5=3x1+1x1+2x1+0x0+5x0+7x0+1x-1+8x-1+2x-1.

Vertical edge detection能够检测图片的垂直方向边缘。下图对应一个垂直边缘检测的例子：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%204.png)

图片边缘有两种渐变方式，一种是由明变暗，另一种是由暗变明。以垂直边缘检测为例，下图展示了两种方式的区别。实际应用中，这两种渐变方式并不影响边缘检测结果，可以对输出图片取绝对值操作，得到同样的结果。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%205.png)

垂直边缘检测和水平边缘检测的滤波器算子如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%206.png)

下图展示一个水平边缘检测的例子：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%207.png)

除了上面提到的这种简单的Vertical、Horizontal滤波器之外，还有其它常用的filters，例如Sobel filter和Scharr filter。这两种滤波器的特点是增加图片中心区域的权重。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Edge%20Detection%208.png)
上图展示的是垂直边缘检测算子，水平边缘检测算子只需将上图顺时针翻转90度即可。

在深度学习中，如果我们想检测图片的各种边缘特征，而不仅限于垂直边缘和水平边缘，那么filter的数值一般需要通过模型训练得到，类似于标准神经网络中的权重W一样由梯度下降算法反复迭代求得。CNN的主要目的就是计算出这些filter的数值。确定得到了这些filter后，CNN浅层网络也就实现了对图片所有边缘特征的检测。

## Padding

如果原始图片尺寸为n x n，filter尺寸为f x f，则卷积后的图片尺寸为(n-f+1) x (n-f+1)，注意f一般为奇数。

这样会带来两个问题：
卷积运算后，输出图片尺寸缩小
原始图片边缘信息对输出贡献得少，输出图片丢失边缘信息

为了解决图片缩小的问题，可以使用padding方法，即把原始图片尺寸进行扩展，扩展区域补零，用p来表示每个方向扩展的宽度
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Padding.png)

经过padding之后，原始图片尺寸为(n+2p) x (n+2p)，filter尺寸为f x f，则卷积后的图片尺寸为(n+2p-f+1) x (n+2p-f+1)。若要保证卷积前后图片尺寸不变，则p应满足：p = (f-1) / 2

Valid Convolutions：没有padding操作，p=0

Same Convolutions：有Padding操作，用p个像素填充边缘

## 卷积步长 Strided Convolutions

Stride表示filter在原图片中水平方向和垂直方向每次的步进长度。之前我们默认stride=1，若stride=2，则表示filter每次步进长度为2，即隔一点移动一次。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Strided%20Convolutions%201.png)

用s表示stride长度，p表示padding长度，如果原始图片尺寸为n x n，filter尺寸为f x f，则卷积后的图片尺寸为：
$$
\lfloor \frac{n+2p-f}{s}+1 \rfloor \times \lfloor \frac{n+2p-f}{s}+1 \rfloor
$$
互相关（cross-correlations）与卷积（convolutions）之间是有区别的。实际上，真正的卷积运算会先将filter绕其中心旋转180度，然后再将旋转后的filter在原始图片上进行滑动计算。filter旋转如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Strided%20Convolutions%202.png)
比较而言，互相关的计算过程则不会对filter进行旋转，而是直接在原始图片上进行滑动计算。

其实，目前为止所阐述的CNN卷积实际上计算的是互相关，而不是数学意义上的卷积。但是，为了简化计算，我们一般把CNN中的这种“相关系数”就称作卷积运算。之所以可以这么等效，是因为滤波器算子一般是水平或垂直对称的，180度旋转影响不大；而且最终滤波器算子需要通过CNN网络梯度下降算法计算得到，旋转部分可以看作是包含在CNN模型算法中。总的来说，忽略旋转运算可以大大提高CNN网络运算速度，而且不影响模型性能。

卷积运算服从结合律：( A ∗ B ) ∗ C = A ∗ ( B ∗ C ) 

##  三维卷积 Convolutions Over Volumes

对于3通道的RGB图片，其对应的滤波器算子同样也是3通道的。例如一个图片是6 x 6 x 3，分别表示图片的高度（height）、宽度（weight）和通道（channel）

3通道图片的卷积运算与单通道图片的卷积运算基本一致。过程是将每个单通道（R，G，B）与对应的filter进行卷积运算求和，然后再将3通道的和相加，得到输出图片的一个像素值
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Convolutions%20Over%20Volume%201.png)

不同通道的滤波算子可以不相同。例如R通道filter实现垂直边缘检测，G和B通道不进行边缘检测，全部置零，或者将R，G，B三通道filter全部设置为水平边缘检测。

为了进行多个卷积运算，实现更多边缘检测，可以增加更多的滤波器组。例如设置第一个滤波器组实现垂直边缘检测，第二个滤波器组实现水平边缘检测。这样，不同滤波器组卷积得到不同的输出，个数由滤波器组决定。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Convolutions%20Over%20Volume%202.png)

维度总结
$$
输入图像尺寸：n \times n \times n_c，filter尺寸：f \times f \times n_c，卷积后的图像尺寸为：(n-f+1) \times (n-f+1) \times n'_c
$$
其中，n_c为图像的通道数目，n'_c为滤波器组的个数，也可以理解为下一层的通道数

## 单层卷积网络 One Layer Of A Convolutional Network

卷积神经网络的单层结构如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/One%20Layer%20of%20a%20Convolutional%20Network.png)

相比之前的卷积过程，CNN的单层结构多了激活函数ReLU和偏移量b。整个过程与标准的神经网络单层结构非常类似：
$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}，A^{[l]}=g^{[l]}(Z^{[l]})
$$
卷积运算对应着上式中的乘积运算，滤波器组个数数值对应着权重W ^[ l ] ，所选的激活函数为ReLU。

计算一下上图中参数的数目：每个滤波器组有3x3x3=27个参数，还有1个偏移量b，则每个滤波器组有27+1=28个参数，两个滤波器组总共包含28x2=56个参数。我们发现，选定滤波器组后，参数数目与输入图片尺寸无关。所以，就不存在由于图片尺寸过大，造成参数过多的情况。例如一张1000x1000x3的图片，标准神经网络输入层的维度将达到3百万，而在CNN中，参数数目只由滤波器组决定，数目相对来说要少得多，这是CNN的优势之一。

CNN单层结构标记符号总结，设层数为 l
$$
f^{[l]}: filter\,size\quad p^{[l]}:padding \quad s^{[l]}:stride \quad n_c^{[l]}:number\,of\,filters
$$

$$
输入维度:n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}, \quad 
每个滤波器组维度:f^{[l]} \times f^{[l]} \times n_c^{[l-1]}
$$

$$
权重维度:f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}, \quad
输出维度:n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}
$$

$$
其中,\quad n_H^{[l]}=\lfloor \frac{n_H^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \rfloor, \quad
n_W^{[l]}=\lfloor \frac{n_W^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \rfloor, \quad
$$

$$
如果有m个样本，进行向量化运算，相应的输出维度为A^{[l]}=m \times n_H^{[l]} \times n_W^{[l]}\times n_c^{[l]}
$$

简单的CNN网络模型示例：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Simple%20Convolutional%20Network%20Example.png)
输出层可以是一个神经元，即二元分类（logistic）；也可以是多个神经元，即多元分类（softmax）。最后得到预测输出hat(y).

注意，随着CNN层数增加，图像的尺寸一般逐渐减小，而滤波器组个数逐渐增加。

CNN有三种类型的layer：
Convolution层（CONV）
Pooling层（POOL）
Fully Connected层（FC）

## 池化层 Pooling Layers

Pooling layers是CNN中用来减小尺寸，提高运算速度的，同样能减小noise影响，让各特征更具有健壮性。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Pooling%20Layers%201.png)
上图输入是一个 4×4 矩阵，用到的池化类型是最大池化（max pooling)。执行最大池化的树池是一个 2×2 矩阵。执行过程非常简单，把 4×4 的输入拆分成不同的区域，这个区域用不同颜色来标记。对于2×2 的输出，输出的每个元素都是其对应颜色区域中的最大元素值。
左上区域的最大值是 9，右上区域的最大元素值是 2，左下区域的最大值是 6，右下区域的最大值是 3。为了计算出右侧这 4 个元素值，我们需要对输入矩阵的 2×2 区域做最大值运算。这就像是应用了一个规模为 2 的过滤器，因为我们选用的是 2×2 区域，步幅是 2，这些就是最大池化的超参数。

Pooling layers的做法比convolution layers简单许多，没有卷积运算，仅仅是在滤波器算子滑动区域内取最大值，即max pooling，这是最常用的做法。注意，超参数p很少在pooling layers中使用。
Max pooling的好处是只保留区域内的最大值（特征），忽略其它值，降低noise影响，提高模型健壮性。而且，max pooling需要的超参数仅为滤波器尺寸f和滤波器步进长度s，没有其他参数需要模型训练得到，计算量很小。
如果是多个通道，那么就每个通道单独进行max pooling操作。

所以最大化运算的实际作用就是，如果在过滤器中提取到某个特征，那么保留其最大值。如果没有提取到这个特征，可能在右上象限中不存在这个特征，那么其中的最大值也还是很小，这就是最大池化的直观理解。

除了max pooling之外，还有一种做法：average pooling。顾名思义，average pooling就是在滤波器算子滑动区域计算平均值。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Pooling%20Layers%202.png)
实际应用中，max pooling比average pooling更为常用。

## CNN Example

一个数字识别的CNN示例：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/CNN%20Example%201.png)
图中，CON层后面紧接一个POOL层，CONV1和POOL1构成第一层，CONV2和POOL2构成第二层。特别注意的是FC3和FC4为全连接层FC，它跟标准的神经网络结构一致。最后的输出层（softmax)由10个神经元构成。

整个网络各层的尺寸和参数如下表格所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/CNN%20Example%202.png)

相比标准神经网络，CNN的优势之一就是参数数目要少得多。参数数目少的原因有两个：
参数共享：一个特征检测器（例如垂直边缘检测）对图片某块区域有用，同时也可能作用在图片其它区域。
连接的稀疏性：因为滤波器算子尺寸限制，每一层的每个输出只与输入部分区域内有关。

除此之外，由于CNN参数数目较小，所需的训练样本就相对较少，从而一定程度上不容易发生过拟合现象。而且，CNN比较擅长捕捉区域位置偏移。也就是说CNN进行物体检测时，不太受物体所处图片位置的影响，增加检测的准确性和系统的健壮性。

## 经典网络 Classic Networks

LeNet-5模型是Yann LeCun教授于1998年提出来的，它是第一个成功应用于数字识别问题的卷积神经网络。在MNIST数据中，它的准确率达到大约99.2%。典型的LeNet-5结构包含CONV layer，POOL layer和FC layer，顺序一般是CONV layer->POOL layer->CONV layer->POOL layer->FC layer->FC layer->OUTPUT layer。下图所示的是一个数字识别的LeNet-5的模型结构：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Classic%20Networks%201.png)
该LeNet模型总共包含了大约6万个参数。值得一提的是，当时Yann LeCun提出的LeNet-5模型池化层使用的是average pool，而且各层激活函数一般是Sigmoid和tanh。现在，我们可以根据需要，做出改进，使用max pool和激活函数ReLU。

AlexNet模型是由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton共同提出的，其结构如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Classic%20Networks%202.png)
AlexNet模型与LeNet-5模型类似，只是要复杂一些，总共包含了大约6千万个参数。同样可以根据实际情况使用激活函数ReLU。原作者还提到了一种优化技巧，叫做Local Response Normalization(LRN)。 而在实际应用中，LRN的效果并不突出

VGG-16模型更加复杂一些，一般情况下，其CONV layer和POOL layer设置如下:
CONV = 3x3 filters, s = 1, same
MAX-POOL = 2x2, s = 2

VGG-16 网络没有那么多超参数，这是一种只需要专注于构建卷积层的简单网络。首先用 3×3，步幅为 1 的过滤器构建卷积层，padding 参数为 same 卷积中的参数。然后用一个 2×2，步幅为 2 的过滤器构建最大池化层
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Classic%20Networks%203.png)
要识上图，在最开始的两层用 64 个 3×3 的过滤器对输入图像进行卷积，输出结果是 224×224×64，因为使用了 same 卷积，通道数量也一样；接下来创建一个池化层，池化层将输入图像进行压缩，减少到 112×112×64。然后又是若干个卷积层，使用 128 个过滤器，以及一些 same卷积，112×112×128.然后进行池化，可以推导出池化后的结果是56×56×128；接着再用 256 个相同的过滤器进行三次卷积操作，然后再池化，然后再卷积三次，再池化。如此进行几轮操作后，将最后得到的 7×7×512 的特征图进行全连接操作，得到 4096 个单元，然后进行 softmax 激活，输出从 1000 个对象中识别的结果。VGG-16的参数多达1亿3千万。

## 残差网络 Residual Networks

经网络层数越多，网络越深，源于梯度消失和梯度爆炸的影响，整个模型难以训练成功。解决的方法之一是人为地让神经网络某些层跳过下一层神经元的连接，隔层相连，弱化每层之间的强联系。这种神经网络被称为Residual Networks(ResNets)。

Residual Networks由许多隔层相连的神经元子模块组成，我们称之为Residual block。单个Residual block的结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Residual%20Networks%201.png)
上图中红色部分就是skip connection，直接建立a^[l]与a^[l+2]之间的隔层联系。相应的表达式如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Residual%20Networks%202.png)
直接隔层与下一层的线性输出相连，与共同通过激活函数ReLU输出。

由多个Residual block组成的神经网络就是Residual Network。实验表明，这种模型结构对于训练非常深的神经网络，效果很好。另外，为了便于区分，我们把非Residual Networks称为Plain Network。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Residual%20Networks%203.png)
Residual Network的结构如上图所示

与Plain Network相比，Residual Network能够训练更深层的神经网络，有效避免发生发生梯度消失和梯度爆炸。从下面两张图的对比中可以看出，随着神经网络层数增加，Plain Network实际性能会变差，training error甚至会变大。然而，Residual Network的训练效果却很好，training error一直呈下降趋势。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Residual%20Networks%204.png)

下面用个例子来解释为什么ResNets能够训练更深层的神经网络。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Residual%20Networks%205.png)

如上图所示，输入x经过很多层神经网络后输出，其中的激活函数为ReLU，经过一个Residual block输出：
$$
a^{[l+2]}=g(z^{[l+2]}+a^{[l]})=g(W^{[l+2]}a^{[l+1]}+b^{[l+2]}+a^{[l]})
$$

$$
输入x经过Big\,NN后，若W^{[l+2]}=b^{[l+2]}=0，则有a^{[l+2]}=g(a^{[l]})=a^{[l]}，when\,a^{[l]}\geq 0
$$

可以看出，即使发生了梯度消失，也能直接建立a^[l+2]与a^[l]的线性关系。从效果来说，相当于直接忽略了Big NN之后的这两层神经层。这样，看似很深的神经网络，其实由于许多Residual blocks的存在，弱化削减了某些神经层之间的联系，实现隔层线性传递，而不是一味追求非线性关系，模型本身也就能“容忍”更深层的神经网络了。而且从性能上来说，这两层额外的Residual blocks也不会降低Big NN的性能。

当然，如果Residual blocks确实能训练得到非线性关系，那么也会忽略short cut，跟Plain Network起到同样的效果。

有一点需要注意的是，如果Residual blocks中输入和输出的维度不同，通常可以引入矩阵W_s，W_s与a^[l]相乘，使得的维度与a^[l+2]一致。参数矩阵有来两种方法得到：一种通过模型训练得到；另一种是固定值（类似单位矩阵），padding的值为0，用0填充a^[l]，使其维度与a^[l+2]一致。

下图所示的是CNN中ResNets的结构：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Residual%20Networks%206.png)
ResNets同类型层之间，例如CONV layers，大多使用same类型，保持维度相同。如果是不同类型层之间的连接，例如CONV layer与POOL layer之间，如果维度不同，则引入矩阵W_s。

普通网络和 ResNets 网络常用的结构是：卷积层-卷积层-卷积层-池化层-卷积层-卷积层-卷积层-池化层……依此重复。直到最后，有一个通过 softmax 进行预测的全连接层。

## 网络中的网络以及 1×1 卷积 Network In Network And 1×1 Convolutions

1x1 Convolutions，也称Networks in Networks。这种结构的特点是滤波器算子filter的维度为1x1。对于单个filter，1x1的维度，意味着卷积操作等同于乘积操作。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Networks%20in%20Networks%201.png)

那么，对于多个filters，1x1 Convolutions的作用实际上类似全连接层的神经网络结构。效果等同于Plain Network中到的过程。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Networks%20in%20Networks%202.png)

1x1 Convolutions可以用来缩减输入图片的通道数目。方法如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Networks%20in%20Networks%203.png)

## Inception网络 Inception Network

CNN单层的滤波算子filter尺寸是固定的，1x1或者3x3等。而Inception Network在单层网络上可以使用多个不同尺寸的filters，进行same convolutions，把各filter下得到的输出拼接起来。除此之外，还可以将CONV layer与POOL layer混合，同时实现各种效果。但是要注意使用same pool。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Inception%20Network%201.png)

Inception Network与其它只选择单一尺寸和功能的filter不同，Inception Network使用不同尺寸的filters并将CONV和POOL混合起来，将所有功能输出组合拼接，再由神经网络本身去学习参数并选择最好的模块。

Inception Network在提升性能的同时，会带来计算量大的问题。例如下面这个例子：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Inception%20Network%202.png)
此CONV layer需要的计算量为：28x28x32x5x5x192=120m，其中m表示百万单位，可以看出但这一层的计算量都是很大的。为此，我们可以引入1x1 Convolutions来减少其计算量，结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Inception%20Network%203.png)
通常我们把该1x1 Convolution称为“瓶颈层”（bottleneck layer)。引入bottleneck layer之后，总共需要的计算量为：28x28x16x192+28x28x32x5x5x16=12.4m。明显地，虽然多引入了1x1 Convolution层，但是总共的计算量减少了近90%，效果还是非常明显的。由此可见，1x1 Convolutions还可以有效减少CONV layer的计算量。

引入1x1 Convolution后的Inception module如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Inception%20Network%204.png)

多个Inception modules组成Inception Network，效果如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Inception%20Network%205.png)
上述Inception Network除了由许多Inception modules组成之外，值得一提的是网络中间隐藏层也可以作为输出层Softmax，有利于防止发生过拟合。

## 数据增强 Data Augmentation

常用的Data Augmentation方法是对已有的样本集进行Mirroring和Random Cropping
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Data%20Augmentation%201.png)

另一种Data Augmentation的方法是color shifting。color shifting就是对图片的RGB通道数值进行随意增加或者减少，改变图片色调。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Data%20Augmentation%202.png)

除了随意改变RGB通道数值外，还可以更有针对性地对图片的RGB通道进行PCA color augmentation，也就是对图片颜色进行主成分分析，对主要的通道颜色进行增加或减少，可以采用高斯扰动做法。这样也能增加有效的样本数量。在构建大型神经网络的时候，data augmentation和training可以由两个不同的线程来进行。

# 目标检测 Object Detection

## 目标定位 Object Localization

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Object%20Localization%201.png)

标准的CNN分类模型，如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Object%20Localization%202.png)
原始图片经过CONV卷积层后，Softmax层输出4 x 1向量，分别是：pedestrain，car，motorcycle和background

对于目标定位和目标检测问题，其模型如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Object%20Localization%203.png)
原始图片经过CONV卷积层后，Softmax层输出8 x 1向量。p_c表示是否含有对象，如果对象属于前三类（行人、汽车、摩托车），其值为1，如果是背景，即图片中没有要检测的对象，其值为0，实际上，它表示矩形区域是目标的概率，数值在0～1之间，且值越大概率越大；(bx, by)，表示目标中心位置坐标，b_h和b_w，表示目标所在矩形区域的高和宽。一般设定图片左上角为原点(0, 0)，右下角为(1, 1)。在模型训练时，b_x、b_y、b_h、b_w都由人为确定其数值。例如上图中，可得b_x=0.5，b_y=0.7，b_h=0.3，b_w=0.4。

对于损失函数Loss function，若使用平方误差形式，有两种情况：
$$
p_c=1,即取y_1=1,L(\hat{y},y)=(\hat{y}_1-y_1)^2+(\hat{y}_2-y_2)^2+……+(\hat{y}_8-y_8)^2
$$

$$
p_c=0,即取y_1=0,表示没有检测到目标，则输出label后面的7个参数都可以忽略,L(\hat{y},y)=(\hat{y}_1-y_1)^2
$$

## 特征点检测 Landmark Detection

除了使用矩形区域检测目标类别和位置外，我们还可以仅对目标的关键特征点坐标进行定位，这些关键点被称为landmarks。

例如人脸识别，可以对人脸部分特征点坐标进行定位检测，并标记出来，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Landmark%20Detection%201.png)

该网络模型共检测人脸上64处特征点，加上是否为face的标志位，输出label共有64x2+1=129个值。通过检测人脸特征点可以进行情绪分类与判断，或者应用于AR领域等等。

除了人脸特征点检测之外，还可以检测人体姿势动作，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Landmark%20Detection%202.png)

## 目标检测 Object Detection

目标检测的一种简单方法是滑动窗算法。这种算法首先在训练样本集上搜集相应的各种目标图片和非目标图片。注意训练集图片尺寸较小，尽量仅包含相应目标，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Object%20Detection%201.png)
然后，使用这些训练集构建CNN模型，使得模型有较高的识别率

最后，在测试图片上，选择大小适宜的窗口、合适的步进长度，进行从左到右、从上倒下的滑动。每个窗口区域都送入之前构建好的CNN模型进行识别判断。若判断有目标，则此窗口即为目标区域；若判断没有目标，则此窗口为非目标区域。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Object%20Detection%202.png)

滑动窗算法的优点是原理简单，且不需要人为选定目标区域（检测出目标的滑动窗即为目标区域）。但是其缺点也很明显，首先滑动窗的大小和步进长度都需要人为直观设定。滑动窗过小或过大，步进长度过大均会降低目标检测正确率。而且，每次滑动窗区域都要进行一次CNN网络计算，如果滑动窗和步进长度较小，整个目标检测的算法运行时间会很长。所以，滑动窗算法虽然简单，但是性能不佳，不够快，不够灵活。

## 滑动窗口的卷积实现 Convolutional Implementation of Sliding Windows

滑动窗算法可以使用卷积方式实现，以提高运行速度，节约重复运算成本。

首先，单个滑动窗口区域进入CNN网络模型时，包含全连接层。那么滑动窗口算法卷积实现的第一步就是将全连接层转变成为卷积层，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Convolutional%20Implementation%20of%20Sliding%20Windows%201.png)
全连接层转变成卷积层的操作很简单，只需要使用与上层尺寸一致的滤波算子进行卷积运算即可。最终得到的输出层维度是1 x 1 x 4，代表4类输出值。

单个窗口区域卷积网络结构建立完毕之后，对于待检测图片，即可使用该网络参数和结构进行运算。例如16 x 16 x 3的图片，单通道内窗口大小为14 x 14，滑动步进长度为2，则16 x 16的通道内可以划分4个滑动窗口，CNN网络得到的输出层为2 x 2 x 4；其中，2 x 2表示共有4个窗口结果，左上角的蓝色对应16 x 16图像中蓝色区域代表的窗口，右上角对应16 x 16中蓝色区域向右滑动2对应的区域。
对于更复杂的28 x 28 x3的图片，CNN网络得到的输出层为8 x 8 x 4，共64个窗口结果。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Convolutional%20Implementation%20of%20Sliding%20Windows%202.png)

之前的滑动窗算法需要反复进行CNN正向计算，例如16 x 16 x 3的图片需进行4次，28 x 28 x3的图片需进行64次。而利用卷积操作代替滑动窗算法，则不管原始图片有多大，只需要进行一次CNN正向计算，因为其中共享了很多重复计算部分，这大大节约了运算成本。值得一提的是，窗口步进长度与选择的MAX POOL大小有关，如果需要步进长度为4，只需设置MAX POOL为4 x 4即可。

## Bounding Box Predictions

滑动窗口算法有时会出现滑动窗不能完全涵盖目标的问题，如下图蓝色窗口所示:
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Bounding%20Box%20Predictions%201.png)

YOLO（You Only Look Once）算法可以解决这类问题，生成更加准确的目标区域（如上图红色窗口）。

YOLO算法首先将原始图片分割成n x n网格，每个网格代表一块区域。为简化说明，下图中将图片分成3 x 3网格。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Bounding%20Box%20Predictions%202.png)
然后，利用上一节卷积形式实现滑动窗口算法的思想，对该原始图片构建CNN网络，得到的的输出层维度为3 x 3 x 8。其中，3 x 3对应9个网格，每个网格的输出包含8个元素：
$$
y=
\begin{bmatrix}
    p_c\\ b_x\\ b_y\\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3
\end{bmatrix}
$$

如果目标中心坐标不在当前网格内，则当前网格Pc=0；相反，则当前网格Pc=1（即只看中心坐标是否在当前网格内）。判断有目标的网格中，限定了目标区域。值得注意的是，当前网格左上角坐标设定为(0, 0)，右下角坐标设定为(1, 1)，范围限定在[0,1]之间，但是可以大于1。因为目标可能超出该网格，横跨多个区域，如上图所示。目标占几个网格没有关系，目标中心坐标必然在一个网格之内。
划分的网格可以更密一些，网格越小，则多个目标的中心坐标被划分到一个网格内的概率就越小，这恰恰是我们希望看到的。

## 交并比 Intersection Over Union

IoU，即交集与并集之比，可以用来评价目标检测区域的准确性。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Intersection%20Over%20Union.png)
如上图所示，红色方框为真实目标区域，蓝色方框为检测目标区域。两块区域的交集为绿色部分，并集为紫色部分。蓝色方框与红色方框的接近程度可以用IoU比值来定义：
$$
IoU=\frac{Intersection}{Union}=\frac{A \cap B}{A \cup B}
$$
一般约定，在计算机检测任务中，如果 IoU ≥ 0.5，就说检测正确，如果预测器和实际边界框完美重叠，loU 就是 1，因为交集就等于并集。但一般来说只要 IoU ≥ 0.5，那么结果是可以接受的，看起来还可以。一般约定，0.5 是阈值，用来判断预测的边界框是否正确。

## 非最大值抑制 Non-Max Suppression

YOLO算法中，可能会出现多个网格都检测出到同一目标的情况，例如几个相邻网格都判断出同一目标的中心坐标在其内。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Non-max%20Suppression%201.png)
上图中，三个绿色网格和三个红色网格分别检测的都是同一目标。那如何判断哪个网格最为准确呢？方法是使用非最大值抑制算法。

非最大值抑制（Non-max Suppression）做法很简单，图示每个网格的Pc值可以求出，Pc值反映了该网格包含目标中心坐标的可信度。首先选取Pc最大值对应的网格和区域，然后计算该区域与所有其它区域的IoU，剔除掉IoU大于阈值（例如0.5）的所有网格及区域。这样就能保证同一目标只有一个网格与之对应，且该网格Pc最大，最可信。接着，再从剩下的网格中选取Pc最大的网格，重复上一步的操作。最后，就能使得每个目标都仅由一个网格和区域对应。如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Non-max%20Suppression%202.png)

非最大值抑制算法的流程：
1、剔除Pc值小于某阈值（例如0.6）的所有网格；
2、选取Pc值最大的网格，利用IoU，摒弃与该网格交叠较大的网格；
3、对剩下的网格，重复步骤2。

## Anchor Boxes

对于多个目标重叠的情况，例如一个人站在一辆车前面，使用不同形状的Anchor Boxes再运用YOLO算法进行检测。

如下图所示，同一网格出现了两个目标：人和车。为了同时检测两个目标，我们可以设置两个Anchor Boxes，Anchor box 1检测人，Anchor box 2检测车。也就是说，每个网格多加了一层输出。原来的输出维度是 3 x 3 x 8，现在是3 x 3 x 2 x 8（也可以写成3 x 3 x 16的形式）。这里的2表示有两个Anchor Boxes，用来在一个网格中同时检测多个目标。每个Anchor box都有一个Pc值，若两个Pc值均大于某阈值，则检测到了两个目标
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Anchor%20Boxes.png)
$$
y= [p_c\quad b_x\quad b_y\quad b_h\quad b_w\quad c_1\quad c_2\quad c_3\quad p_c\quad b_x\quad b_y\quad b_h\quad b_w\quad c_1\quad c_2\quad c_3]^T
$$
在使用YOLO算法时，只需对每个Anchor box使用非最大值抑制即可，Anchor Boxes之间并行实现。

Anchor Boxes形状的选择可以通过人为选取，也可以使用其他机器学习算法，例如k聚类算法对待检测的所有目标进行形状分类，选择主要形状作为Anchor Boxes。

## 候选区域网络 Region Proposals Network

滑动窗算法会对原始图片的每个区域都进行扫描，即使是一些空白的或明显没有目标的区域，例如下图所示。这样会降低算法运行效率，耗费时间。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Region%20Proposals%201.png)

为了解决这一问题，尽量避免对无用区域的扫描，可以使用Region Proposals的方法。具体做法是先对原始图片进行分割算法处理，然后支队分割后的图片中的块进行目标检测。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Region%20Proposals%202.png)

Region Proposals共有三种方法：
1、R-CNN: 滑动窗的形式，一次只对单个区域块进行目标检测，运算速度慢。
2、Fast R-CNN: 利用卷积实现滑动窗算法。
3、Faster R-CNN: 利用卷积对图片进行分割，进一步提高运行速度。

RPN全称是Region Proposal Network，Region
Proposal的中文意思是“区域选取”，也就是“提取候选框”的意思，所以RPN就是用来提取候选框的网络

RPN的引入，可以说是真正意义上把物体检测整个流程融入到一个神经网络中，这个网络结构叫做Faster R-CNN；
Faster R-CNN = RPN + Fast RCNN

Faster R-CNN的整体结构如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/RPN%201.png)

RPN的结构如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/RPN%202.png)

Faster R-CNN和RPN的结构关系如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/RPN%203.png)

关于Anchor的问题解释如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/RPN%204.png)

RPN整个流程：首先通过一系列卷积得到公共特征图，假设他的大小是N x 16 x 16，然后我们进入RPN阶段，首先经过一个3 x 3的卷积，得到一个256 x 16 x 16的特征图，也可以看作16 x 16个256维特征向量，然后经过两次1 x 1的卷积，分别得到一个18 x 16 x 16的特征图，和一个36 x 16 x 16的特征图，也就是16 x 16 x 9个结果，每个结果包含2个分数和4个坐标，再结合预先定义的Anchors，经过后处理，就得到候选框
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/RPN%205.png)

## YOLO Algorithm Summarize

网络结构如下图所示，包含了两个Anchor Boxes。

1. For each grid call, get 2 predicted bounding boxes.
2. Get rid of low probability predictions.
3. For each class (pedestrian, car, motorcycle) use non-max suppression to generate final predictions.

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/YOLO%20Algorithm.png)



# 人脸识别 Face Recognition

人脸验证 Face Verification：输入一张人脸图片，验证输出与模板是否为同一人，即一对一问题。

人脸识别 Face Recognition：输入一张人脸图片，验证输出是否为K个模板中的某一个，即一对多问题。

一般地，人脸识别比人脸验证更难一些。因为假设人脸验证系统的错误率是1%，那么在人脸识别中，输出分别与K个模板都进行比较，则相应的错误率就会增加，约K%。模板个数越多，错误率越大一些。

## One-Shot Learning

One-shot learning意味着数据库中每个人的训练样本只包含一张照片，然后训练一个CNN模型来进行人脸识别。若数据库有K个人，则CNN模型输出softmax层就是K+1维的（K个人中的一个，或者都不符合）。

但是One-shot learning的性能并不好，其包含了两个缺点：
每个人只有一张图片，训练样本少，构建的CNN网络不够健壮
若数据库增加另一个人，输出层softmax的维度就要发生变化，相当于要重新构建CNN网络，使模型计算量大大增加，不够灵活

为了解决One-shot learning的问题，先引入相似函数（similarity function）。相似函数表示两张图片的相似程度，用d(img1,img2) = degree of difference between images 来表示。若d(img1,img2)较小，则表示两张图片相似；若d(img1,img2)较大，则表示两张图片不是同一个人。相似函数可以在人脸验证中使用：
$$
d(img1,img2)\leq \tau ：两张图片是同一人 \quad
d(img1,img2) > \tau ：两张图片不是同一人
$$
对于人脸识别问题，则只需计算测试图片与数据库中K个目标的相似函数，取其中d(img1,img2)最小的目标为匹配对象。若所有的d(img1,img2)都很大，则表示数据库没有这个人。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/One%20Shot%20Learning.png)

## Siamese Network

若一张图片经过一般的CNN网络（包括CONV层、POOL层、FC层），最终得到全连接层FC，该FC层可以看成是原始图片的编码encoding，表征了原始图片的关键特征。这个网络结构我们称之为Siamese network。也就是说每张图片经过Siamese network后，由FC层每个神经元来表征。

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Siamese%20Network.png)

建立Siamese network后，两张图片x(1)和x(2)的相似度函数可由各自FC层f(x(1))与f(x(2))之差的范数来表示：
$$
d(x^{(1)},x^{(2)})=||f(x^{(1))})-f(x^{(2))})||^2
$$

$$
If\;x^{(i)},x^{(j)}\;are\;the\;same\;person,||f(x^{(i))})-f(x^{(j))})||^2\;is\;small
$$

$$
If\;x^{(i)},x^{(j)}\;are\;different\;persons,||f(x^{(i))})-f(x^{(j))})||^2\;is\;large
$$

## Triplet Loss

构建人脸识别的CNN模型，需要定义合适的损失函数，引入Triplet Loss。

Triplet Loss需要每个样本包含三张图片：靶目标（Anchor）、正例（Positive）、反例（Negative），这就是triplet名称的由来。顾名思义，靶目标和正例是同一人，靶目标和反例不是同一人。Anchor和Positive组成一类样本，Anchor和Negative组成另外一类样本。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Triplet%20Loss%201.png)

把三张图片Anchor、Positive、Negative 图片简写成A、P、N。
我们希望构建的CNN网络输出编码f(A)接近f(P)，即||f(A)−f(P)||^2尽可能小，而||f(A)−f(N)||^2尽可能大，数学上满足：
$$
||f(A)-f(P)||^2\leq||f(A)-f(N)||^2,\quad即d(A,P)\leq d(A,N)
$$
根据上面的不等式，如果所有的图片都是零向量，即f(A)=0,f(P)=0,f(N)=0，那么上述不等式也满足。但是这对我们进行人脸识别没有任何作用，是不希望看到的。
我们希望得到||f(A)−f(P)||^2远小于||f(A)−f(N)||^2，所以，添加一个超参数α，且α>0，对上述不等式做出如下修改：
$$
||f(A)-f(P)||^2 -||f(A)-f(N)||^2\leq-\alpha,\quad即||f(A)-f(P)||^2 -||f(A)-f(N)||^2+\alpha\leq0
$$
超参数α也被称为边界margin，它拉大了 Anchor 和 Positive 图片对和 Anchor 与 Negative 图片对之间的差距，类似与支持向量机中的margin。例如，若d(A,P)=0.5，α=0.2，则d(A,N) ≥ 0.7。

根据A，P，N三张图片，就可以定义Loss function为：
$$
L(A,P,N)=max(||f(A)-f(P)||^2 -||f(A)-f(N)||^2+\alpha,0)
$$
相应地，对于m组训练样本，Cost function为：
$$
J=\sum_{i=1}^mL(A^{(i)},P^{(i)},N^{(i)})
$$
关于训练样本，必须保证同一人包含多张照片，否则无法使用这种方法。例如10k张照片包含1k个不同的人脸，则平均一个人包含10张照片，这个训练样本是满足要求的。然后，就可以使用梯度下降算法，不断训练优化CNN网络参数，让Cost Function不断减小接近0。

同一组训练样本，A，P，N的选择尽可能不要使用随机选取方法。因为随机选择的A与P一般比较接近，A与N相差也较大，毕竟是两个不同人脸。这样的话，也许模型不需要经过复杂训练就能实现这种明显识别，但是抓不住关键区别。所以，最好的做法是人为选择A与P相差较大（例如换发型，留胡须等），A与N相差较小（例如发型一致，肤色一致等）。这种人为地增加难度和混淆度会让模型本身去寻找学习不同人脸之间关键的差异，“尽力”让d(A,P)更小，让d(A,N)更大，即让模型性能更好。

一些A，P，N的例子：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Triplet%20Loss%202.png)

值得一提的是，现在许多商业公司构建的大型人脸识别模型都需要百万级别甚至上亿的训练样本。如此之大的训练样本我们一般很难获取。但是一些公司将他们训练的人脸识别模型发布在了网上，可供我们使用。

## 人脸验证与二分类 Face Verification And Binary Classification

除了构造Triplet loss来解决人脸识别问题之外，还可以使用二分类结构。
做法是将两个siamese网络组合在一起，将各自的编码层输出经过一个逻辑输出单元，该神经元使用sigmoid函数，输出1则表示识别为同一人，输出0则表示识别为不同人。结构示意图如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Face%20Verification%20and%20Binary%20Classification%201.png)

每组训练样本包含两张图片，每个siamese网络结构和参数完全相同。这样就把人脸识别问题转化成了一个二分类问题。引入逻辑输出层参数w和b，输出ŷ 表达式为：
$$
\widehat{y}=\sigma(\sum_{k=1}^Kw_k|f(x^{(i)})_k-f(x^{(j)})_k|+b)
$$
符号解释：𝑓(𝑥(𝑖))𝑘代表图片𝑥(𝑖)的编码，下标𝑘代表选择这个向量中的第𝑘个元素，|𝑓(𝑥(𝑖))𝑘 − 𝑓(𝑥(𝑗))𝑘|对这两个编码取元素差的绝对值，其中参数wk和b都是通过梯度下降算法迭代训练得到。

ŷ的另一种表达形式
$$
\widehat{y}=\sigma(\sum_{k=1}^Kw_k\frac{(f(x^{(i)})_k-f(x^{(j)})_k)^2}{f(x^{(i)})_k+f(x^{(j)})_k}+b)，分式部分被称为𝜒2公式，也被称为𝜒平方相似度
$$

在训练好网络之后，进行人脸识别的常规方法是测试图片与模板分别进行网络计算，编码层输出比较，计算逻辑输出单元。为了减少计算量，可以使用预计算的方式在训练时就将数据库每个模板的编码层输出f(x)保存下来。因为编码层输出f(x)比原始图片数据量少很多，所以无须保存模板图片，只要保存每个模板的f(x)即可，节约存储空间。而且，测试过程中，无须计算模板的siamese网络，只要计算测试图片的siamese网络，得到的f(x(i))直接与存储的模板f(x(j))进行下一步的逻辑输出单元计算即可，计算时间减小了接近一半。这种方法也可以应用在Triplet loss网络中。

总结：把人脸验证当作一个监督学习，创建一个只有成对图片的训练集，不是三个一组，而是成对的图片，目标标签是 1 表示一对图片是一个人，目标标签是 0 表示图片中是不同的人。利用不同的成对图片，使用反向传播算法去训练神经网络，训练 Siamese神经网络。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Face%20Verification%20And%20Binary%20Classification%202.png)



# 神经风格迁移 Neural Style Transfer

神经风格迁移是CNN模型一个非常有趣的应用。它可以实现将一张图片的风格“迁移”到另外一张图片中，生成具有其特色的图片。比如我们可以将毕加索的绘画风格迁移到我们自己做的图中，生成类似的“大师作品”。

下面列出几个神经风格迁移的例子：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Style%20Transfer%201.png)
一般用C表示内容图片，S表示风格图片，G表示生成的图片。

## 深度卷积网络学习什么 What Are Deep ConvNets Learning

在进行神经风格迁移之前，我们先来从可视化的角度看一下卷积神经网络每一层到底是什么样子？它们各自学习了哪些东西。

典型的CNN网络如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/What%20Are%20Deep%20ConvNets%20Learning%201.png)

首先来看第一层隐藏层，遍历所有训练样本，找出让该层激活函数输出最大的9块图像区域；然后再找出该层的其它单元（不同的滤波器通道）激活函数输出最大的9块图像区域；最后共找9次，得到9 x 9的图像如下所示，其中每个3 x 3区域表示一个运算单元
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/What%20Are%20Deep%20ConvNets%20Learning%202.png)

可以看出，第一层隐藏层一般检测的是原始图像的边缘和颜色阴影等简单信息。
继续看CNN的更深隐藏层，随着层数的增加，捕捉的区域更大，特征更加复杂，从边缘到纹理再到具体物体。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/What%20Are%20Deep%20ConvNets%20Learning%203.png)

## 代价函数 Cost Function

神经风格迁移生成图片G的Cost function由两部分组成：C与G的相似程度和S与G的相似程度。
$$
J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)
$$
其中，α，β是超参数，用来调整Jcontent(C,G)与Jstyle(S,G)的相对比重
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Style%20Transfer%20Cost%20Function%201.png)

神经风格迁移的基本算法流程是：首先令G为随机像素点，然后使用梯度下降算法，不断修正G的所有像素点，使得J(G)不断减小，从而使G逐渐有C的内容和G的风格，如下图所示。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Neural%20Style%20Transfer%20Cost%20Function%202.png)

## Content Cost Function

J_content(C,G)表示内容图片C与生成图片G之间的相似度。

使用的CNN网络是之前训练好的模型，例如Alex-Net。C，S，G共用相同模型和参数。首先，需要选择合适的层数 l 计算J_content(C,G)。CNN的每个隐藏层分别提取原始图片的不同深度特征，由简单到复杂。如果 l 太小，则G与C在像素上会非常接近，没有迁移效果；如果 l 太深，则G上某个区域将直接会出现C中的物体。因此，l 既不能太浅也不能太深，一般选择网络中间层。
$$
J_{content}(C,G)=\frac{1}{2}||a^{[l][C]}-a^{[l][G]}||^2
$$
该代价函数比较的是C和G在l层的激活函数输出，方法就是使用梯度下降算法，不断迭代修正G的像素值，使代价函数减小。

## Style Cost Function

什么是图片的风格？利用CNN网络模型，图片的风格可以定义成第l层隐藏层不同通道间激活函数的乘积（相关性）。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Style%20Cost%20Function%201.png)

例如我们选取第 l 层隐藏层，其各通道使用不同颜色标注，如下图所示。因为每个通道提取图片的特征不同，比如1通道（红色）提取的是图片的垂直纹理特征，2通道（黄色）提取的是图片的橙色背景特征。那么计算这两个通道的相关性大小，相关性越大，表示原始图片及既包含了垂直纹理也包含了该橙色背景；相关性越小，表示原始图片并没有同时包含这两个特征。也就是说，计算不同通道的相关性，反映了原始图片特征间的相互关系，从某种程度上刻画了图片的“风格”。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Style%20Cost%20Function%202.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Style%20Cost%20Function.png)



# 循环神经网络 Recurrent Neural Networks

## 序列模型 Sequence Models

序列模型能够应用在许多领域，例如：语音识别、音乐发生器、情感分类、DNA序列分析、机器翻译、视频动作识别、命名实体识别
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20Models%201.png)
这些序列模型基本都属于监督式学习，输入x和输出y不一定都是序列模型。如果都是序列模型的话，模型长度不一定完全一致

下面以命名实体识别为例，介绍序列模型的命名规则。示例语句为：

Harry Potter and Hermione Granger invented a new spell.

该句话包含9个单词，输出y即为1 x 9向量，每位表征对应单词是否为人名的一部分，1表示是，0表示否。很明显，该句话中“Harry”，“Potter”，“Hermione”，“Granger”均是人名成分，所以，对应的输出y可表示为：y=[1  1  0  1  1  0  0  0  0]
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20Models%202.png)

## 循环神经网络模型 Recurrent Neural Network Model

对于序列模型，如果使用标准的神经网络，其模型结构如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Recurrent%20Neural%20Network%20Model%201.png)

使用标准的神经网络模型存在两个问题：
1、不同样本的输入序列长度或输出序列长度不同，即T_x和T_y可能不同，造成模型难以统一。解决办法之一是设定一个最大序列长度，对每个输入和输出序列补零并统一到最大长度，但是这种做法实际效果并不理想。
2、这种标准神经网络结构无法共享从文本的不同位置上学到的特征。例如，如果某个位置识别到“Harry”是人名成分，那么句子其它位置出现了“Harry”也能自动识别其为人名的一部分，这是共享特征的结果，如同CNN网络特点一样。但是，上图所示的网络不具备共享特征的能力。值得一提的是，共享特征还有助于减少神经网络中的参数数量，一定程度上减小了模型的计算复杂度。例如上图所示的标准神经网络，假设每个扩展到最大序列长度为100，且词汇表长度为10000，则输入层就已经包含了100 x 10000个神经元了，权重参数很多，运算量将是庞大的。

标准的神经网络不适合解决序列模型问题，而循环神经网络（RNN）是专门用来解决序列模型问题的。RNN模型结构如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Recurrent%20Neural%20Network%20Model%202.png)
序列模型从左到右，依次传递，由输入序列生成输出序列，同时也要考虑前期时间步的结果，以实现特征共享。在这个模型中 T_x=T_y，如果它们不同，模型要做出适当改变。

循环神经网络是从左向右扫描数据，同时每个时间步的参数也是共享的。要开始整个流程，需要在零时刻构造一个激活值a^<0>，它通常是零向量；用W_ax来表示管理着从x^<1>到隐藏层的连接的一系列参数，每个时间步使用的都是相同的参数W_ax；而激活值也就是水平联系是由参数W_𝑎𝑎决定的，同时每一个时间步都使用相同的参数，输出结果由W\_ya决定。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Recurrent%20Neural%20Network%20Model%203.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Recurrent%20Neural%20Network%20Model%204.png)

RNN的正向传播（Forward Propagation）过程为（激活函数g的选择依模型而定）：
$$
a^{<t>}=g_1(W_{aa}a^{<t-1>}+W_{ax}x^{<t>}+b_a),\quad \widehat{y}^{<t>}=g_2(W_{ya}a^{<t>}+b_y)
$$
为了简化表达式:
$$
W_{aa}a^{<t-1>}+W_{ax}x^{<t>}=[W_{aa}\;W_{ax}]
\begin{bmatrix}
    a^{<t-1>} \\ x^{<t>}
\end{bmatrix} 
\rightarrow 
W_a[a^{<t-1>}\;x^{<t>}],\quad
其中W_a=[W_{aa}\;W_{ax}]
$$
那么可以RNN正向传播可以表达为：
$$
a^{<t>}=g_1(W_a[a^{<t-1>}\;x^{<t>}]+b_a),\quad \widehat{y}^{<t>}=g_2(W_ya^{<t>}+b_y)
$$
如果在任意层，a的维度是100，x的维度是10000，那么W_aa的维度为（100，100），W_ax的维度为（100，10000），W_a的维度为：（100，10100）

以上所述的RNN为单向RNN，即按照从左到右顺序，单向进行，只与左边的元素有关。但是，有时候也可能与右边元素有关。例如下面两个句子中，单凭前三个单词，无法确定“Teddy”是否为人名，必须根据右边单词进行判断。

He said, “Teddy Roosevelt was a great President.”
He said, “Teddy bears are on sale!”

因此，有另外一种RNN结构是双向RNN，简称为BRNN，与左右元素均有关系。

RNN模型包含以下几个类型：
Many to many
Many to one
One to many
One to one

不同类型相应的示例结构如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Different%20Types%20Of%20RNNs.png)

## 通过时间的反向传播 Backpropagation Through Time

针对上面识别人名的例子，经过RNN正向传播，单个元素的Loss function为：
$$
L^{<t>}(\widehat{y}^{<t>},y^{<t>})=-y^{<t>}log\widehat{y}^{<t>}-(1-y^{<t>})log(1-\widehat{y}^{<t>})
$$
该样本所有元素的Loss function为：
$$
L(\widehat{y},y)=\sum_{t=1}^{T_y}L^{<t>}(\widehat{y}^{<t>},y^{<t>})
$$
然后，反向传播（Backpropagation）过程就是从右到左分别计算对参数求偏导数。思路与做法与标准的神经网络是一样的。一般可以通过成熟的深度学习框架自动求导，例如PyTorch、Tensorflow等。这种从右到左的求导过程被称为Backpropagation through time。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Backpropagation%20through%20time.png)



## 语言模型和序列生成 Language Model And Sequence Generation

语言模型是自然语言处理（NLP）中最基本和最重要的任务之一，使用RNN能够很好地建立需要的不同语言风格的语言模型。

语言模型的例子：
在语音识别中，某句语音有两种翻译
The apple and pair salad.
The apple and pear salad.

很明显，第二句话更有可能是正确的翻译。语言模型实际上会计算出这两句话各自的出现概率。比如第一句话概率为3.2x10^-13，第二句话概率为5.7x10^-10。也就是说，利用语言模型得到各自语句的概率，选择概率最大的语句作为正确的翻译。概率计算的表达式为：
$$
P(y^{<1>},y^{<2>}……,y^{<T_y>})
$$
使用RNN构建语言模型：首先，需要一个足够大的训练集，训练集由大量的单词语句语料库（corpus）构成。然后，对corpus的每句话进行切分词（tokenize），建立vocabulary，对每个单词进行one-hot编码。例如下面这句话：

The Egyptian Mau is a bread of cat.

还需注意的是，每句话结束末尾，需要加上< EOS >作为语句结束符。另外，若语句中有词汇表中没有的单词，用< UNK >表示。假设单词“Mau”不在词汇表中，则上面这句话可表示为：

The Egyptian < UNK > is a bread of cat. < EOS >

准备好训练集并对语料库进行切分词等处理之后，接下来构建相应的RNN模型：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Language%20Model%20And%20Sequence%20Generation.png)
语言模型的RNN结构如上图所示，𝑎^[0]和x^<1>均为零向量。于是𝑎^<1>要做的就是它会通过 softmax 进行一些预测，来计算出第一个词可能会是什么，其结果就是𝑦^<1>，这一步其实就是通过一个 softmax 层来预测字典中的任意单词会是第一个词的概率，比如说第一个词是𝑎的概率有多少，第一个词是 Aaron 的概率有多少，第一个词是 cats 的概率又有多少，就这样一直到 Zulu 是第一个词的概率是多少，还有第一个词是 UNK（未知词)的概率有多少，还有第一个词是句子结尾标志< EOS >的概率有多少，表示不必阅读。所以𝑦^<1>的输出是 softmax 的计算结果，它只是预测第一个词的概率，而不去管结果是什么。对于x^<2>=y^<1>，此处我们告知模型真实的第一个词是什么，后面的输入同理。

RNN 中的每一步都会考虑前面得到的单词，比如给它前 3 个单词，让它给出下个词的分布，这就是 RNN 如何学习从左往右地每次预测一个词。接下来为了训练这个网络，要定义代价函数。于是，在某个时间步𝑡，如果真正的词是𝑦^<𝑡>，而神经网络的 softmax 层预测结果值是hat(𝑦)<𝑡>，那么单个元素的 softmax loss function表示为：
$$
L^{<t>}(\widehat{y}^{<t>},y^{<t>})=-\sum_iy_i^{<t>}log\widehat{y}_i^{<t>}
$$
总的Loss function为：
$$
L(\widehat y,y)=\sum_tL^{<t>}(\widehat{y}^{<t>},y^{<t>})
$$
对语料库的每条语句进行RNN模型训练，最终得到的模型可以根据给出语句的前几个单词预测其余部分，将语句补充完整。例如给出“Cats average 15”，RNN模型可能预测完整的语句是“Cats average 15 hours of sleep a day.”。

最后补充一点，整个语句出现的概率等于语句中所有元素出现的条件概率乘积。例如某个语句包含y<1>,y<2>,y<3>，则整个语句出现的概率为：
$$
P(y^{<1>},y^{<2>},y^{<3>})=P(y^{<1>})·P(y^{<2>}|y^{<1>})·P(y^{<3>}|y^{<1>},y^{<2>})
$$

## 对新序列采样 Sampling Novel Sequences

利用训练好的RNN语言模型，可以进行新的序列采样，从而随机产生新的语句，相应的RNN模型如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Language%20Model%20And%20Sequence%20Generation.png)
首先，从第一个元素输出的softmax分布中随机选取一个word作为新语句的首单词。然后，让其作为下一层的输入，得到的softmax分布。从中选取概率最大的word作为，继续将其作为下一层的输入，以此类推，直到产生< EOS >结束符，则标志语句生成完毕。当然，也可以设定语句长度上限，达到长度上限即停止生成新的单词。最终，根据随机选择的首单词，RNN模型会生成一条新的语句。

举例：假如说对第一个词进行采样后，得到的是 The，The 作为第一个词的情况很常见，然后把 The 当成𝑥^<2>，现在𝑥^<2>就是𝑦^<1>，现在需要计算出在第一词是 The 的情况下，第二个词应该是什么，然后得到的结果就是hat(𝑦)^<2>，然后再次用这个采样函数来对𝑦^<2>进行采样；然后再到下一个时间步，无论得到什么样的用 one-hot 码表示的选x择结果，都把它传递到下一个时间步，然后对第三个词进行采样。不管得到什么都把它传递下去，一直这样直到最后一个时间步。

值得一提的是，如果不希望新的语句中包含< UNK >标志符，可以在每次产生< UNK >时重新采样，直到生成非< UNK >标志符为止。

以上介绍的是word level RNN，即每次生成单个word，语句由多个words构成。另外一种情况是character level RNN，即词汇表由单个英文字母或字符组成。
Character level RNN与Word level RNN不同的是，由单个字符组成而不是word。训练集中的每句话都当成是由许多字符组成的。Character level RNN的优点是能有效避免遇到词汇表中不存在的单词< UNK >。但是，character level RNN的缺点也很突出，由于是字符表征，每句话的字符数量很大，这种大的跨度不利于寻找语句前部分和后部分之间的依赖性。另外，character level RNN的在训练时的计算量也是庞大的。基于这些缺点，目前character level RNN的应用并不广泛，但是在特定应用下仍然有发展的趋势。

## 循环神经网络的梯度消失 Vanishing Gradients With RNNs

语句中可能存在跨度很大的依赖关系，即某个word可能与它距离较远的某个word具有强依赖关系。例如下面这两条语句：

The cat, which already ate fish, was full.

The cats, which already ate fish, were full.

第一句话中，was受cat影响；第二句话中，were受cats影响，它们之间都跨越了很多单词。而一般的RNN模型每个元素受其周围附近的影响较大，难以建立跨度较大的依赖性。上面两句话的这种依赖关系，由于跨度很大，普通的RNN网络容易出现梯度消失，捕捉不到它们之间的依赖，造成语法错误。

另一方面，RNN也可能出现梯度爆炸的问题，即gradient过大。常用的解决办法是设定一个阈值，一旦梯度最大值达到这个阈值，就对整个梯度向量进行尺度缩小，这种做法被称为梯度修剪gradient clipping。

## 门控循环单元 Gated Recurrent Unit(GRU)

RNN的隐藏层单元结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Gated%20Recurrent%20Unit%201.png)
在这个模型中，RNN的正向传播公式为：
$$
a^{<t>}=tanh(W_a[a^{<t-1>}\;x^{<t>}]+b_a),\quad \widehat{y}^{<t>}=softmax(W_ya^{<t>}+b_y)
$$
为了解决梯度消失问题，对上述单元进行修改，添加了记忆单元，构建GRU，如下图所示:
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Gated%20Recurrent%20Unit%202.png)
GRU 单元将会有个新的变量称为𝑐，代表细胞（cell)，即记忆细胞。记忆细胞的作用是提供了记忆的能力，比如说一只猫是单数还是复数，所以当它看到之后的句子的时候，它仍能够判断句子的主语是单数还是复数。
在 GRU 中真正重要的思想是我们有一个门，我先把这个门叫做𝛤𝑢，这是个下标为𝑢的大写希腊字母𝛤，𝑢代表更新门，这是一个 0 到 1 之间的值，实际上这个值是把这个式子带入 sigmoid 函数得到的。
GRU 的关键部分是用候选值𝑐̃更新𝑐的等式，然后更新门𝛤𝑢决定是否要真的更新它。记忆细胞𝑐<𝑡>将被设定为 0 或者 1，这取决于考虑的谓词在句子中是单数还是复数，如果假定单数情况设为1，复数情况设为0。然后 GRU 单元将会一直记住𝑐<𝑡>的值，直到需要表达谓词单复数不同的情况，如果𝑐<𝑡>的值还是 1，这就说明是单数。而门𝛤𝑢的作用就是决定什么时候会更新这个值，假如是看到词组 the cat，即句子的主语是单数，这就是一个好的时机去更新𝑐<𝑡>。然后当𝑐<𝑡>使用完的时候，即确定了“The cat, which already ate……, was full.”，那么就可以不必再记忆。
如果更新值𝛤𝑢 = 1，也就是说把这个新值𝑐<𝑡>设为候选值，𝛤𝑢 = 1时简化上式𝑐<𝑡> = 𝑐̃<𝑡>，将门值设为 1，然后往前再更新这个值。那么对于所有中间的值，应该把门的值设为 0，即𝛤𝑢 = 0，意思就是说不更新它，就用旧的值，因为如果𝛤𝑢 = 0，则𝑐<𝑡> = 𝑐<𝑡−1>，𝑐<𝑡>等于旧的值。
因此，𝑐 和 𝛤 能够保证RNN模型中跨度很大的依赖关系不受影响，消除梯度消失问题。
$$
\widetilde{c}^{<t>}=tanh(W_c[c^{<t-1>},x^{<t>}]+b_c)
$$

$$
\Gamma_u=\sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)
$$

$$
c^{<t>}=\Gamma_u*\widetilde{c}^{<t>}+(1-\Gamma_u)*c^{<t-1>}
$$

上面介绍的是简化的GRU模型，完整的GRU添加了另外一个gate，即𝛤𝑟，r可以认为代表相关性（relevance），这个𝛤𝑟门表示计算出的下一个𝑐<𝑡>的候选值𝑐̃<𝑡>跟𝑐<𝑡−1>有多大的相关性，表达式如下：
$$
\widetilde{c}^{<t>}=tanh(W_c[\Gamma_r*c^{<t-1>},x^{<t>}]+b_c)
$$

$$
\Gamma_u=\sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)
$$

$$
\Gamma_r=\sigma(W_r[c^{<t-1>},x^{<t>}]+b_r)
$$

$$
c^{<t>}=\Gamma_u*\widetilde{c}^{<t>}+(1-\Gamma_u)*c^{<t-1>}
$$

$$
a^{<t>}=c^{<t>}
$$

## 长短期记忆 Long-Short Term Memory(LSTM)

LSTM是另一种更强大的解决梯度消失问题的方法，它对应的RNN隐藏层单元结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Long%20Short%20Term%20Memory.png)
LSTM有三个门，𝛤𝑢（更新门）、𝛤𝑓 （遗忘门）、𝛤𝑜（输出门)。上图表示的模型对应的表达式为：
$$
\widetilde{c}^{<t>}=tanh(W_c[a^{<t-1>},x^{<t>}]+b_c)\quad LSTM\;中不再有𝑎^{<𝑡>} = 𝑐^{<𝑡>}的情况
$$

$$
\Gamma_u=\sigma(W_u[a^{<t-1>},x^{<t>}]+b_u)
$$

$$
\Gamma_f=\sigma(W_f[a^{<t-1>},x^{<t>}]+b_f)
$$

$$
\Gamma_o=\sigma(W_o[a^{<t-1>},x^{<t>}]+b_o)
$$

$$
c^{<t>}=\Gamma_u*\widetilde{c}^{<t>}+\Gamma_f*c^{<t-1>}
$$

$$
a^{<t>}=\Gamma_o*tanh(c^{<t>})
$$

GRU可以看成是简化的LSTM，两种方法都具有各自的优势。

## 双向循环神经网络 Bidirectional RNN

Bidirectional RNN，它的结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Bidirectional%20RNN.png)

BRNN对应的输出表达式为：
$$
\widehat{y}^{<t>}=g(W_y[a^{\rightarrow<t>},a^{\leftarrow<t>}]+b_y)
$$
BRNN的基本单元不仅仅是标准 RNN 单元，也可以是 GRU单元或者 LSTM 单元。事实上，很多的 NLP 问题，对于大量有自然语言处理问题的文本，有 LSTM 单元的双向 RNN 模型是用的最多的。BRNN能够同时对序列进行双向处理，性能大大提高，但是计算量较大，且在处理实时语音时，需要等到完整的一句话结束时才能进行分析。

## 深层循环神经网络 Deep RNNs

Deep RNNs由多层RNN组成，其结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Deep%20RNNs%201.png)

与DNN一样，用中括号上标表示层数。Deep RNNs中的表达式为：
$$
a^{[l]<t>}=g(W_a^{[l]}[a^{[l]<t-1>},a^{[l-1]<t>}]+b_a^{[l]})
$$
DNN层数可达100多，而Deep RNNs一般没有那么多层，3层RNNs已经较复杂了。

另外一种Deep RNNs结构是每个输出层上还有一些垂直单元，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Deep%20RNNs%202.png)



# 自然语言处理与词嵌入 Natural Language Processing And Word Embeddings

## 词汇表征 Word Representation

前文内容中表征单词的方式是首先建立一个较大的词汇表（例如10000），然后使用one-hot的方式对每个单词进行编码。但是one-hot表征单词的方法最大的缺点就是每个单词都是独立的、正交的，无法知道不同单词之间的相似程度。例如Apple和Orange都是水果，词性相近，但是单从one-hot编码上来看，内积为零，无法知道二者的相似性。在NLP中，我们更希望能掌握不同单词之间的相似程度。

因此，我们可以使用特征表征（Featurized representation）的方法对每个单词进行编码，也就是使用一个特征向量表征单词，特征向量的每个元素都是对该单词某一特征的量化描述，量化范围可以是[-1,1]之间。特征表征的例子如下图所示
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Word%20Representation%201.png)

特征向量的长度依情况而定，特征元素越多则对单词表征得越全面。这里的特征向量长度设定为300。使用特征表征之后，词汇表中的每个单词都可以使用对应的300 x 1的向量来表示，该向量的每个元素表示该单词对应的某个特征值。
这种特征表征的优点是根据特征向量能清晰知道不同单词之间的相似程度，例如Apple和Orange之间的相似度较高，很可能属于同一类别。这种单词“类别”化的方式，大大提高了有限词汇量的泛化能力，这种特征化单词的操作被称为Word Embeddings，即单词嵌入。

值得一提的是，这里特征向量的每个特征元素含义是具体的，对应到实际特征，例如性别、年龄等。而在实际应用中，特征向量很多特征元素并不一定对应到有物理意义的特征，是比较抽象的。但是，这并不影响对每个单词的有效表征，同样能比较不同单词之间的相似性。

每个单词都由高维特征向量表征，为了可视化不同单词之间的相似性，可以使用降维操作，例如t-SNE算法，将300D降到2D平面上。如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Word%20Representation%202.png)
从上图可以看出相似的单词分布距离较近，从而也证明了Word Embeddings能有效表征单词的关键特征。

## 使用词嵌入 Using Word Embeddings

在Named Entity识别中，每个单词采用的是one-hot编码。如下图所示，因为“orange farmer”是份职业，很明显“Sally Johnson”是一个人名。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Using%20Word%20Embeddings.png)

如果采用featurized representation对每个单词进行编码，再构建该RNN模型。对于一个新的句子：
Robert Lin is an apple farmer
由于这两个句子中，“apple”与“orange”特征向量很接近，很容易能判断出“Robert Lin”也是一个人名，这就是featurized representation的优点之一。

可以看出，featurized representation的优点是可以减少训练样本的数目，前提是对海量单词建立特征向量表述（word embedding）。这样，即使训练样本不够多，测试时遇到陌生单词，例如“durian cultivator”，根据之前海量词汇特征向量就判断出“durian”也是一种水果，与“apple”类似，而“cultivator”与“farmer”也很相似，从而得到与“durian cultivator”对应的应该也是一个人名。这种做法将单词用不同的特征来表示，即使是训练样本中没有的单词，也可以根据word embedding的结果得到与其词性相近的单词，从而得到与该单词相近的结果，有效减少了训练样本的数量。

featurized representation的特性使得很多NLP任务能方便地进行迁移学习，具体步骤是：
1、从海量词汇库中学习word embeddings，即所有单词的特征向量，或者从网上下载预训练好的word embeddings
2、使用较少的训练样本，将word embeddings迁移到新的任务中
3、（可选）：继续使用新数据微调word embeddings

建议仅当训练样本足够大的时候，再进行上述第三步。

有趣的是，word embeddings与人脸特征编码有很多相似性，人脸图片经过Siamese网络，得到其特征向量，这点跟word embedding是类似的。二者不同的是Siamese网络输入的人脸图片可以是数据库之外的，而word embedding一般都是已建立的词汇库中的单词，非词汇库单词统一用< UNK >表示。

## 词嵌入的特性 Properties Of Word Embeddings

Word embeddings可以帮助我们找到不同单词之间的相似类别关系，如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Properties%20Of%20Word%20Embeddings%201.png)

上例中，特征维度是4维的，分别是[Gender, Royal, Age, Food]。常识地，“Man”与“Woman”的关系类比于“King”与“Queen”的关系，而利用Word embeddings可以找到这样的对应类比关系。

将“Man”的embedding vector与“Woman”的embedding vector相减：
$$
e_{man}-e_{woman}=
\begin{bmatrix}
-1 \\ 0.01 \\ 0.03 \\ 0.09
\end{bmatrix}
-
\begin{bmatrix}
1 \\ 0.02 \\ 0.02 \\ 0.01
\end{bmatrix}
=
\begin{bmatrix}
-2 \\ -0.01 \\ 0.01 \\ 0.08
\end{bmatrix}
\approx
\begin{bmatrix}
-2 \\ 0 \\ 0 \\ 0
\end{bmatrix}
$$
类似地，将“King”的embedding vector与“Queen”的embedding vector相减：
$$
  e_{king}-e_{queen}=
\begin{bmatrix}
-0.95 \\ 0.93 \\ 0.70 \\ 0.02
\end{bmatrix}
-
\begin{bmatrix}
0.97 \\ 0.95\\ 0.69 \\ 0.01
\end{bmatrix}
=
\begin{bmatrix}
-1.92 \\ -0.02 \\ 0.01 \\ 0.01
\end{bmatrix}
\approx
\begin{bmatrix}
-2 \\ 0 \\ 0 \\ 0
\end{bmatrix}
$$
相减结果表明，“Man”与“Woman”的主要区别是性别，“King”与“Queen”也是一样。

一般地，A类比于B相当于C类比于“？”，这类问题可以使用embedding vector进行运算
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Properties%20Of%20Word%20Embeddings%202.png)
如上图所示，那么要做的就是找到单词w来最大化相似程度：
$$
Find\;word\;w:arg\;max\;Sim(e_w,e_{king}-e_{man}+e_{woman}) 
$$
通过这种方法来做类比推理准确率大概只有 30%~75%.

最常用的相似度函数叫做余弦相似度cosine similarity，其表达式为：
$$
Sim(u,v)=\frac{u^Tv}{||u||_2||v||_2}=cos\theta
$$
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Properties%20Of%20Word%20Embeddings%203.png)
这种相似度取决于角度在向量𝑢和𝑣之间。如果向量𝑢和𝑣非常相似，它们的余弦相似度将接近1; 如果它们不相似，余弦相似度将取较小值

## 嵌入矩阵 Embedding Matrix

假设某个词汇库包含了10000个单词，每个单词包含的特征维度为300，那么表征所有单词的embedding matrix维度为300 x 10000，用E来表示，某单词w的one-hot向量表示为𝑂w，维度为10000 x 1，则该单词的embedding vector表达式为：
$$
e_w=E·O_w
$$
因此，只要知道了embedding matrix ，就能计算出所有单词的embedding vector 。值得一提的是，上述这种矩阵乘积运算效率并不高，矩阵维度很大，且大部分元素为零，通常做法是直接从中选取 E 第w列即可。

Embedding matrix 可以通过构建自然语言模型，运用梯度下降算法得到。

举个简单的例子，输入样本是：I want a glass of orange (juice).通过这句话的前6个单词，预测最后的单词“juice”。
待求未知，每个单词可用one-hot vector 表示，然后要做的就是生成一个embedding matrix E，embedding matrix 乘以 one-hot vector 得到 embedding vector。构建的神经网络模型结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Embedding%20Matrix.png)

神经网络输入层包含6个embedding vactors，每个embedding vector维度是300，则输入层总共有1800个输入。Softmax层有10000个概率输出，与词汇表包含的单词数目一致，正确的输出label是“juice”。对足够的训练例句样本，运用梯度下降算法，迭代优化，最终求出embedding matrix 。

为了让神经网络输入层数目固定，可以选择只取预测单词的前4个单词作为输入，例如该句中只选择“a glass of orange”四个单词作为输入，这里的4是超参数，可调。

一般地，我们把输入叫做context，输出叫做target。对应到上面的例子：
context: a glass of orange
target: juice

关于context的选择有多种方法：
target前n个单词或后n个单词，n可调
target前1个单词
target附近某1个单词（Skip-Gram）

事实证明，不同的context选择方法都能计算出较准确的embedding matrix 

## Word2Vec Algorithm

Context和Target的选择方法，比较流行的是采用Skip-Gram模型。
以这句话为例：I want a glass of orange juice to go along with my cereal.
Skip-Gram模型的做法是：首先随机选择一个单词作为context，例如“orange”；然后使用一个宽度为5或10（自定义）的滑动窗，在context附近选择一个单词作为target，可以是“juice”、“glass”、“my”等等。最终得到了多个context—target对作为监督式学习样本。

但是构造这个监督学习问题的目标并不是想要解决这个监督学习问题本身，而是想要使用这个学习问题来学到一个好的词嵌入模型。训练的过程是构建自然语言模型，𝑒𝑐是context的embedding vector，经过softmax单元的输出为：
$$
e_c=E·O_c\quad Softmax:p(t|c)=\frac{e^{\theta^T_te_c}}{\sum_{j=1}^{10000}e^{\theta^T_je_c}},其中\theta_t是一个与输出𝑡有关的参数，表示某个词𝑡和标签相符的概率是多少
$$
我们用𝑦表示目标词，𝑦和hat(y)都是用 one-hot 表示的，于是损失函数就会是：
$$
L(\widehat{y},y)=-\sum_{i=1}^{10000}y_ilog\widehat{y}_i
$$
然后，运用梯度下降算法，迭代优化，最终得到embedding matrix 。

然而，这种算法计算量大，影响运算速度。主要因为softmax输出单元为10000个，计算公式中包含了大量的求和运算。解决的办法之一是使用hierarchical softmax classifier，即树形分类器。其结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Word2Vec%20Algorithm.png)

这种树形分类器是一种二分类，与之前的softmax分类器不同，它在每个数节点上对目标单词进行区间判断，最终定位到目标单词。这好比是猜数字游戏，数字范围0～100。我们可以先猜50，如果分类器给出目标数字比50大，则继续猜75，以此类推，每次从数据区间中部开始。这种树形分类器最多需要log_2^N步就能找到目标单词，N为单词总数。

实际应用中，对树形分类器做了一些改进。改进后的树形分类器是非对称的，通常选择把比较常用的单词放在树的顶层，而把不常用的单词放在树的底层，这样更能提高搜索速度。

关于context的采样，需要注意的是如果使用均匀采样，那么一些常用的介词、冠词，例如the, of, a, and, to等出现的概率更大一些。但是，这些单词的embedding vectors通常不是我们最关心的，我们更关心例如orange, apple， juice等这些名词等。所以，实际应用中，一般不选择随机均匀采样的方式来选择context，而是使用其它算法来处理这类问题。

Skip-Gram模型是Word2Vec的一种，Word2Vec的另外一种模型是CBOW（Continuous Bag of Words）。CBOW 是从原始语句推测目标字词；而 Skip-Gram 正好相反，是从目标字词推测出原始语句。 Skip-Gram 模型关键问题在于 softmax 这个步骤的计算成本非常昂贵，因为它需要在分母里对词汇表中所有词求和。通常情况下，Skip-Gram 模型用地更多。

## 负采样 Negative Sampling

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Negative%20Sampling.png)
Negative sampling是另外一种有效的求解embedding matrix 的方法。它的做法是判断选取的context word和target word是否构成一组正确的context-target对，一般包含一个正样本和k个负样本。例如，“orange”为context word，“juice”为target word，很明显“orange juice”是一组context-target对，为正样本，相应的target label为1。若“orange”为context word不变，target word随机选择“king”、“book”、“the”或者“of”等。这些都不是正确的context-target对，为负样本，相应的target label为0。一般地，固定某个context word对应的负样本个数k一般遵循：若训练样本较小，k一般选择5～20；若训练样本较大，k一般选择2～5即可。

新的输入𝑥，𝑦将是要预测的值。为了定义模型，使用记号𝑐表示上下文词，记号𝑡表示可能的目标词，再用𝑦表示 0 和 1，表示是否是一对上下文-目标词。要做的是定义一个逻辑回归模型，给定输入的𝑐，𝑡对的条件下，𝑦 = 1的概率，即：
$$
P(y=1|c,t)=\sigma(\theta_t^T·e_c)\quad \sigma表示sigmoid函数
$$
很明显，negative sampling某个固定的正样本对应k个负样本，即模型总共包含了k+1个binary classification。对比之前介绍的10000个输出单元的softmax分类，negative sampling转化为k+1个二分类问题，计算量要小很多，大大提高了模型运算速度。

这个算法有一个重要的细节就是如何选取负样本，即在选取了上下文词 (例如orange) 之后，如何对这些词进行采样生成负样本？可以使用随机选择的方法。但有资料提出一个更实用、效果更好的方法，就是根据该词出现的频率进行选择，相应的概率公式为：
$$
P(w_i)=\frac{f(w_i)^{\frac{3}{4}}}{\sum_{j=1}^{10000}f(w_j)^\frac{3}{4}}\quad 𝑓(𝑤_𝑖)是观测到的在语料库中的某个英文词的词频
$$

## GloVe 词向量 GloVe Word Vectors

GloVe 代表用词表示的全局变量（global vectors for word representation）,在此之前，我们曾通过挑选语料库中位置相近的两个词，列举出词对，即上下文和目标词，GloVe 算法做的就是使其关系开始明确化。

GloVe算法引入了一个新的参数：X_ij (表示i出现在j之前的次数，即i和j同时出现的次数)

其中，i表示context，j表示target。一般地，如果不限定context一定在target的前面，则有对称关系，即X_ij=X_ji。接下来的讨论中，我们默认存在对称关系。
对于 GloVe 算法，我们可以定义上下文和目标词为任意两个位置相近的单词，假设是左右各 10 词的距离，那么𝑋𝑖𝑗就是一个能够获取单词𝑖和单词𝑗出现位置相近时或是彼此接近的频率的计数器。GloVe 模型做的就是进行优化，我们将他们之间的差距进行最小化处理：
$$
loss function:minimize\sum_{i=1}^{10000}\sum_{j=1}^{10000}=f(X_{ij})(\theta_i^Te_j+b_i+b'_j-logX_{ij})^2
$$
从上式可以看出，若两个词的embedding vector越相近，同时出现的次数越多，则对应的loss越小。b_i 和 b'_j 是偏移量。为了防止出现“log 0”，即两个单词不会同时出现，无相关性的情况，对loss function引入一个权重因子：f(X_ij)，那么即使是像 durion 这样不常用的词，它也能给予大量有意义的运算，同时也能够给像 this，is，of，a 这样在英语里出现更频繁的词更大但不至于过分的权重。

关于这个算法𝜃和𝑒现在是完全对称的，所以𝜃𝑖和𝑒𝑗就是对称的。如果只看数学式的话，𝜃𝑖和𝑒𝑗 的功能其实很相近，可以将它们颠倒
或者将它们进行排序，实际上它们都输出了最佳结果。因此一种训练算法的方法是一致地初始化𝜃和𝑒，然后使用梯度下降来最小化输出，当每个词都处理完之后取平均值，所以，给定一个词𝑤，使用优化算法得到所有参数之后，最终的可表示为：
$$
e_w=\frac{e_w+\theta_w}{2}
$$
因为𝜃和𝑒在这个特定的公式里是对称的，而不像之前提到的模型，𝜃和𝑒功能不一样，因此也不能像那样取平均。

最后提一点的是，无论使用Skip-Gram模型还是GloVe模型等等，计算得到的embedding matrix 的每一个特征值不一定对应有实际物理意义的特征值，如gender，age等。

## 情感分类 Sentiment Classification

情感分类一般是根据一句话来判断其喜爱程度，例如1～5星分布。如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sentiment%20Classification%201.png)

情感分类问题的一个主要挑战是缺少足够多的训练样本，而Word embedding恰恰可以帮助解决训练样本不足的问题，使用word embedding能够有效提高模型的泛化能力，即使训练样本不多，也能保证模型有不错的性能。

使用word embedding解决情感分类问题的一个简单模型算法表示如下：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sentiment%20Classification%202.png)
如上图所示，这句话的4个单词分别用embedding vector表示。计算均值，这样得到的平均向量的维度仍是300，最后经过softmax输出1～5星。这种模型结构简单，计算量不大，不论句子长度多长，都使用平均的方式得到300D的embedding vector，该模型实际表现较好。

但是，这种简单模型的缺点是使用平均方法，没有考虑句子中单词出现的次序，忽略其位置信息。而有时候，不同单词出现的次序直接决定了句意，即情感分类的结果。例如下面这句话：

Completely lacking in good taste, good service, and good ambience.

虽然这句话中包含了3个“good”，但是其前面出现了“lacking”，很明显这句话句意是negative的。如果使用上面介绍的平均算法，则很可能会错误识别为positive的，因为忽略了单词出现的次序。

为了解决这一问题，情感分类的另一种模型是RNN：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sentiment%20Classification%20%203.png)
该RNN模型是典型的many-to-one模型，考虑单词出现的次序，能够有效识别句子表达的真实情感。

## 词嵌入除偏 Debiasing Word Embeddings

Word embeddings中存在一些性别、宗教、种族等偏见或者歧视。例如下面这三句话：

Man: Woman as King: Queen

Man: Computer programmer as Woman: Homemaker

Father: Doctor as Mother: Nurse

很明显，第二句话和第三句话存在性别偏见，因为Woman和Mother也可以是Computer programmer和Doctor。

以性别偏见为例，接下来探讨如何消除word embeddings中偏见。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Debiasing%20Word%20Embeddings%201.png)
上图展示了bias direction和non-bias direction。

首先，确定偏见bias的方向，方法是对所有性别对立的单词求差值，再平均：
$$
bias\;direction=\frac{1}{N}[(e_{he}-e_{she})+(e_{male}-e_{female})+...]
$$
然后，单词中立化Neutralize。将需要消除性别偏见的单词投影到non-bias direction上去，消除bias维度，例如babysitter，doctor等。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Debiasing%20Word%20Embeddings%202.png)

最后，均衡对（Equalize pairs）。让性别对立单词与上面的中立词距离相等，具有同样的相似度。例如让grandmother和grandfather与babysitter的距离同一化。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Debiasing%20Word%20Embeddings%203.png)

值得注意的是，掌握哪些单词需要中立化非常重要。一般来说，大部分英文单词，例如职业、身份等都需要中立化，消除embedding vector中性别这一维度的影响。



# 序列模型 Sequence Models

## Sequence To Sequence Models

Sequence to sequence模型在机器翻译和语音识别方面都有着广泛的应用。

机器翻译的简单例子：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20To%20Sequence%20Models%201.png)
针对该机器翻译问题，可以使用“编码网络（encoder network）”+“解码网络（decoder network）”两个RNN模型组合的形式来解决。encoder network将输入语句编码为一个特征向量，传递给decoder network，完成翻译。具体模型结构如下图所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20To%20Sequence%20Models%202.png)
其中，encoder vector代表了输入语句的编码特征。encoder network和decoder network都是RNN模型，可使用GRU或LSTM单元。这种“编码网络（encoder network）”+“解码网络（decoder network)”的模型，在实际的机器翻译应用中有着不错的效果。

这种模型也可以应用到图像捕捉领域。图像捕捉，即捕捉图像中主体动作和行为，描述图像内容。例如下面这个例子，根据图像，捕捉图像内容。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20To%20Sequence%20Models%203.png)
首先，可以将图片输入到CNN，例如使用预训练好的AlexNet，删去最后的softmax层，保留至最后的全连接层。则该全连接层就构成了一个图片的特征向量（编码向量)，表征了图片特征信息。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20To%20Sequence%20Models%204.png)
然后，将encoder vector输入至RNN，即decoder network中，进行解码翻译。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Sequence%20To%20Sequence%20Models%205.png)

## 选择最可能的句子 Picking The Most Likely Sentence

Sequence to sequence machine translation model与Language model有一些相似，但也存在不同之处，二者模型结构如下所示：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Picking%20The%20Most%20Likely%20Sentence%201.png)
Language model是自动生成一条完整语句，语句是随机的。而machine translation model是根据输入语句，进行翻译，生成另外一条完整语句。上图中，绿色部分表示encoder network，紫色部分表示decoder network。decoder network与language model是相似的，encoder network可以看成是language model的，是整个模型的一个条件。也就是说，machine translation model在输入语句的条件下，生成正确的翻译语句。因此，machine translation可以看成是有条件的语言模型（conditional language model)。这就是二者之间的区别与联系。

所以，machine translation的目标就是根据输入语句，作为条件，找到最佳翻译语句，使其概率最大：
$$
max\;P(y^{<1>},y^{<2>},...,y^{<T_y>}|x^{<1>},x^{<2>},...,x^{<T_x>})
$$
例如，列举几个模型可能得到的翻译：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Picking%20The%20Most%20Likely%20Sentence%202.png)
显然，第一条翻译“Jane is visiting Africa in September.”最为准确，那我们的优化目标就是要让这条翻译对应的最大化。

实现优化目标的方法之一是使用贪婪搜索（greedy search）。Greedy search根据条件，每次只寻找一个最佳单词作为翻译输出，力求把每个单词都翻译准确。例如，首先根据输入语句，找到第一个翻译的单词“Jane”，然后再找第二个单词“is”，再继续找第三个单词“visiting”，以此类推。这也是其“贪婪”名称的由来。

Greedy search存在一些缺点。首先，因为greedy search每次只搜索一个单词，没有考虑该单词前后关系，概率选择上有可能会出错。例如，上面翻译语句中，第三个单词“going”比“visiting”更常见，模型很可能会错误地选择了“going”，而错失最佳翻译语句。其次，greedy search大大增加了运算成本，降低运算速度。

因此，greedy search并不是最佳的方法。

Greedy search每次是找出预测概率最大的单词，而集束搜索Beam search则是每次找出预测概率最大的B个单词。其中，参数B表示取概率最大的单词个数，可调。
按照beam search的搜索原理，首先，先从词汇表中找出翻译的第一个单词概率最大的B个预测单词。例如上面的例子中，令B=3，预测得到的第一个单词为：in，jane，september。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Picking%20The%20Most%20Likely%20Sentence%203.png)

然后，再分别以in，jane，september为条件，计算每个词汇表单词作为预测第二个单词的概率。从中选择概率最大的3个作为第二个单词的预测值，得到：in september，jane is，jane visits。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Picking%20The%20Most%20Likely%20Sentence%204.png)
此时，得到的前两个单词的3种情况的概率为：
$$
P(\widehat{y}^{<1>},\widehat{y}^{<2>}|x)=P(\widehat{y}^{<1>}|x)·P(\widehat{y}^{<2>}|x,\widehat{y}^{<1>})
$$
接着，再预测第三个单词。方法一样，分别以in september，jane is，jane visits为条件，计算每个词汇表单词作为预测第三个单词的概率。从中选择概率最大的3个作为第三个单词的预测值，得到：in september jane，jane is visiting，jane visits africa。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Picking%20The%20Most%20Likely%20Sentence%205.png)
此时，得到的前三个单词的3种情况的概率为：
$$
P(\widehat{y}^{<1>},\widehat{y}^{<2>}，\widehat{y}^{<3>}|x)=P(\widehat{y}^{<1>}|x)·P(\widehat{y}^{<2>}|x,\widehat{y}^{<1>})·P(\widehat{y}^{<3>}|x,\widehat{y}^{<1>},\widehat{y}^{<2>})
$$
以此类推，每次都取概率最大的三种预测。最后，选择概率最大的那一组作为最终的翻译语句。

Jane is visiting Africa in September.

值得注意的是，如果参数B=1，则就等同于greedy search。实际应用中，可以根据不同的需要设置B为不同的值。一般B越大，机器翻译越准确，但同时也会增加计算复杂度。

## 改进集束搜索 Refinements To Beam Search

Beam search中，最终机器翻译的概率是乘积的形式：
$$
arg\;max\displaystyle \prod_{t=1}^{T_y}P(\widehat{y}^{<t>}|x,\widehat{y}^{<1>},...,\widehat{y}^{<t-1>})
$$
多个概率相乘可能会使乘积结果很小，远小于1，造成数值下溢。为了解决这个问题，可以对上述乘积形式进行取对数log运算，即：
$$
arg\;max\displaystyle \sum_{t=1}^{T_y}logP(\widehat{y}^{<t>}|x,\widehat{y}^{<1>},...,\widehat{y}^{<t-1>})
$$
因为取对数运算，将乘积转化为求和形式，避免了数值下溢，使得数据更加稳定有效。

这种概率表达式还存在一个问题，就是机器翻译的单词越多，乘积形式或求和形式得到的概率就越小，这样会造成模型倾向于选择单词数更少的翻译语句，使机器翻译受单词数目的影响，这显然是不太合适的。因此，一种改进方式是进行长度归一化，消除语句长度影响。
$$
arg\;max\frac{1}{T_y}\displaystyle \sum_{t=1}^{T_y}logP(\widehat{y}^{<t>}|x,\widehat{y}^{<1>},...,\widehat{y}^{<t-1>})
$$
实际应用中，通常会引入归一化因子𝑎：
$$
arg\;max\frac{1}{T_y^\alpha}\displaystyle \sum_{t=1}^{T_y}logP(\widehat{y}^{<t>}|x,\widehat{y}^{<1>},...,\widehat{y}^{<t-1>})
$$
在实践中，有个探索性的方法，相比于直接除𝑇𝑦，也就是输出句子的单词总数，我们有时会用一个更柔和的方法，在𝑇𝑦上加上指数𝑎，𝑎可以等于 0.7。如果𝑎等于 1，就相当于完全用长度来归一化，如果𝑎等于 0，𝑇𝑦的 0 次幂就是 1，就相当于完全没有归一化，这就是在完全归一化和没有归一化之间。𝑎就是算法另一个超参数（hyper parameter），需要调整大小来得到最好的结果。不得不承认，这样用𝑎实际上是试探性的，它并没有理论验证。但是大家都发现效果很好，大家都发现实践中效果不错，所以很多人都会这么做。你可以尝试不同的𝑎值，看看哪一个能够得到最好的结果。

值得一提的是，与BFS (Breadth First Search) 、DFS (Depth First Search)算法不同，beam search运算速度更快，但是并不保证一定能找到正确的翻译语句。

## Bleu Score

使用Bleu score（bilingual evaluation understudy 双语评估替补），对机器翻译进行打分。

首先，对原语句建立人工翻译参考，一般有多个人工翻译（利用验证集或测试集）
例如下面这个例子：
French: Le chat est sur le tapis.
Reference 1: The cat is on the mat.
Reference 2: There is a cat on the mat.
上述两个人工翻译都是正确的，作为参考

相应的机器翻译如下所示：
French: Le chat est sur le tapis.
Reference 1: The cat is on the mat.
Reference 2: There is a cat on the mat.
MT output: the the the the the the the.
如上所示，机器翻译为“the the the the the the the.”，效果很差。Bleu Score的宗旨是机器翻译越接近参考的人工翻译，其得分越高，方法原理就是看机器翻译的各个单词是否出现在参考翻译中。

最简单的准确度评价方法是看机器翻译的每个单词是否出现在参考翻译中。显然，上述机器翻译的每个单词都出现在参考翻译里，准确率为 7/7，其中，分母为机器翻译单词数目，分子为相应单词是否出现在参考翻译中。但是，这种方法很不科学，并不可取。

另外一种评价方法是看机器翻译单词出现在参考翻译单个语句中的次数，取最大次数。上述例子对应的准确率为 2/7，其中，分母为机器翻译单词数目，分子为相应单词出现在参考翻译中的最大次数（分子为2是因为“the”在参考1中出现了两次）。这种评价方法较为准确。

上述两种方法都是对单个单词进行评价。按照beam search的思想，另外一种更科学的打分方法是bleu score on bigrams，即同时对两个连续单词进行打分。

仍然是上面那个翻译例子：
French: Le chat est sur le tapis.
Reference 1: The cat is on the mat.
Reference 2: There is a cat on the mat.
MT output: The cat the cat on the mat.

对MIT output进行分解，得到的bigrams及其出现在MIT output中的次数count为：
the cat: 2
cat the: 1
cat on: 1
on the: 1
the mat: 1

然后，统计上述bigrams出现在参考翻译单个语句中的次数（取最大次数）为：
the cat: 1
cat the: 0
cat on: 1
on the: 1
the mat: 1

相应的bigrams precision为：
$$
\frac{count_{clip}}{count}=\frac{1+0+1+1+1}{2+1+1+1+1}=\frac{4}{6}=\frac{2}{3}
$$
如果只看单个单词，相应的unigrams precision为：
$$
p_1=\displaystyle \frac{\sum_{unigram\in\widehat y}count_{clip}(unigram)}{\sum_{unigram\in\widehat y}count(unigram)}
$$
如果是n个连续单词，相应的n-grams precision为：
$$
p_n=\displaystyle \frac{\sum_{n-grams\in\widehat y}count_{clip}(n-gram)}{\sum_{n-grams\in\widehat y}count(n-gram)}
$$
可以同时计算，再对其求平均作为e的指数，所以Bleu Score可以记为：
$$
p=e^{\frac{1}{n}\sum_{i=1}^np_i}
$$
再引入参数因子brevity penalty，记为BP。顾名思义，BP是为了“惩罚”机器翻译语句过短而造成的得分“虚高”的情况
$$
p=BP·exp(\frac{1}{n}\sum_{i=1}^np_i)
$$
BP值由机器翻译长度和参考翻译长度共同决定
$$
f(n)= \begin {cases}
1, & \text {if MT\_output\_length > reference\_output\_length} 
\\
exp(1-MT\_output\_length / reference\_output\_length), & \text{otherwise}
\end{cases}
$$
