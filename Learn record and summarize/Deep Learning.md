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
注意：上式求方差中的向量x是已经完成零均值操作的向量x

以二维平面为例，下图展示了其归一化过程：
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Normalizing%20inputs%202.png)

值得注意的是，由于训练集进行了标准化处理，那么对于测试集或在实际应用时，应该使用同样的和对其进行标准化处理。这样保证了训练集合测试集的标准化操作一致。

之所以要对输入进行标准化操作，主要是为了让所有输入归一化同样的尺度上，方便进行梯度下降算法时能够更快更准确地找到全局最优解。假如输入特征是二维的，且x1的范围是[1,1000]，x2的范围是[0,1]。如果不进行标准化处理，x1与x2之间分布极不平衡，训练得到的w1和w2也会在数量级上差别很大。这样导致的结果是cost function与w和b的关系可能是一个非常细长的椭圆形碗。对其进行梯度下降算法时，由于w1和w2数值差异很大，只能选择很小的学习因子，来避免J发生振荡。一旦较大，必然发生振荡，J不再单调下降。如下左图所示。然而，如果进行了标准化操作，x1与x2分布均匀，w1和w2数值差别不大，得到的cost function与w和b的关系是类似圆形碗。对其进行梯度下降算法时，可以选择相对大一些，且J一般不会发生振荡，保证了J是单调下降的。如下右图所示。
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Deep%20Learning/Normalizing%20inputs%203.png)

另外一种情况，如果输入特征之间的范围本来就比较接近，那么不进行标准化操作也是没有太大影响的。但是，标准化处理在大多数场合下还是值得推荐的。

## 梯度消失/梯度爆炸 Vanishing / Exploding gradients

在神经网络尤其是深度神经网络中存在可能存在这样一个问题：梯度消失和梯度爆炸。意思是当训练一个 层数非常多的神经网络时，计算得到的梯度可能非常小或非常大，甚至是指数级别的减小或增大。这样会让训练过程变得非常困难。

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





