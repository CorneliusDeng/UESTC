# 数据导入方法

需要用的两个库

```python
import numpy as np
import pandas as pd

# NumPy Arrays
data_array.dtype  # 数组元素的数据类型
data_array.shape  # 阵列尺寸
len(data_array)   # 数组的长度
# Pandas DataFrames
df.head()  # 返回DataFrames前几行（默认5行）
df.tail()  # 返回DataFrames最后几行（默认5行）
df.index   # 返回DataFrames索引
df.columns # 返回DataFrames列名
df.info()  # 返回DataFrames基本信息
data_array = data.values # 将DataFrames转换为NumPy数组
```

## 纯文本文件

```python
filename = 'demo.txt'
file = open(filename, mode='r') # 打开文件进行读取
text = file.read() # 读取文件的内容
print(file.closed) # 检查文件是否关闭
file.close() # 关闭文件
print(text)

# 使用上下文管理器 -- with
with open('demo.txt', 'r') as file:
    print(file.readline()) # 一行一行读取
    print(file.readline())
    print(file.readline())
```

## 表格数据：Flat文件

Flat 文件是一种包含没有相对关系结构的记录的文件（支持Excel、CSV和Tab分割符文件 ）

```python
# 具有一种数据类型的文件，用于分隔值的字符串跳过前两行,在第一列和第三列读取结果数组的类型。
filename = 'mnist.txt'
data = np.loadtxt(filename,
                  delimiter=',',
                  skiprows=2,
                  usecols=[0,2],
                  dtype=str)

# 具有混合数据类型的文件
filename = 'titanic.csv'
data = np.genfromtxt(filename,
                     delimiter=',',
                     names=True,
                     dtype=None)

# 使用 Pandas 读取Flat文件
filename = 'demo.csv' 
data = pd.read_csv(filename, 
                   nrows=5,        # 要读取的文件的行数
                   header=None,    # 作为列名的行号
                   sep='\t',       # 分隔符使用
                   comment='#',    # 分隔注释的字符
                   na_values=[""]) # 可以识别为NA/NaN的字符串
```

## Excel 电子表格

Pandas中的ExcelFile()是pandas中对excel表格文件进行读取相关操作非常方便快捷的类，尤其是在对含有多个sheet的excel文件进行操控时非常方便

```python
file = 'demo.xlsx'
data = pd.ExcelFile(file)
df_sheet2 = data.parse(sheet_name='1960-1966',
                       skiprows=[0],
                       names=['Country',
                              'AAM: War(2002)'])
df_sheet1 = pd.read_excel(data,
                          sheet_name=0,
                          parse_cols=[0],
                          skiprows=[0],
                          names=['Country']
# 使用sheet_names属性获取要读取工作表的名称
data.sheet_names                          
```

## SAS 文件

SAS (Statistical Analysis System)是一个模块化、集成化的大型应用软件系统，其保存的文件即sas是统计分析文件。

```python
from sas7bdat import SAS7BDAT
with SAS7BDAT('demo.sas7bdat') as file:
  df_sas = file.to_data_frame()
```

## Stata 文件

Stata 是一套提供其使用者数据分析、数据管理以及绘制专业图表的完整及整合性统计软件。其保存的文件后缀名为.dta的Stata文件

```python
data = pd.read_stata('demo.dta')
```

## Pickled 文件

python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化。python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。

```python
import pickle
with open('pickled_demo.pkl', 'rb') as file:
   pickled_data = pickle.load(file) # 下载被打开被读取到的数据

# 与其相对应的操作是写入方法
pickle.dump()
```

## HDF5 文件

HDF5文件是一种常见的跨平台数据储存文件，可以存储不同类型的图像和数码数据，并且可以在不同类型的机器上传输，同时还有统一处理这种文件格式的函数库。
HDF5 文件一般以 .h5 或者 .hdf5 作为后缀名，需要专门的软件才能打开预览文件的内容

```python
import h5py
filename = 'H-H1_LOSC_4_v1-815411200-4096.hdf5'
data = h5py.File(filename, 'r')
```

## Matlab 文件

```python
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
```

## 关系型数据库

```python
from sqlalchemy import create_engine
engine = create_engine('sqlite://Northwind.sqlite')

# 使用table_names()方法获取一个表名列表
table_names = engine.table_names()

# 直接查询关系型数据库
con = engine.connect()
rs = con.execute("SELECT * FROM Orders")
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()
# 使用上下文管理器 -- with
with engine.connect() as con:
    rs = con.execute("SELECT OrderID FROM Orders")
    df = pd.DataFrame(rs.fetchmany(size=5))
    df.columns = rs.keys()
# 使用Pandas查询关系型数据库
df = pd.read_sql_query("SELECT * FROM Orders", engine)
```

# 距离度量方法

距离度量用于计算给定问题空间中两个对象之间的差异，即数据集中的特征，然后可以使用该距离来确定特征之间的相似性， 距离越小特征越相似。

对于距离的度量，我们可以在几何距离测量和统计距离测量之间进行选择，应该选择哪种距离度量取决于数据的类型。特征可能有不同的数据类型（例如，真实值、布尔值、分类值），数据可能是多维的或由地理空间数据组成。

## 欧氏距离 Euclidean Distance

欧氏距离度量两个实值向量之间的最短距离，由于其直观，使用简单和对许多用例有良好结果，所以它是最常用的距离度量和许多应用程序的默认距离度量。

欧氏距离也可称为 l2 范数，其计算方法为:
$$
d = \sqrt{\sum_{i=1}^n(x_i-y_i)^2}
$$

```python
from scipy.spatial import distance
distance.euclidean(vector_1, vector_2)
```

欧氏距离有两个主要缺点

1. 距离测量不适用于比2D或3D空间更高维度的数据
2. 如果我们不将特征规范化和/或标准化，距离可能会因为单位的不同而倾斜

## 曼哈顿距离 Manhattan Distance

曼哈顿距离也被称为出租车或城市街区距离，因为两个实值向量之间的距离是根据一个人只能以直角移动计算的。这种距离度量通常用于离散和二元属性，这样可以获得真实的路径。

曼哈顿距离以 l1 范数为基础，计算公式为:
$$
d=\sum_{i=1}^n(x_i-y_i)
$$

```python
from scipy.spatial import distance
distance.cityblock(vector_1, vector_2)
```

曼哈顿的距离有两个主要的缺点

1. 它不如高维空间中的欧氏距离直观
2. 它也没有显示可能的最短路径

## 切比雪夫距离 Chebyshev Distance

切比雪夫距离也称为棋盘距离，因为它是两个实值向量之间任意维度上的最大距离。它通常用于仓库物流中，其中最长的路径决定了从一个点到另一个点所需的时间。

切比雪夫距离由 l - 无穷范数计算:
$$
d=max_i(|x_i-y_i|)
$$

```python
from scipy.spatial import distance
distance.chebyshev(vector_1, vector_2)
```

切比雪夫距离只有非常特定的用例，因此很少使用。

## 闵可夫斯基距离 Minkowski Distance

闵可夫斯基距离是上述距离度量的广义形式。它可以用于相同的用例，同时提供高灵活性。我们可以选择 p 值来找到最合适的距离度量。

闵可夫斯基距离的计算方法为:
$$
d=\sqrt[p]{\sum_{i=1}^n(x_i-y_i)^p} \\
Chebyshev\;Distance->p=\infty \quad Manhattan\;Distance->p=1 \quad Euclidean\;Distance->p=2
$$

```python
 from scipy.spatial import distance
 distance.minkowski(vector_1, vector_2, p)
```

由于闵可夫斯基距离表示不同的距离度量，它就有与它们相同的主要缺点，例如在高维空间的问题和对特征单位的依赖。此外，p值的灵活性也可能是一个缺点，因为它可能降低计算效率，因为找到正确的p值需要进行多次计算。

## 余弦相似度 Cosine Similarity

余弦相似度是方向的度量，他的大小由两个向量之间的余弦决定，并且忽略了向量的大小。余弦相似度通常用于与数据大小无关紧要的高维，例如，推荐系统或文本分析

余弦相似度可以介于-1(相反方向)和1(相同方向)之间，余弦相似度常用于范围在0到1之间的正空间中。余弦距离就是用1减去余弦相似度，位于0(相似值)和1(不同值)之间，计算方法为:
$$
Sim(u,v)=\frac{u^Tv}{||u||_2||v||_2}=cos\theta
$$

```python
from scipy.spatial import distance
distance.cosine(vector_1, vector_2)
```

余弦距离的主要缺点是它不考虑大小而只考虑向量的方向。因此，没有充分考虑到值的差异。

## 半正矢距离 Haversine Distance

半正矢距离测量的是球面上两点之间的最短距离。因此常用于导航，其中经度和纬度和曲率对计算都有影响

半正矢距离的公式如下：
$$
d=2r·arcsin(\sqrt{sin^2(\frac{\varphi_2-\varphi_1}{2})+cos\varphi_1cos\varphi_2sin^2(\frac{\lambda_2-\lambda_1}{2})}) \\
r\;为球面半径，\varphi\;为经度，\lambda\;为纬度
$$

```python
from sklearn.metrics.pairwise import haversine_distances
haversine_distances([vector_1, vector_2])
```

半正矢距离的主要缺点是假设是一个球体，而这种情况很少出现。

## 汉明距离

汉明距离衡量两个二进制向量或字符串之间的差异，对向量按元素进行比较，并对差异的数量进行平均，如果两个向量相同，得到的距离是0，如果两个向量完全不同，得到的距离是1。

```python
from scipy.spatial import distance
distance.hamming(vector_1, vector_2)
```

汉明距离有两个主要缺点：距离测量只能比较相同长度的向量，它不能给出差异的大小。所以当差异的大小很重要时，不建议使用汉明距离。



# 回归评价指标

## Mean Absolute Error(MAE)

平均绝对误差（Mean Absolute Error，MAE），也称为 L1 损失，是最简单的损失函数之一，也是一种易于理解的评估指标。它是通过取预测值和实际值之间的绝对差值并在整个数据集中取平均值来计算的。从数学上讲，它是绝对误差的算术平均值。MAE 仅测量误差的大小，不关心它们的方向。MAE越低，模型的准确性就越高。
$$
MAE=\frac{1}{n}\sum_{i=1}^n|y_i-\widehat{y}_i|
$$

- 优点
  - 由于采用了绝对值，因此所有误差都以相同的比例加权。
  - 如果训练数据有异常值，MAE 不会惩罚由异常值引起的高错误。
  - 它提供了模型执行情况的平均度量。
- 缺点
  - 有时来自异常值的大错误最终被视为与低错误相同。
  - 在零处不可微分。许多优化算法倾向于使用微分来找到评估指标中参数的最佳值。在 MAE 中计算梯度可能具有挑战性。

```python
def mean_absolute_error(true, pred):
    abs_error = np.abs(true - pred)
    sum_abs_error = np.sum(abs_error)
    mae_loss = sum_abs_error / true.size
    return mae_loss
```

## Mean Bias Error (MBE)

平均偏差误差是测量过程高估或低估参数值的趋势。偏差只有一个方向，可以是正的，也可以是负的。正偏差意味着数据的误差被高估，负偏差意味着误差被低估。平均偏差误差 是预测值与实际值之差的平均值。该评估指标量化了总体偏差并捕获了预测中的平均偏差。它几乎与 MAE 相似，唯一的区别是这里没有取绝对值。这个评估指标应该小心处理，因为正负误差可以相互抵消。
$$
MAE=\frac{1}{n}\sum_{i=1}^n(y_i-\widehat{y}_i)
$$

- 优点
  - 想检查模型的方向（即是否存在正偏差或负偏差）并纠正模型偏差，MBE 是一个很好的衡量标准。
- 缺点
  - 就幅度而言，这不是一个好的衡量标准，因为误差往往会相互补偿。
  - 它的可靠性不高，因为有时高个体错误会产生低MBE。
  - 作为一种评估指标，它在一个方向上可能始终是错误的。

```python
def mean_bias_error(true, pred):
    bias_error = true - pred
    mbe_loss = np.mean(np.sum(diff) / true.size)
    return mbe_loss
```

## Relative Absolute Error (RAE)

相对绝对误差是通过将总绝对误差除以平均值和实际值之间的绝对差来计算的。RAE并以比率表示，RAE的值从0到1。一个好的模型将具有接近于零的值，其中零是最佳值。
$$
RAE=\frac{\sum_{i=1}^n|y_i-\widehat y_i|}{\sum_{i=1}^n|y_i-\overline y_i|},\;\overline y=\frac{1}{n}\sum_{i=1}^ny_i
$$

- 优点
  - RAE 可用于比较以不同单位测量误差的模型。
  - RAE 是可靠的，因为它可以防止异常值。

```python
def relative_absolute_error(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.abs(true - pred))
    squared_error_den = np.sum(np.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss
```

## Mean Absolute Percentage Error (MAPE)

平均绝对百分比误差是通过将实际值与预测值之间的差值除以实际值来计算的。MAPE 也称为平均绝对百分比偏差，随着误差的增加而线性增加。MAPE 越小，模型性能越好。
$$
MAPE=\frac{1}{n}\sum_{i=1}^n\frac{|y_i-\widehat y_i|}{y_i}·100\%
$$

- 优点

  - MAPE与变量的规模无关，因为它的误差估计是以百分比为单位的。

  - 所有错误都在一个共同的尺度上标准化，很容易理解。

  - MAPE避免了正值和负值相互抵消的问题。

- 缺点

  - 分母值为零时，面临着“除以零”的问题。
  - MAPE对数值较小的误差比对数值大的误差错误的惩罚更多。
  - 因为使用除法运算，所欲对于相同的误差，实际值的变化将导致损失的差异。

```python
def mean_absolute_percentage_error(true, pred):
    abs_error = (np.abs(true - pred)) / true
    sum_abs_error = np.sum(abs_error)
    mape_loss = (sum_abs_error / true.size) * 100
    return mape_loss
```

## Mean Squared Error (MSE)

均方误差也称为 L2 损失，MSE通过将预测值和实际值之间的差平方并在整个数据集中对其进行平均来计算误差。MSE 也称为二次损失，因为惩罚与误差不成正比，而是与误差的平方成正比。平方误差为异常值赋予更高的权重，从而为小误差产生平滑的梯度。

MSE 永远不会是负数，因为误差是平方的。误差值范围从零到无穷大。MSE 随着误差的增加呈指数增长。一个好的模型的 MSE 值接近于零。
$$
MSE=\frac{1}{n}\sum_{i=1}^n(y_i-\widehat y_i)^2
$$

- 优点
  - MSE会得到一个只有一个全局最小值的梯度下降。
  - 对于小的误差，它可以有效地收敛到最小值。没有局部最小值。
  - MSE 通过对模型进行平方来惩罚具有巨大错误的模型。
- 缺点
  - 对异常值的敏感性通过对它们进行平方来放大高误差。
  - MSE会受到异常值的影响，会寻找在整体水平上表现足够好的模型。

```python
def mean_squared_error(true, pred):
    squared_error = np.square(true - pred) 
    sum_squared_error = np.sum(squared_error)
    mse_loss = sum_squared_error / true.size
    return mse_loss
```

## Root Mean Squared Error (RMSE)

RMSE 是通过取 MSE 的平方根来计算的。RMSE 也称为均方根偏差。它测量误差的平均幅度，并关注与实际值的偏差。RMSE 值为零表示模型具有完美拟合。RMSE 越低，模型及其预测就越好。
$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\widehat y_i)^2}
$$

- 优点

  - 易于理解，计算方便

- 缺点

  - 建议去除异常值才能使其正常运行。

  - 会受到数据样本大小的影响。

```python
def root_mean_squared_error(true, pred):
    squared_error = np.square(true - pred) 
    sum_squared_error = np.sum(squared_error)
    rmse_loss = np.sqrt(sum_squared_error / true.size)
    return rmse_loss
```

## Relative Squared Error (RSE)

相对平方误差需要使用均方误差并将其除以实际数据与数据平均值之间的差异的平方。
$$
RAE=\frac{\sum_{i=1}^n(y_i-\widehat y_i)^2}{\sum_{i=1}^n(y_i-\overline y_i)^2},\;\overline y=\frac{1}{n}\sum_{i=1}^ny_i
$$
优点：对预测的平均值和规模不敏感。

```python
def relative_squared_error(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.square(true - pred))
    squared_error_den = np.sum(np.square(true - true_mean))
    rse_loss = squared_error_num / squared_error_den
    return rse_loss
```



# Pytorch基础

## 步骤1：创建数据

Tensors张量是一种特殊的数据结构，它和数组还有矩阵十分相似。在Pytorch中，Tensors可以在gpu或其他专用硬件上运行来加速计算之外，其他用法类似Numpy。

```python
import torch
import numpy as np
# 直接从数据创建
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
x_data.shape
# 全为1
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# 全为0
x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# 查看tensor类型
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

## 步骤2：自动梯度计算

在Pytorch中可以使用tensor进行计算，并最终可以从计算得到的tensor计算损失，并进行梯度信息。在Pytorch中主要关注正向传播的计算即可。

```python
# x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([[1, 2], [3, 4]], dtype=float, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)  # y就多了一个AddBackward

z = y * y * 3
out = z.mean()

print(z)  # z多了MulBackward
print(out)  # out多了MeanBackward

# 计算公式：out = 0.25 ((x+2) * (x+2) * 3)
out.backward()
print(x.grad)
```

## 步骤3：拟合曲线

接下来我们将尝试使用Pytorch拟合一条曲线，我们首先的创建待你和的参数，并加载待训练的数据。

```python
# 需要计算得到的参数
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)

# 数据
x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# 目标模型
# y = wx + b
# 定义损失
def mse(label, pred):
    diff = label - pred
    return torch.sqrt((diff ** 2).mean())

pred = x_tensor * w + b
loss = mse(y_tensor, pred)
# 执行20次参数更新
for _ in range(20):

    # 重新定义一下，梯度清空
    w = w.clone().detach().requires_grad_(True)
    b = b.clone().detach().requires_grad_(True)

    # 正向传播
    pred = x_tensor * w + b
    
    # 计算损失
    loss = mse(y_tensor, pred)
    print(loss)

    # 计算梯度
    loss.backward()
```

## 步骤4：加载MNIST数据集

torchvision是pytorch官方的用于视觉任务的库，这里我们加载最常见的MNST数据集。当然也可以自定义数据读取。

```python
# torchvision 是pytorch官方的用于视觉任务的库
import torchvision.datasets as datasets  # 内置的数据集读取
import torchvision.transforms as transforms  # 内置的对图像的操作
from torch import nn

# 组合多个数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 图片的读取，（图片、类别）
# 28 * 28，数字0、1、2、3、4、5、6、7、8、9
dataset1 = datasets.MNIST('./', train=True, download=True)
dataset2 = datasets.MNIST('./', train=False, download=True)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=40)
test_loader = torch.utils.data.DataLoader(dataset2,  batch_size=40)
```

## 步骤5：定义全连接网络

接下来我们定义网络结构，由于是图像分类任务，因此我们的节点维度使用逐步降低的定义。

```python
net = nn.Sequential(
    nn.Flatten(), # 将维度转换为二维
    nn.Linear(784, 256), # 全连接层
    nn.ReLU(), # 激活函数
    nn.Linear(256, 10) # 全连接层
)
```

## 步骤6：训练卷积神经网络

如果需要定义CNN网络，则可以参考如下的方式。先定义卷积层，然后定义全连接层。

```python
import torch
from torch import nn

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)
```

## 步骤7：模型训练

定义训练时的超参数，如batch size、学习率和优化器。这里可以自定定义。

```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=lr)
train_acc, test_acc = [], []
# epoch维度训练
for _ in range(num_epochs):
    acc = 0
    
    # 读取训练数据
    # batch维度训练
    for data in train_loader:
        pred = net(data[0]) # 正向传播
        pred_loss = loss(pred, data[1]) # 计算损失
        updater.zero_grad() # 清空梯度
        pred_loss.backward()  # 梯度计算
        updater.step()  # 参数更新
        
        # 累计准确样本个数
        acc += (pred.argmax(1) == data[1]).sum()
    
    # 计算准确率
    acc = acc.float() / len(train_loader.dataset)
    train_acc.append(acc)
    
    # 读取验证数据
    # batch维度预测
    with torch.no_grad(): # 不记录梯度信息
        acc = 0
        for data in test_loader:
            pred = net(data[0]) # 正向传播
            pred_loss = loss(pred, data[1]) # 累计梯度
            acc += (pred.argmax(1) == data[1]).sum() # 累计准确样本个数
        
        # 计算准确率
        acc = acc.float() / len(test_loader.dataset)
        test_acc.append(acc)

    print(train_acc[-1], test_acc[-1])
```



# BERT的Pooling方法

从模型复杂度上：AttentionPooling > WeightedLayerPooling >  MeanPooling / MinPooling / MaxPooling

从模型精度上：AttentionPooling > WeightedLayerPooling > MeanPooling > MaxPooling > MinPooling

使用多种Pooling的目的是增加BERT模型的多样性，考虑在模型集成中使用

## MeanPooling

将每个token对应的输出计算均值，这里需要考虑attention_mask，也就是需要考虑有效的输入的token

```python
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings
```

## MaxPooling

将每个token对应的输出计算最大值，这里需要考虑attention_mask，也就是需要考虑有效的输入的token

```python
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
```

## MinPooling

将每个token对应的输出计算最小值，这里需要考虑attention_mask，也就是需要考虑有效的输入的token

```python
class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings
```

## WeightedPooling

将每个token对应的输出计算出权重，这里的权重可以通过特征进行计算，也可以考虑通过IDF计算出权重

```python
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average
```

## AttentionPooling

将每个token的特征单独加入一层，用于注意力的计算，增加模型的建模能力

```python
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings
```



# 损失函数

## L1范数损失 L1Loss

计算 output 和 target 之差的绝对值。

```python
torch.nn.L1Loss(reduction='mean')
```

参数：reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

## 均方误差损失 MSELoss

计算 output 和 target 之差的均方差

```python
torch.nn.MSELoss(reduction='mean')
```

参数：reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

## 交叉熵损失 CrossEntropyLoss

当训练有 C 个类别的分类问题时很有效. 可选参数 weight 必须是一个1维 Tensor, 权重将被分配给各个类别. 对于不平衡的训练集非常有效。

在多分类任务中，经常采用 softmax 激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。所以需要 softmax激活函数将一个向量进行“归一化”成概率分布的形式，再采用交叉熵损失函数计算 loss。

```python
torch.nn.CrossEntropyLoss(weight=None,ignore_index=-100, reduction='mean')
```

参数

weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor

ignore_index (int, optional) – 设置一个目标值, 该目标值会被忽略, 从而不会影响到 输入的梯度。

reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

##  KL 散度损失 KLDivLoss

计算 input 和 target 之间的 KL 散度。KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上(离散采样)上进行直接回归时很有效

```python
torch.nn.KLDivLoss(reduction='mean')
```

参数：reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

## Others

可参考：https://mp.weixin.qq.com/s/-fgZZ4nBdKsBeCsmDBVVsA



# PyTorch-dataloader

PyTorch的dataloader是用于读取训练数据的工具，它可以自动将数据分割成小batch，并在训练过程中进行数据预处理。

以下是制作PyTorch的dataloader的简单步骤：

1. 导入必要的库。定义数据集类，需要自定义一个继承自`torch.utils.data.Dataset`的类，在该类中实现`__len__`和`__getitem__`方法

   ```python
   import torch
   from torch.utils.data import DataLoader, Dataset
   
   class MyDataset(Dataset):
       def __init__(self, data):
           self.data = data
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, index):
           # 返回第index个数据样本
           return self.data[index]
   ```

2. 创建数据集实例

   ```python
   data = [1, 2, 3, 4, 5]
   dataset = MyDataset(data)
   ```

3. 创建dataloader实例。使用`torch.utils.data.DataLoader`创建dataloader实例，可以设置`batch_size、shuffle`等参数

   ```python
   dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
   ```

4. 使用dataloader读取数据。至此是制作PyTorch的dataloader的简单步骤，根据实际需求可以进行更复杂的操作，如数据增强、并行读取等。

   ```python
   for batch in dataloader:
       # batch为一个batch的数据，可以直接用于训练
       print(batch)
   ```

5. 已经分类的文件生成标注文件

   假设你已经将所有的图片按照类别分别放到了十个文件夹中，可以使用以下代码生成标注文件：

   ```python
   import os
   # 定义图片所在的文件夹路径和标注文件的路径
   img_dir = '/path/to/image/directory'
   ann_file = '/path/to/annotation/file.txt'
   # 遍历每个类别文件夹中的图片，将标注信息写入到标注文件中
   with open(ann_file, 'w') as f:
       for class_id in range(1, 11):
           class_dir = os.path.join(img_dir, 'class{}'.format(class_id))
           for filename in os.listdir(class_dir):
               if filename.endswith('.jpg'):
                   # 写入图片的文件名和类别
                   f.write('{} {}\n'.format(filename, class_id))
   ```

   在上述代码中，首先定义了图片所在的文件夹路径`img_dir`和标注文件的路径`ann_file`

   然后，使用`with open(ann_file, 'w') as f:`语句打开标注文件，使用`for`循环遍历每个类别文件夹中的图片，并将标注信息写入到标注文件中

   其中，`os.path.join`函数用于拼接路径字符串，`f.write`函数用于将图片的文件名和类别写入到标注文件中，且每个标注信息占据一行，文件名和类别之间使用空格分隔。需要注意的是，上述代码假设每个类别文件夹的名称为`class1`、`class2`、...、`class10`，图片文件名的后缀为`.jpg`，且标注文件中每行仅包含一个文件名和一个标签，且它们之间使用空格分隔。如果文件夹名称、文件名后缀或标注文件格式不同，需要对代码进行相应的修改

   生成的标注文件是一个文本文件，每行包含一个图片的文件名和类别标签，两者之间使用空格分隔。举个例子，如果第一个文件夹中有三张图片，它们的文件名分别为`img_001.jpg`、`img_002.jpg`和`img_003.jpg`，类别标签为1，则生成的标注文件内容如下：

   ```python
   img_001.jpg 1
   img_002.jpg 1
   img_003.jpg 1
   ```

   这个标注文件可以被用作训练深度学习模型时的标签数据

6. 图像读取示例

   如果数据集已经按照类别分好了文件夹，我们可以使用`torchvision.datasets.ImageFolder`类来读取数据集。`ImageFolder`类会自动将每个文件夹中的图像按照类别进行标记，并且支持数据增强和数据预处理等操作。以下是一个示例，展示如何使用`ImageFolder`类读取数据集，并使用`DataLoader`批量加载数据集：

   ```python
   import torch
   import torchvision
   from torchvision import transforms
   # 数据增强和预处理
   transform = transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   # 读取数据集
   dataset = torchvision.datasets.ImageFolder('path/to/data', transform=transform)
   # 创建数据加载器
   dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
   # 使用数据加载器迭代数据集
   for batch_data, batch_labels in dataloader:
       print(batch_data.shape)
       print(batch_labels.shape)
   ```

   在上述代码中，我们首先定义了数据增强和预处理的操作，然后使用`ImageFolder`类读取数据集，将数据增强和预处理操作传递给`transform`参数

   `ImageFolder`类会自动将图像按照类别进行标记

   然后，我们使用`DataLoader`将数据集打包成批量，每个批量大小为32，并且开启了shuffle功能和4个线程

   最后，我们使用`for`循环迭代数据加载器，逐批加载数据，并输出每个批量的数据和标签

   需要注意的是，使用`ImageFolder`类前需要将数据集的文件夹按照类别进行命名，例如两个文件夹的名字分别为`class1`和`class2`。另外，`transforms.Normalize`中的`mean`和`std`参数需要根据数据集进行调整

   

# ubuntu clash

1. 导出 config.yml 文件
2. 替换 /etc/clash/config.yml 文件，具体可参考 temp.yml
3. systemctl status/stop/start/restart clashd.service
