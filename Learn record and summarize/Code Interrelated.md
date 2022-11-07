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
