## Question

已知条件
有某种网络设备有N个出端口，每个出端口的带宽分别为B(x)(1≤ x ≤N)，不同端口的传输带宽可能不同。流进入设备后进入调度区，此处可以指定流什么时间发送，从哪个出端口发送。一个出端口可以同时发送多条流，每个出端口下的流在发送过程中会占用固定大小的带宽S(y )(1≤ y ≤M,M为正整数)，发送流所需要的时间为T(y) (1≤ y ≤M,M为正整数)，不同的流占用带宽可能不等，时间不等。每个端口都有一个排队区，当流被调度进入出端口时，若此时出端口带宽已被发送的流占满或者排队区中已经存在排队等待的流，新的流会进入排队区进行等待，否则会立即发出。

赛题数据：
端口文件：port.txt
端口文件第一行是列信息介绍，数据从第二行开始
格式：
端口id,端口带宽
0,1000
1,9000
2,4000
3,2000
4,7000

流文件：flow.txt
流信息文件第一行是列信息介绍，数据从第二行开始
格式：
流id,流带宽,进入设备时间,发送流所需时间
0,208,10,20
1,486,9,97
2,465,9,58
3,358,4,2
4,97,8,74

输出文件：result.txt
给定每个流对应端口id,流的开始发送时间
格式：
流id,端口id,开始发送时间
0,4,1
2,2,1
1,2,2

评分规则：
1、传输完所有流的总时间越短越好；
2、总时间相等的基础上，提交的可执行程序在验收数据集上运行的时间越短越好；
3、若有结果中有流缺失，流给定的端口id不存在，流开始发送时间小于进入设备时间等其他异常数据，该结果作废；
4、程序在跑分平台运行总时长5分钟内，空间限制3G，该时间内需要处理多份样本文件，超时只会计算已有输出的样本成绩，未跑完的样本成绩为0。

请给出满足条件与规则的基于线性规划算法的python代码



## Solution

由于题目中涉及到调度问题，我们可以使用线性规划（LP）模型来求解。LP模型是优化问题的常见方法，可以将问题转化为线性方程组求解，具有较好的可行性和计算效率。

首先，我们需要定义变量。设 $x_{ij}$ 为第 $i$ 个流走第 $j$ 个端口的流量， $y_{ij}$ 为第 $i$ 个流在第 $j$ 个端口的开始发送时间。

接下来，我们需要定义目标函数和约束条件。

目标函数：最小化总时间。由于该问题涉及到流量和时间两个变量，我们可以将目标函数分为两部分，分别计算总流量和总时间：

$$\text{Minimize} \quad \sum_{i=1}^{n} \sum_{j=1}^{m} x_{ij}T_i + \sum_{i=1}^{n} \sum_{j=1}^{m} y_{ij}$$

其中， $n$ 为流的数量， $m$ 为端口的数量， $T_i$ 表示第 $i$ 个流的传输时间。

约束条件：

1. 流量约束：每个流的总流量不能超过其带宽，即：

$$\sum_{j=1}^{m} x_{ij} \leq B_i, \quad i=1,2,\dots,n$$

1. 端口带宽约束：每个端口的总流量不能超过其带宽，即：

$$\sum_{i=1}^{n} x_{ij} \leq S_j, \quad j=1,2,\dots,m$$

1. 发送时间约束：每个流的开始发送时间必须不小于其进入设备时间，并且下一个流的开始发送时间必须不小于上一个流的发送时间加上其传输时间，即：

$$y_{i,j}-y_{i',j}+T_i\geq 0, \quad i<i', j=1,2,\dots,m$$

$$y_{i,j}-t_i\geq 0, \quad i=1,2,\dots,n, j=1,2,\dots,m$$

其中， $t_i$ 表示第 $i$ 个流进入设备的时间。

1. 非负约束：所有变量必须非负，即：

$$x_{ij}, y_{ij} \geq 0, \quad i=1,2,\dots,n, j=1,2,\dots,m$$

将上述目标函数和约束条件转化为LP模型，可以使用Python中的PuLP库求解。具体实现代码如下：

```python
import pulp

# 读取输入文件
def read_input():
    with open('port.txt', 'r') as f:
        ports = f.readlines()[1:]
        ports = {int(p.split(',')[0]): int(p.split(',')[1]) for p in ports}
    with open('flow.txt', 'r') as f:
        flows = f.readlines()[1:]
        flows = [(int(f.split(',')[0]), int(f.split(',')[1]), int(f.split(',')[2]), int(f.split(',')[3])) for f in flows]
    return ports, flows

# 定义LP模型
def define_model(ports, flows):
    # 创建LP问题
    prob = pulp.LpProblem('Scheduling Problem', pulp.LpMinimize)
    
    # 定义决策变量
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(len(flows)) for j in range(len(ports))], lowBound=0, cat='Continuous')
    y = pulp.LpVariable.dicts('y', [(i, j) for i in range(len(flows)) for j in range(len(ports))], lowBound=0, cat='Continuous')
    
    # 定义目标函数
    prob += pulp.lpSum([x[(i, j)] * flows[i][3] for i in range(len(flows)) for j in range(len(ports))]) + \
            pulp.lpSum([y[(i, j)] for i in range(len(flows)) for j in range(len(ports))])
    
    # 添加约束条件
    # 流量约束
    for i in range(len(flows)):
        prob += pulp.lpSum([x[(i, j)] for j in range(len(ports))]) <= flows[i][1]
    
    # 端口带宽约束
    for j in range(len(ports)):
        prob += pulp.lpSum([x[(i, j)] for i in range(len(flows))]) <= ports[j]
    
    # 发送时间约束
    for i in range(1, len(flows)):
        for j in range(len(ports)):
            prob += y[(i, j)] - y[(i-1, j)] + flows[i-1][3] * x[(i-1, j)] >= 0
    for i in range(len(flows)):
        for j in range(len(ports)):
            prob += y[(i, j)] - flows[i][2] >= 0
    
    return prob, x, y

# 求解LP模型
def solve_model(prob):
    prob.solve()
    return prob

# 输出结果
def write_output(prob, x, y):
    with open('result.txt', 'w') as f:
        f.write('流id,端口id,开始发送时间\n')
        for i in range(len(flows)):
            for j in range(len(ports)):
                if x[(i, j)].varValue > 0:
                    f.write('{},{},{}\n'.format(i, j, int(y[(i, j)].varValue)))
        f.close()

# 主函数
if __name__ == '__main__':
    ports, flows = read_input()
    prob, x, y = define_model(ports, flows)
    prob = solve_model(prob)
    write_output(prob, x, y)
```

以上代码将端口和流信息读入后，定义LP模型并求解，最后将结果输出到result.txt中。请将上述代码保存为schedule.py，然后在命令行中执行以下命令即可运行该程序：

```bash
python schedule.py
```