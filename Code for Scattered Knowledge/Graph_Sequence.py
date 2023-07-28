import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 判断图序列
def is_graph(seq):
    seq = sorted(seq, reverse=True) # 降序排序
    while seq:
        v = seq.pop(0)  # 取出最大值
        if v > len(seq): # 如果最大值大于序列长度，说明无法构成简单图，返回False
            return False
        for i in range(v): # 将序列中前v个元素减1
            seq[i] -= 1
            if seq[i] < 0: # 如果出现负数，说明不是图序列，返回False
                return False
        seq = sorted(seq, reverse=True) # 对序列进行降序排序
    return True # 如果序列中所有元素都为0，说明是图序列，返回True


# 构建一个简单图
def build_graph(seq):
    n = len(seq)
    # 获取节点度数的最大值，用于后续判断是否存在孤立点
    max_deg = max(seq) 
    edges = set()
    degree = [0] * n  # 记录每个节点的度数
    isolated_nodes = []  # 记录孤立点
    for i in range(n):
        # 判断当前节点的度数是否小于等于前面节点度数之和
        if seq[i] >= i - sum(seq[:i]):
            # 如果当前节点度数为0，将其添加到孤立点列表中
            if seq[i] == 0:
                isolated_nodes.append(i + 1) 
            # 遍历当前节点后面的节点
            for j in range(i + 1, n):
                # 判断当前节点和后面节点的度数是否小于它们应有的度数，同时判断它们之间是否可以有边相连
                if degree[i] < seq[i] and degree[j] < seq[j] and (seq[i] == 0 or seq[j] == 0 or seq[i] + seq[j] >= j - i):
                    edge = (i + 1, j + 1)
                    # 判断这条边是否已经存在于边集合中，避免重复添加
                    if edge not in edges and (j + 1, i + 1) not in edges:
                        edges.add(edge)
                        degree[i] += 1
                        degree[j] += 1
                        if seq[j] == 0:
                            isolated_nodes.append(j + 1)  # 将度数为0的节点添加到孤立点列表中
        else:
            print("您输入的不是图序列！")
            return None
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from(edges)
    if isolated_nodes:
        isolated_nodes = sorted(isolated_nodes, reverse=True)
        for node in isolated_nodes:
            G.add_node(node)
    return G


# 显示一个简单图
def draw_graph(seq):
    # 判断输入的序列是否是合法的图序列
    if is_graph(seq):
        G = build_graph(seq)
        # 创建绘图空间
        fig = Figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        # 指定布局算法
        pos = nx.fruchterman_reingold_layout(G) 
        # 绘制图形，并添加标签
        nx.draw(G, pos=pos, ax=ax, with_labels=True)
        # 将绘图空间绑定到Tkinter窗口上
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        # 将绘图区域添加到Tkinter窗口中
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # 隐藏之前的图
        if hasattr(draw_graph, "canvas"):
            draw_graph.canvas.get_tk_widget().pack_forget()
        # 记录当前绘制的图形，以便下次更新
        draw_graph.canvas = canvas
    else:
        messagebox.showerror("错误", "您输入的不是图序列！")

# 当用户点击“提交”按钮时调用
def on_submit():
    seq = [int(x) for x in input_entry.get().split()]
    draw_graph(seq)

# 创建主窗口
root = tk.Tk()
root.title("判断序列是否为图序列")

# 创建输入框和按钮
input_label = tk.Label(root, text="请输入一个有限非负整数序列（用空格作为间隔）：")
input_label.pack()
input_entry = tk.Entry(root, width=50)
input_entry.pack()
submit_button = tk.Button(root, text="提交", command=on_submit)
submit_button.pack()
 # 进入主循环，等待用户交互事件发生
root.mainloop()
