import numpy as np
import matplotlib.pyplot as plt

# 生成五阶单位主对角阵
np.eye(5)

# Read training set
train_data = np.loadtxt("./ex1data1.txt", delimiter=",")

# Plot the data distribution
populations = train_data[:, 0]
profits = train_data[:, 1]
m = len(profits)
print(m)

# suplot(nrows, ncols, sharex, sharey, subplot_kw, **fig_kw)
#
#       nrows : subplot的行数
#       ncols : subplot的列数
#      sharex : 所有subplot应该使用相同的X轴刻度（调节xlim将会影响所有subplot）
#      sharey : 所有subplot应该使用相同的Y轴刻度（调节ylim将会影响所有subplot）
#  subplot_kw : 用于创建各subplot的关键字字典
#    **fig_kw : 创建figure时的其他关键字，如plt.subplots(2, 2, figsize=(8, 6))
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.set_title('profit and population distribution')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

plt.scatter(populations, profits, color='red', marker='x')

plt.show()