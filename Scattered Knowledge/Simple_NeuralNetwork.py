# Reference:
# https://mp.weixin.qq.com/s/M0up_QPMfEQDYm-NmbB8jg

import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # mean squared error loss
  return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feed_forward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 10000

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # --- 做一个前馈
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- 计算偏导数。
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_y_pred = -2 * (y_true - y_pred)

        # Neuron o1
        d_y_pred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_y_pred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_y_pred_d_b3 = deriv_sigmoid(sum_o1)

        d_y_pred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_y_pred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- 更新权重和偏差
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_y_pred * d_y_pred_d_w5
        self.w6 -= learn_rate * d_L_d_y_pred * d_y_pred_d_w6
        self.b3 -= learn_rate * d_L_d_y_pred * d_y_pred_d_b3

      # --- 在每次epoch结束时计算总损失 
      if epoch % 10 == 0:
        # 函数原型：numpy.apply_along_axis(func, axis, arr, *args, **kwargs)
        # 将arr数组的每一个元素经过func函数变换形成的一个新数组，axis表示函数func对arr是作用于行还是列
        y_pred = np.apply_along_axis(self.feed_forward, 1, data)
        loss = mse_loss(all_y_trues, y_pred)
        print("Epoch %d loss: %.3f" % (epoch, loss))
        print("w1=%.3f, w2=%.3f, w3=%.3f, w4=%.3f, w5=%.3f, w6=%.3f, b1=%.3f, b2=%.3f, b3=%.3f\n" 
              % (self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.b1, self.b2, self.b3))

# dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# 训练神经网络
network = NeuralNetwork()
network.train(data, all_y_trues)

print("\n The train has finished, we have got the good parameter\n")

# predict
emily = np.array([-7, -3]) # 128 磅, 63 英寸
frank = np.array([20, 2])  # 155 磅, 68 英寸

def Judge_Gender(name, predict):
    if predict >= 0.5:
        print("%s is female, the probability of prediction is %.3f" % (name, predict))
    else:
        print("%s is male, the probability of prediction is %.3f" % (name, 1 - predict))

Judge_Gender("Emily", network.feed_forward(emily))
Judge_Gender("Frank", network.feed_forward(frank))