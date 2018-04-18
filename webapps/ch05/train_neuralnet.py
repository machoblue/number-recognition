import sys, os
sys.path.append(os.pardir)
import numpy as np
#import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
#iters_nums = [] # edit
#epochs = [] # edit

#iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]
  
  grad = network.gradient(x_batch, t_batch)
  
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]
  
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)
#  iters_nums.append(i) # edit
  
  # 
  if (i % 100 == 0):
    print('i:%d, loss:%f' % (i, loss))
  #
  
  if i % 1000 == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print('train_acc:%f, test_acc:%f' % (train_acc, test_acc))
#    epochs.append(i)

#x = np.array(iters_nums)
#y = np.array(train_loss_list)
#plt.plot(x, y)
#plt.show()
#
#x1 = np.array(epochs);
#y1 = np.array(test_acc_list);
#plt.plot(x1, y1)
#plt.show()

# pickle
with open('params.pickle', mode='wb') as f:
  pickle.dump(network.params, f)