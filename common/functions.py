import numpy as np
#np.seterr(invalid='raise')

def softmax(x):
  if x.ndim == 2:
    x = x.T # T:転置
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T
  
  print('x:', str(x))
  x = x - np.max(x) # オーバーフロー対策
#  print('max:', str(np.max(x)))
  print('x:', str(x))
  print('x1:', str(np.exp(x)))
  print('s1:', str(np.sum(np.exp(x))))
  return np.exp(x) / np.sum(np.exp(x))
#def softmax(a):
#  c = np.max(a)
#  exp_a = np.exp(a - c)
#  sum_exp_a = np.sum(exp_a)
#  y = exp_a / sum_exp_a
#  return y

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def cross_entropy_error(y, t):
  # 一次元の場合(サンプルではここには来ない)
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    
  # 
  if t.size == y.size:
    t = t.argmax(axis=1)
             
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
#  batch_size = y.shape[0]
#  return -np.sum(t * np.log(y)) / batch_size


