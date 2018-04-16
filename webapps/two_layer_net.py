import sys, os
sys.path.append(os.pardir) # parentdirをimportするため
import numpy as np # new 
from layers import * # new
from functions import *
from gradient import numerical_gradient
from collections import OrderedDict # new

class TwoLayerNet:
  
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    
    # new
    # ? なぜ2層？なぜReLU?
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Rulu1'] = ReLU()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    
    self.lastLayer = SoftmaxWithLoss()
  
  # new 
  def refresh(self, params_):
    self.params = params_
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Rulu1'] = ReLU()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    
    self.lastLayer = SoftmaxWithLoss()

    
  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    
    return x
#    W1, W2 = self.params['W1'], self.params['W2']
#    b1, b2 = self.params['b1'], self.params['b2']
#    
#    a1 = np.dot(x, W1) + b1
#    z1 = sigmoid(a1)
#    a2 = np.dot(z1, W2) + b2
#    y = softmax(a2)
#    
#    return y
  
  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)
#    y = self.predict(x)
#    return cross_entropy_error(y, t)
  
  def accuracy(self, x, t):
    y = self.predict(x)
    
    # argmax:return the indices of maximum values along an axis
    y = np.argmax(y, axis=1)
    if t.ndim != 1 : t = np.argmax(t, axis=1) # new
      
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)
    
    grad = {}
    grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
    
    return grad
  
  def gradient(self, x, t):
    # forward
    self.loss(x, t)
    
    # backward
    dout = 1
    dout = self.lastLayer.backward(dout)
    
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)
      
    grads = {}
    grads['W1'] = self.layers['Affine1'].dw
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dw
    grads['b2'] = self.layers['Affine2'].db
    
    return grads