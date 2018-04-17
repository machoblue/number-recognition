import sys, os
sys.path.append(os.pardir)
import numpy as np
#np.seterr(invalid='raise')
from common.functions import *
#from functions import softmax


class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.bd = None
  
  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b
    
    return out
  
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dw = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)
    
    return dx

class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None
    
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    return self.loss
  
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size
    return dx
  
class ReLU:
  def __init__(self):
    self.mask = None
    
  def forward(self, x):
    try:
      self.mask = (x <= 0)
    except:
      print("exception")
      print("x:", x)
    out = x.copy()
    out[self.mask] = 0
    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx