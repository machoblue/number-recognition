import sys, os
sys.path.append(os.pardir)
import numpy as np
from two_layer_net import TwoLayerNet
#from dataset.mnist import load_mnist
from functions import *
import pickle

from PIL import Image


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

with open('params.pickle', mode='rb') as f:
  network.refresh(pickle.load(f))
  
def predict(image):
  return softmax(network.predict(image))
  
#  return np.around(softmax(network.predict(image)), decimals=3)
  
# test
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # t:teacher
#
#one_of_x_test = x_test[20]
#one_of_t_test = t_test[20]
#print('t:', str(one_of_t_test))
#print('r:', str(infer(one_of_x_test)))

#image = np.asarray(Image.open('./images/15.jpg'))
#print(image.shape)
#image = image.reshape(784,)
#print('r:', str(infer(image)))
