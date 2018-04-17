import numpy as np

def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)
  
  print("range:" + str(range(x.size)))
  print("shape:" + str(x.shape))
    
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = float(tmp_val) + h
    fxh1 = f(x) # f(x+h)
        
    x[idx] = tmp_val - h 
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)
    
    x[idx] = tmp_val # 値を元に戻す
    it.iternext()   
        
  return grad
  
#  for idx in range(x.size):
##    print("idx:" + str(idx))
#    tmp_val = x[idx]
#    
#    x[idx] = tmp_val + h
#    fxh1 = f(x)
#    
#    x[idx] = tmp_val - h
#    fxh2 = f(x)
#    
#    grad[idx] = (fxh1 - fxh2) / (2*h)
#    x[idx] = tmp_val
#    
#  return grad
