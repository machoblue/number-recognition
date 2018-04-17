#import tensorflow as tf
import sys,os
sys.path.append(os.pardir)
import numpy as np
from ch05.use_neuralnet import predict

from PIL import Image
from PIL import ImageOps

class ImageRec():
    
    def run(self, image_path):
        img = Image.open(image_path)
        img = img.resize((28, 28), Image.LANCZOS)
        img = img.convert('RGB')
        img = ImageOps.invert(img)
        img = img.convert("L")
        img.save('imagetest3.png')

        image = np.asarray(img)
        print(image.shape)
        image = image.reshape(784,)
      
        result = predict(image)
        print('result:%s' % result)
        
        image_info = []
        for i in range(10):
            label = str(i)
            score = result[i]
            print('%s (score = %.5f)' % (label, score))
            score_info = {}
            score_info['name']=label
            score_info['score']=round(score * 100.0, 2)
            image_info.append(score_info)

        return image_info