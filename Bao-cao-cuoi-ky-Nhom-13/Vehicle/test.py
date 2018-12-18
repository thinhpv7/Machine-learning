

from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load model
model = load_model('model5.h5')
print('load ok')
# data tự chuẩn bị

img = Image.open('car_test.jpg').convert('RGB')
img = img.resize((64,64))
imgArr = np.array(img)
print(imgArr.shape)

# shape that CNN expects is a 4D array (batch, height, width, channels) <batch: số samples>
imgArr = imgArr.reshape(1,imgArr.shape[0],imgArr.shape[1],3)
#print(imgArr)
plt.imshow(img)
plt.show()

####################################################################### Predicting the Test set results
y_pred = model.predict(imgArr)

print("Predict results: \n", y_pred[0])
print("Predict lable: \n", np.argmax(y_pred[0]))


if(np.argmax(y_pred[0]) == 0):
    print("Predict lable: car")
        
else:
     print("Predict lable: motobike")
     

# print(y_pred)

