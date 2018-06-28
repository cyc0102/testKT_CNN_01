from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)


from keras.models import load_model

# 載入模型
model = load_model('model.h5')

scores = model.evaluate(x_Test4D_normalize , y_TestOneHot)
print('Test accuracy:', scores[1])

prediction=model.predict_classes(x_Test4D_normalize)
print('First 10 images are', prediction[:10])
