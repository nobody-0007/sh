#### MC-Pit
```python
#Step1: Generate a vector of inputs and a vector of weights
import numpy as np
np.random.seed(seed=0)
I=np.random.choice([0,1], 3)
W=np.random.choice([-1,1], 3)
print(f'Input Vector: {I}, Weight Vector{W}')

#step 2:  Calculate Dot Product
dot = I@W
print(f'Dot Product: {dot}')

#step 3: Define the threshold activation function
def linear_threshold_gate (dot: int, T: float) -> int:
    if dot >= T:
        return 1
    else:
        return 0

# Compute the output based on threshold value
T = 1
activation = linear_threshold_gate(dot, T)
print(f'Activation: {activation}')

# Compute the output based on threshold value
T = 1
activation = linear_threshold_gate(dot, T)
print(f'Activation: {activation}')

#and
#weights: all positive
#threshold: 2
input_table = np.array([[0,0], [0,1], [1,0], [1,1]])
print(f'input table:\n{input_table}')
weights = np.array([1,1])
print(f'weights: {weights}')
dot_product = input_table @ weights
print(f'Dot Product: {dot_product}')
#and
T = 2
for i in range(0,4):
    activation = linear_threshold_gate(dot_product[i], T)
    print(f'Activation: {activation}')


#nor
#weights: all positive
#threshold: 1
weights = np.array([-1,-1])
print(f'weights: {weights}')
dot_product = input_table @ weights
print(f'Dot Product: {dot_product}')
T = 0
for i in range(0,4):
    activation = linear_threshold_gate(dot_product[i], T)
    print(f'Activation: {activation}')
```
#### Hebbian
```python
import numpy as np
import pandas as pd
w=np.array([1,-1,0,0.5]).transpose()
Xi=[np.array([1,-2,1.5,0]).transpose(), np.array([1,-0.5,-2,-1.5]).transpose(), np.array([0,1,-1,1.5]).transpose()]
C=1
Iteration=0
print(Xi)

for i in range(len(Xi)):
    net=sum(w.transpose()*Xi[i])
    Fnet=np.sign(net)
    dw=C*Fnet*Xi[i]
    w=w+dw
    print('weight vector:',w)
    Iteration+=1

print('Final vector: ', w)
print('No. of iterations: ', i)
```
#### Perceptron
```python
import numpy as np
import pandas as pd

w=np.array([1,-1,0,0.5]).transpose()
Xi=[np.array([1,-2,0,1]).transpose(), np.array([0, 1.5, -0.5, -1]).transpose(), np.array([-1, 1, 0.5, -1]).transpose()]
c=0.1
d=[-1, -1, 1]
i=0
print(Xi)

for i in range(len(Xi)):
    net=sum(w.transpose()*Xi[i])
    o=np.sign(net)
    r=d[i]-o
    dw=c*r*Xi[i]
    w=w+dw
    print('weight vector:',w)
    i+=1

print('Final vector: ', w)
print('No. of iterations: ', i)
```
#### Delta
```python
# Delta Learning Rule
import numpy as np
import pandas as pd

W=np.array([1, -1, 0, 0.5]).transpose()
X=np.array([np.array([1, -2, 0, -1]).transpose(), np.array([0, 1.5, 0.5, -1]).transpose(), np.array([-1, 1, 0.5, -1]).transpose()])
C=0.1
d=[-1, -1, 1]
it=0

for i in range (len(X)):
  print('Iteartion: ', it)

  net = sum(W.transpose() * X[i])
  print('net = ', net)

  o = (2/(1+np.exp(-1*net)))-1
  o_ = (0.5) * (1-(o**2))

  error = d[i] - o
  Error = round(error, 1)
  print('Error = ', Error)

  dW = C * o_ * Error * X[i]

  W = W + dW
  print('W = ', W)
  print('\n')
  it+=1

```
#### SDPTA
```python
import numpy as np
import pandas as pd

W= np.array([1, -1, 0, 0.5]).transpose()

X = [np.array ([1, -2, 0, -1]).transpose(),
      np.array ([0, 1.5, 0.5, -1]).transpose(),
      np.array([-1, 1, 0.5, -1]).transpose()]
print('Initial Weights:', W)
print('X:', X)

d= [-1, -1, 1]
c = 0.1

error = 0
Error = 1

iteration = 0
i = 0
j = 0

while (Error != 0.0):

  net=sum(W.transpose()*X[i])

  o = 1
  if net < 0:
    o = -1

  error = error + 0.5 * ((d[i] - o) ** 2)
  Error = np.round(error, 1)

  print('Error:', Error)
  dw = c * (d[i] - o) * Xi[i]
  W = W + dw

  print('Updated Weights', i + 1, W)

  iteration += 1
  print('Iteration', i + 1, ':', iteration)

  i += 1
  if i > 2:
    i = 0
    j += 1
    error = 0

  if Error == 0.0:
    break

print("Final Weight Matrix:{}".format(W))
print("Cycle Epoch Counter:{}".format(j))
print("Iteration:{}".format(iteration))
```
#### SCPTA
```python
import numpy as np
import pandas as pd

W= np.array([1, 1, 0, 0.5]).transpose()

Xi = [np.array ([1, 2, 0, -1]).transpose(),
      np.array ([0, 1.5, 0.5, -1]).transpose(),
      np.array([-1, 1, 0.5, -1]).transpose()]
print('Initial Weights:', W)
print('X:', Xi)

d= [-1, -1, 1]
c = 0.1

error = 0
Error = 1

iteration = 0
i = 0
j = 0

while Error != 0.0:
  net = np.dot(W, Xi[i])

  o = (2 / (1+ np.exp(-net))) - 1

  error += 0.5 * ((d[i] - o) ** 2)
  Error = np.round(error, 1)

  print('Error:', Error)
  dw = c * (d[i] - o) * Xi[i]
  W = W + dw

  print('Updated Weights', i + 1, W)

  iteration += 1
  print('Iteration', i + 1, ':', iteration)

  i += 1
  if i > 2:
    i = 0
    j += 1
    error = 0

  if Error == 0.0:
    break

print(f'Final Weight Matrix:{W}')
print(f'Cycle Epoch Counter:{j}')
print(f'Iteration:{iteration}')
```
#### Iris-Perceptron
```python
import numpy as np
from sklearn.datasets import load_iris

iris=load_iris()
iris.target_names

targets=(iris.target==0).astype(np.int8)
print(targets)

from sklearn.model_selection import train_test_split
datasets=train_test_split(iris.data,targets,test_size=0.2)
train_data, test_data, train_labels, test_labels=datasets

from sklearn.linear_model import Perceptron
p = Perceptron(max_iter=10, eta0=0.001)
p.fit(train_data, train_labels)

import random
sample=random.sample(range(len(test_data)),10)
for i in sample:
  print(i, p.predict(train_data[i].reshape(1, -1)))

from sklearn.metrics import classification_report
print(classification_report(p.predict(train_data), train_labels))
```
#### breast-cancer
```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cancer_data=load_breast_cancer()
x,y = cancer_data.data, cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1000,random_state=42)
mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report=classification_report(y_test,y_pred)
print("Classification Report:\n",class_report)
```
#### MLP
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255
x_train / gray_scale
x_test / gray_scale
print("Feature Matrix: ", x_train.shape)
print("Target Matrix: ", x_test.shape)
print("Feature Matrix: ", y_train.shape)
print("Target Matrix: ", y_test.shape)

fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28), aspect='auto')
        k += 1
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation

layers = [
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
]
model = Sequential(layers)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=2000, validation_split=0.2)
result=model.evaluate(x_test, y_test, verbose=0)
print("Test Loss: ", result[0])
print("Test Accuracy: ", result[1])
```
#### CNN
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images/255.0, test_images / 255.0
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] 

plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show

model=models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  
])
model.summary()

model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history=model.fit(train_images, train_labels)

history=model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")
```
#### RNN
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.6
print(x_train.shape)
print(x_train[0].shape)

model=Sequential()
model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))

model=Sequential()
model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))
model.add(Dropout(0.1))

opt=tf.keras.optimizers.Adam(learning_rate=0.001,decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))

score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
