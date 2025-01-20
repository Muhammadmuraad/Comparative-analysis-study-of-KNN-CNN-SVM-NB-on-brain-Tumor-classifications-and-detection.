import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical


path='Dataset/'

yes_tumor_images=os.listdir(path+ 'Cancerize/')
no_tumor_images=os.listdir(path+ 'Normal/')
dataset=[]
label=[]

#INPUT_SIZE=100
#print(no_tumor_images)
# path='no0.jpg'
# print(path.split('.')[1])

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(path+'Cancerize/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((100,100))
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(path+'Normal/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((100,100))
        dataset.append(np.array(image))
        label.append(0)

dataset=np.array(dataset)
label=np.array(label)
#print(dataset)
#print(label)


#spliting of dataset 30% test data and 70% train data
x_train, x_test, y_train, y_test=train_test_split(dataset,label, test_size=0.2 ,random_state=0)

# Reshape = (n, image_width, image_height, n_channel)
print(x_train.shape)
print(y_train.shape)

#labels
print(x_test.shape)
print(y_test.shape)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)



# Model Building
# 100,100,3

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train,
batch_size=5,
verbose=1, epochs=32,
validation_data=(x_test, y_test),
shuffle=False)


model.save('BrainTumor10Epochs.h5')





