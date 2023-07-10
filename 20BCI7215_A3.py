#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.vgg16 import preprocess_input
import random


# In[2]:


get_ipython().system('pip install kaggle')


# In[3]:


get_ipython().system(' mkdir ~/.kaggle')


# In[4]:


get_ipython().system(' cp kaggle.json ~/.kaggle/')


# In[5]:


get_ipython().system(' chmod 600 ~/.kaggle/kaggle.json')


# In[6]:


get_ipython().system('kaggle datasets download -d akash2907/bird-species-classification')


# In[7]:


get_ipython().system('unzip bird-species-classification.zip')


# In[8]:


train_ds="/content/test_data/test_data"
test_ds="/content/train_data/train_data"


# In[9]:


len(os.listdir(train_ds))


# In[32]:


def label_images2(DIR, dataset):
    label = []
    image = []
    j=0
    for i in range (0,30):
        j = random.randint(0, len(dataset.filenames))
        label.append(dataset.filenames[j].split('/')[0])
        image.append(DIR + '/' + dataset.filenames[j])
    return [label,image]

#plot the random images.
y,x = label_images2(test_ds, test_generator)

for i in range(6):
    X = load_img(x[i])
    plt.subplot(2,3,+1 + i)
    plt.axis(False)
    plt.title(y[i], fontsize=8)
    plt.imshow(X)
plt.show()


# In[19]:


train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        train_ds,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical')


val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_directory(
        test_ds,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical')


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
        test_ds,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical')


# In[22]:


base_model=tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3))


# In[23]:


base_model.trainable = False


# In[25]:


from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(2048,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.35))
model.add(Dense(2048,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.35))
model.add(Dense(16,activation='softmax',kernel_initializer='glorot_normal'))


# In[26]:




model.summary()

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=40,validation_data=val_generator)


# In[ ]:



scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

