#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers,models,Sequential


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense,Dropout

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# In[4]:





train_datagen = ImageDataGenerator(
                rescale=1. / 255,#Normalization
                shear_range = 0.2,
                zoom_range =0.2,
                horizontal_flip = True) #augumentation

training_set = train_datagen.flow_from_directory('food/food/train',
                                        target_size = (280,280),
                                        batch_size = 6819,
                                        class_mode = 'categorical')
                                

X_train, y_train = training_set.next()


# In[5]:


print(X_train.shape)


# In[6]:


print(y_train.shape)


# In[7]:


y_train.shape


# In[8]:





test_datagen = ImageDataGenerator(
                rescale=1. / 255) #Data Normalization

test_set = test_datagen.flow_from_directory('food/food/test',
                                        target_size = (280,280),
                                        batch_size = 1527,
                                        class_mode = 'categorical')
                                

X_test, y_test = test_set.next()


# In[9]:


print(X_test.shape)


# In[10]:


print(y_test.shape)


# In[11]:


y_test.shape


# In[12]:


from tensorflow.keras.applications.vgg16 import VGG16

#Intialization of vgg16 architecture
base_model = VGG16(input_shape = (280, 280, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')


# In[13]:


for layer in base_model.layers:
    layer.trainable = False


# In[14]:


# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)


x = layers.Dropout(0.5)(x)

x = layers.Dense(20, activation='sigmoid')(x)

#base model - VGG-16 (all the convolution and pooling) + x (densely connected layers)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[15]:


model.summary()


# In[16]:


vgghist = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))


# In[17]:


model.evaluate(X_test,y_test)


# In[35]:


plt.matshow(X_test[1000])


# In[36]:


y_pred = model.predict(X_test)
y_pred[1000]


# In[37]:


for i in range(20):
    print(y_pred[1000][i])


# In[38]:


result = np.argmax(y_pred[20])
print(result)


# In[43]:


classes = ['apple_pie','cheesecake','chocolate_cake','donuts','french_fries','fried_rice','garlic_bread','hamburger',
 'pancakes','ice_cream','lasagna','nachos','omelette','hot_dog','pizza','samosa','sushi','tacos','tiramisu','waffles']


# In[44]:


print(classes[result])


# In[41]:


model_json = model.to_json()
with open("model-food2.json","w") as json_file:
    json_file.write(model_json)
print('Model Saved')
model.save_weights('model-food2.h5')
print('Weights Saved')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




