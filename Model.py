#!/usr/bin/env python
# coding: utf-8

# ### Make CNN from Keras
# 

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPool2D


# In[3]:


from keras.layers import Flatten  #to flatten the image and input to neural net further


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.optimizers import Adam ,RMSprop ,SGD ,Nadam ,Adamax


# In[6]:


from keras.models import Sequential
model = Sequential()


# ###  Adding layers to the sequential model
# 

# In[7]:


import random


# In[8]:


model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu',
                        input_shape=(100,100,3)
                       ))
model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),border_mode='same'))


# In[9]:


def architecture(option):
    if option == 1:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
    elif option == 2:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
    elif option == 3:
        #two convolutional and 2 max pooling layers
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    elif option == 4:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    
    else:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        


# In[10]:


architecture(random.randint(1,4))


# In[11]:


model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu',
                        border_mode='same'
                       ))
model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),border_mode='same'))


# In[12]:


model.add(Flatten())


# In[13]:


def fullyconnected(option):
    if option == 1:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 2:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 3:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 4:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        
    else:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    
          


# In[14]:


fullyconnected(random.randint(1,5))


# In[15]:


model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))


# In[16]:


model.add(Dense(units=1,activation='sigmoid'))


# In[17]:


print(model.summary())


# ###  Compile the Model

# In[18]:


model.compile(optimizer=random.choice((RMSprop(lr=.001),Adam(lr=.0001),SGD(lr=.001),Nadam(lr=.001),Adamax(lr=.001))),loss='binary_crossentropy',metrics=['accuracy'])


# ###  Train Model

# In[19]:


from keras.preprocessing.image import ImageDataGenerator


# In[23]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'covid_images/train/',
        target_size=(100,100),
        batch_size=40,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'covid_images/test/',
        target_size=(100,100),
        batch_size=40,
        class_mode='binary')


# In[29]:


out = model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=28)


# In[63]:


out.history


# In[64]:


print(out.history['val_accuracy'][0])


# In[62]:


mod =str(model.layers)
accuracy = str(out.history['val_accuracy'][0])


# In[61]:


if out.history['val_accuracy'][0] >= .80:
    import smtplib
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security 
    s.starttls()

    # Authentication 
    s.login("user@gmail.com", "password")


    # message to be sent 
    message1 = accuracy
    message2 = mod


    # sending the mail 
    s.sendmail("user@gmail.com", "receiver@gmail.com", message1)
    s.sendmail("user@gmail.com", "receiver@gmail.com", message2)

    # terminating the session 
    s.quit()

