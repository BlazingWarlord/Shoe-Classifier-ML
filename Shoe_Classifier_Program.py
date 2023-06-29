import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

import pickle

import numpy as np

filepath = 'train\\adidas\\108.jpg'

try:
    with open('model.bin','rb') as mdl:
        model = pickle.load(mdl)
        print("\n\nModel Loaded\n\n")
        
except:
    train_path = 'train'
    valid_path = 'test'


    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_batches = train_datagen.flow_from_directory(train_path, target_size=(224,224), classes=['adidas','converse','nike'], batch_size=32)
    valid_batches = valid_datagen.flow_from_directory(valid_path, target_size=(224,224), classes=['adidas','converse','nike'], batch_size=32)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(train_batches, validation_data=valid_batches, epochs=15)

    with open('model.bin','wb') as mdl:
        pickle.dump(model,mdl)


img = load_img(filepath,target_size = (224,224))
x = img_to_array(img)
x = x/255.0
x = np.expand_dims(x,axis=0)

prediction = model.predict(x)

print(prediction)


if(max(prediction[0]) == prediction[0][0]):
    print("Adidas")
elif(max(prediction[0]) == prediction[0][1]):
    print("Converse")
else:
    print("Nike")


