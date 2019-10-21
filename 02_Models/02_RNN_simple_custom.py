# -*- coding: utf-8 -*-
'''
tensorflow version : 2.0.0

@author: Onedas
'''
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras import Model

import time

#define my model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = Embedding(88588 ,64)
        self.bi_LSTM = Bidirectional(LSTM(64))
        self.d1 = Dense(64, activation = 'relu')
        self.d2 = Dense(1, activation = 'sigmoid')
        
    def call(self, x):
        x= self.embedding(x)
        x= self.bi_LSTM(x)
        x= self.d1(x)
        x= self.d2(x)
        return x
        
        
model = MyModel()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.0001)

@tf.function
def train_step(texts, labels):
    with tf.GradientTape() as tape:
        predicts = model(texts)
        loss = loss_object(labels, predicts)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        for images, labels in dataset:
            train_step(images,labels)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


if __name__ == "__main__":
    # load data
    imdb = tf.keras.datasets.imdb
    (x_train, y_train),(x_test,y_test) = imdb.load_data()
    word_index = imdb.get_word_index()

    word_index = {w:(i+3) for w,i in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)    
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    
    # model
    model = MyModel()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.0001)
    
    #%% train
    train(train_ds, 1)
    
    #%%
    