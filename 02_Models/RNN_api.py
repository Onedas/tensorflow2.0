# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:52:56 2019

@author: Onedas
"""


import tensorflow as tf

def make_model(vocab_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

def model_compile(model):
    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

def fit(model,train_dataset, valid_dataset):
    history = model.fit(train_dataset, epochs=10,
                        validation_data= valid_dataset,
                        validation_steps = 30)
    return history

def evaluate(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset)
    return test_loss, test_acc


if __name__ == "__main__":
    # load dataset : IMDB
    imdb = tf.keras.datasets.imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    word_index = imdb.get_word_index()
    word_index = {w:(i+3) for w,i in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3    
    index2word = {i:w for w,i in word_index.items()}
    
    # padding
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
    # ds
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        
    ##
    model = make_model(len(word_index))
    model_compile(model)
    history = fit(model, train_ds, test_ds)    
    test_loss, test_acc = evaluate(model, test_ds)








    