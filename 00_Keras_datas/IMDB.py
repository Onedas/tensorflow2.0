# -*- coding: utf-8 -*-
'''
tensorflow version : 2.0.0

@author: Onedas
'''

import tensorflow as tf

imdb = tf.keras.datasets.imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data()
# good review = 0, bad review = 1

word_index = imdb.get_word_index()
word_index = {w:(i+3) for w,i in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

index2word = {i:w for w,i in word_index.items()}
def decode_text(text):
    return ' '.join([index2word[i] for i in text])
#%% padding with tf.keras API
    
train_data = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
