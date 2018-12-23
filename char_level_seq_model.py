# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 01:09:14 2018

@author: jaydeep thik
"""

import keras
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import random
import sys

#using temperature to control the entropy/ stocasticity in the prediction sample new character
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

path = keras.utils.get_file('nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
#print(len(text))

#preparing data to feed ino LSTMs
max_len = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text)-max_len, step):
    sentences.append(text[i:i+max_len])
    next_chars.append(text[i+max_len])
    
unique_cahrs = sorted(list(set(text)))
char_indices = dict((char, unique_cahrs.index(char)) for char in unique_cahrs)

#one_hot
X = np.zeros((len(sentences), max_len, len(unique_cahrs)), dtype=np.bool)
y = np.zeros((len(sentences), len(unique_cahrs)), dtype = np.bool)

for i, sentence in enumerate(sentences):
    for t , char in enumerate(sentence):
        X[i, t, char_indices[char]]=1
        y[i, char_indices[next_chars[i]]]=1
        
        
#learning model

model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(max_len, len(unique_cahrs))))
model.add(layers.Dense(len(unique_cahrs), activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(0.01), loss='categorical_crossentropy')

#train and generate outputs
tot_epoches=60
temperature = 0.2
for epoch in range(tot_epoches):
    print("\nepoch :", epoch)
    model.fit(X,y, batch_size=128, epochs=1)
    start_index = random.randint(0,len(text)-max_len-1)
    generated_text = text[start_index: start_index+max_len]
    
    print('-----seed-----"'+generated_text+'"\n')
    sys.stdout.write(generated_text)
    
    for i in range(400):
        sampled = np.zeros(1, max_len, len(unique_cahrs))
        for t, char in enumerate(generated_text):
            sampled[0,t,char_indices[char] ]=1
            
        preds = model.predict(sampled, verbose=0)[0]
        next_id = sample(preds, temperature)
        next_char = unique_cahrs[next_id]
        generated_text+=next_char
        generated_text=generated_text[1:]
        sys.stdout.write(next_char)