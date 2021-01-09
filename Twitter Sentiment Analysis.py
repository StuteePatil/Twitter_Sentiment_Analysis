from __future__ import print_function
import import_ipynb
import re
import testsets
import evaluation
import pandas as pd
import numpy as np
import sys
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

MAX_SEQUENCE_LENGTH=100

def preprocessing(filename):
    corpus = []
    # Importing the dataset
    cols = ["id", "label", "text"]
    dataset = pd.read_csv(filename, delimiter = '\t', names=cols, header=None)
    index = list(dataset.iloc[:,0].values)
    labels = list(dataset.iloc[:,1].values)

    for i in range(len(dataset)):
        #removes url from the data
        sub_url = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=])*', ' ', dataset['text'][i])
        #removes #hashtags
        sub_hash = re.sub('^#[A-Za-z0-9]*', '', sub_url)
        #removes @users
        sub_rate = re.sub('^@[A-Za-z0-9]*', '', sub_hash)
        #removes digits
        sub_num = re.sub('(^|\s)[0-9]+\s', ' ', sub_rate)
        #removes special characters
        sub_char = re.sub(r'[^\w|:\)|:\-\)|:\(|:\-\(|:\-O]', ' ', sub_num)
        
        text = re.sub(' +', ' ', sub_char)   #substitutes subsequent spaces with a single space
        text = text.lower()
    
        lemmatizer = WordNetLemmatizer()
        lem = text.split()
        lem_review = []
        for i in lem:
            lem_review.append(lemmatizer.lemmatize(i)) # lemmatizes each word in every document
        review = ' '.join(lem_review) # joins the lemmatized word to create a sequence of words as found in the original document
    
        corpus.append(review) # creates a list of preprocessed documents (tweets)
    
    return index, corpus, labels

# TODO: load training data
train = os.path.join('C:/Users/stute/semeval tweets/', 'twitter-training-data.txt')
val = os.path.join('C:/Users/stute/semeval tweets/', 'twitter-dev-data.txt')

#Tokenize the sentences
tokenizer = Tokenizer()

index, X_train, y_train = preprocessing(train) # preprocess training data
index, X_val, y_val = preprocessing(val) # preprocess validation data

tokenizer.fit_on_texts(X_train)

#converting text into integer sequences
x_tr_seq  = tokenizer.texts_to_sequences(np.array(X_train)) 
x_val_seq = tokenizer.texts_to_sequences(np.array(X_val))

#padding to prepare sequences of same length
x_tr_seq  = pad_sequences(x_tr_seq, maxlen=100)
x_val_seq = pad_sequences(x_val_seq, maxlen=100)

size_of_vocabulary=len(tokenizer.word_index) + 1 #+1 for padding
print(size_of_vocabulary)

# Import label encoder 
from sklearn import preprocessing

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()

X_train = np.array(X_train)
train_labels = np.asarray(y_train)

X_val = np.array(X_val)
val_labels = np.asarray(y_val)

# Encode labels in training data 
y_train= label_encoder.fit_transform(train_labels) 
# Encode labels in vaidation data
y_val = label_encoder.transform(val_labels)


#deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *

model=Sequential()

#embedding layer
model.add(Embedding(size_of_vocabulary,100,input_length=100,trainable=True)) 

#lstm layer
model.add(LSTM(64,return_sequences=True,dropout=0.2))

#Global Maxpooling
model.add(GlobalMaxPooling1D())

model.add(Dense(64, activation='relu'))
model.add(Dense(3,activation='softmax')) 

#Add loss function, metrics, optimizer
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=["acc"]) 

#Adding callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=1)  
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

#Print summary of model
print(model.summary())

history = model.fit(x_tr_seq, y_train, batch_size=100, epochs=10, validation_data=(x_val_seq, y_val), verbose=1, callbacks=[es,mc])


accuracy = []

for test in testsets.testsets:
    # TODO: classify tweets in test set
    testset = os.path.join('C:/Users/stute/semeval tweets', test)      
    index_values, X_test, y_test = preprocessing(testset) # preprocess test data

    predictions = {}
    x_tt_seq  = tokenizer.texts_to_sequences(np.array(X_test)) 
    x_tt_seq  = pad_sequences(x_tt_seq, maxlen=100)

    test_labels = np.asarray(y_test)
    y_test = label_encoder.transform(test_labels)
    y_pred = model.predict(x_tt_seq) # perform classification on test data
    y_pred = np.argmax(y_pred, axis=1)
    
    count=0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            count+=1

    acc = round((count/len(y_pred)), 2)
    accuracy.append(acc)
    
    print(test + ' ---> ' + str(acc))
    
print('Average Accuracy: ', round(sum(accuracy)/len(accuracy), 4))
