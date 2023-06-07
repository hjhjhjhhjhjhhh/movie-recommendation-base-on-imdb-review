from keras.layers.core import Activation, Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
# from keras.preprocessing import sequence
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import re

# data_cleaning is used to clear the html tags and move the emojis to the tail of the text
# This function is also used for change the text into lower case
def data_cleaning(text):
    text=re.sub('<[^>]*>','',text)
    emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower()) + ' '.join(emojis).replace('-','')
    # text = re.sub('[^a-zA-Z\s]', " ", text)
    return text

def clear_puntuation(tokens):
    import string
    filtered_tokens = [token for token in tokens if token not in string.punctuation]
    return filtered_tokens

import nltk
from nltk.corpus import stopwords
stopword_list = stopwords.words("english")
stopword_list.remove("no")
stopword_list.remove("not")

def preprocess(text):
    text = data_cleaning(text)
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stopword_list]
    clear_puntuation(words)
    return words

import random
def generate_random_integers(start, end, count):
    random_integers = set()
    while len(random_integers) < count:
        random_int = random.randint(start, end)
        random_integers.add(random_int)
    
    return sorted(list(random_integers))

# random choose 2500 pos reviews and 2500 neg reviews
list_pos = generate_random_integers(0, 12499, 2500)
list_neg = generate_random_integers(12500, 24999, 2500)

max_len = 0
word_freqs = collections.Counter()
num_recs = 0
cnt_pos = 0
cnt_neg = 0
choose_pos = 0
choose_neg = 12500
with open('train_ds.txt','r', encoding='UTF-8') as f:
    
    for line in f.readlines():
        if choose_pos < 12500:
            if choose_pos in list_pos:
                cnt_pos += 1
                num_recs += 1
                label, sentence = line[0], line[2:-1]
                words = preprocess(sentence)
                if len(words) > max_len:
                    max_len = len(words)
                for word in words:
                    word_freqs[word] += 1
            choose_pos += 1
        if choose_pos >=12500 and choose_neg < 25000:
            if choose_neg in list_neg:
                cnt_neg += 1
                num_recs += 1
                label, sentence = line[0], line[2:-1]
                words = preprocess(sentence)
                if len(words) > max_len:
                    max_len = len(words)
                for word in words:
                    word_freqs[word] += 1
            choose_neg += 1
        if(cnt_neg == 2500):
            break
print('The maximum length of all sentence is ',max_len)
print('The total number of words is ', len(word_freqs))
# print("cnt_neg", cnt_neg)
# print("cnt_pos", cnt_pos)


# # max_feature corresponds to word_freqs / max_sentence_length corresponds to max_len
Max_Features = 20000
Max_Sentence_Length = 128
vocab_size = min(Max_Features, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(Max_Features))}
word2index["PAD"] = 0
word2index["UNK"] = 1
# print(word2index)

# transform the word_index into the txt format
with open('word2index.txt', 'w', encoding="utf-8") as file:
    for word, num in word2index.items():
        file.write(word + "\t" + str(num) + "\n")

X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
cnt_pos = 0
cnt_neg = 0
choose_pos = 0
choose_neg = 12500
k=0
# use the dictionary to save the words
with open("train_ds.txt", 'r', encoding = 'utf-8') as file:
    for line in file.readlines():
        if choose_pos < 12500:
            if choose_pos in list_pos:
                cnt_pos += 1
                label, sentence = line[0], line[2:-1]
                words = preprocess(sentence)
                array = []
                for word in words:
                    if word in word2index:
                        array.append(word2index[word])
                    else:
                        array.append(word2index["UNK"])
                X[k] = array
                y[k] = int(label)
                k += 1
            choose_pos += 1
        if choose_pos >=12500 and choose_neg < 25000:
            if choose_neg in list_neg:
                cnt_neg += 1
                label, sentence = line[0], line[2:-1]
                words = preprocess(sentence)
                array = []
                for word in words:
                    if word in word2index:
                        array.append(word2index[word])
                    else:
                        array.append(word2index["UNK"])
                X[k] = array
                y[k] = int(label)
                k += 1
            choose_neg += 1
        if(cnt_neg == 2500):
            break
print("k is ",k )
        
    
X = pad_sequences(X, maxlen=Max_Sentence_Length)
# data split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# construct the model
Embedding_Size = 128
Hidden_Layer_Size = 64
Batch_Size = 32
Epoch_Num = 3
model = Sequential()
# add embedding layer
model.add(Embedding(vocab_size, Embedding_Size,input_length=Max_Sentence_Length))
# add lstm layer
model.add(LSTM(Hidden_Layer_Size, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
model.summary()
# train the model
history = model.fit(Xtrain, ytrain, batch_size=Batch_Size, epochs=Epoch_Num,validation_data=(Xtest, ytest))

score, acc = model.evaluate(Xtest, ytest, batch_size=Batch_Size)
print("\nValidation loss: %.3f, accuracy: %.3f\n" % (score, acc))
    
# save the model
model.save('Sentiment.h5')  # creates a HDF5 file 'model.h5'

num_test = 25000
X_test = np.empty(num_test, dtype=list)
y_test = np.zeros(num_test)
k=0
# use the dictionary to save the words
with open("test_ds.txt", 'r', encoding = 'utf-8') as file:
    for line in file.readlines():
        label, sentence = line[0], line[2:-1]
        # words = nltk.word_tokenize(sentence.lower())
        sentence = data_cleaning(sentence)
        words = nltk.word_tokenize(sentence.lower())
        words = [word for word in words if word not in stopword_list]
        clear_puntuation(words)
        array = []
        for word in words:
            if word in word2index:
                array.append(word2index[word])
            else:
                array.append(word2index["UNK"])
        X_test[k] = array
        y_test[k] = int(label)
        k += 1
        
X_test = pad_sequences(X_test, maxlen=Max_Sentence_Length)

score, acc = model.evaluate(X_test, y_test)
print("\nTest loss: %.3f, accuracy: %.3f" % (score, acc))
predicts = [int(round(x[0])) for x in model.predict(X_test) ]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicts)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
Accuracy = (TP + TN) / (TP + TN + FP + FN) 
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_score = 2 * (Precision * Recall) / (Precision + Recall)
print(f"accuracy of {k} test data is {Accuracy}")
print(f"precision of {k} test data is {Precision}")
print(f"recall of {k} test data is {Recall}")
print(f"f1_score of {k} test data is {F1_score}\n")

import matplotlib.pyplot as plt
# plot loss between training and validation data
loss_train = history.history['loss']
loss_val = history.history['val_loss']
acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1,len(acc_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy between training and validation data
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()