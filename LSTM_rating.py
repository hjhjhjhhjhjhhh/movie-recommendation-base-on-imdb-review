from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import re
import os
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
from glob import glob


options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_experimental_option("prefs", {"profile.password_manager_enabled": False, "credentials_enable_service": False})
options.chrome_executable_path = 'chromedriver.exe'

url = input("please the link of a movie review: ")

# from webdriver_manager.chrome import ChromeDriverManager
# driver = webdriver.Chrome(ChromeDriverManager().install())
driver=webdriver.Chrome(options=options)
driver.get(url)
page = 1
# IMDB中每個頁面只有25則評論，因此我們必須翻10頁來取得200筆以上的資訊
while page < 10:
    try:
        # 用css_selector找尋'load-more-trigger'的位置
        css_selector = 'load-more-trigger'
        driver.find_element(By.ID, css_selector).click()
        time.sleep(3)
        page += 1
    except:
        break
# 尋找class = review-container的標籤
review = driver.find_elements(By.CLASS_NAME, 'review-container')
# 儲存星星數與評價的list
rating = []
lis = []
cnt = 0
# 設定最多找200筆資訊
for n in range(0,250):
    try:
        if cnt >=200:
            break
        # 用戶評論必須同時具備rating和title的資料，否則略過並尋找下一筆
        frating = review[n].find_element(By.CLASS_NAME, 'rating-other-user-rating').text
        flist = review[n].find_element(By.CLASS_NAME, 'title').text

        rating.append(frating)
        lis.append(flist)
        cnt += 1
    except:
        continue
# 將rating的資料從string轉成int
for j in range(len(rating)):
    rating[j] = rating[j].replace('/10', "")
    rating[j] = int(rating[j])


query = []
with open('word2index.txt', 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        word, number = line.split("\t")
        query.append((word, int(number)))

word_index = dict(query)

from keras.utils import pad_sequences
import nltk
import numpy as np
from keras.models import load_model
import re

def data_cleaning(text):
    text=re.sub('<[^>]*>','',text)
    emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower()) + ' '.join(emojis).replace('-','')
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

MAX_FEATURES = 20000
MAX_SENTENCE_LENGTH = 128
# load the LSTM model.
model = load_model('train/lstm_after_trained/Sentiment.h5')

X = np.empty(len(lis),dtype=list)

i=0
# use the word_index to store the sentence
for sentence in lis:
    sentence = data_cleaning(sentence)
    words = nltk.word_tokenize(sentence.lower())
    words = [word for word in words if word not in stopword_list]
    clear_puntuation(words)
    array = []
    for word in words:
        if word in word_index:
            array.append(word_index[word])
        else:
            array.append(word_index["UNK"])
    X[i] = array
    i += 1
# if the sentence_length is not enough, then padding it.        
X = pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

labels = [int(round(x[0])) for x in model.predict(X)]

total = 0
positive = 0
for i in range(len(lis)):
    total += 1
    if(labels[i] == 1):
        positive += 1

print(f'In total {total} comments, {positive} comments are positive, {100 * positive / total}% people think it is a good movie')
print(f'averge rate of {total} comments is {sum(rating) / len(rating)}')
            
# 等待5秒
time.sleep(5)
# 關閉瀏覽器
driver.quit()    
