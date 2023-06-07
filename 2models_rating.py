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
        # print("error loading")
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


path_pos = "aclImdb/train/pos"
path_neg = "aclImdb/train/neg"
path_pos_test = "aclImdb/test/pos"
path_neg_test = "aclImdb/test/neg"

import random
def generate_random_integers(start, end, count):
    random_integers = set()
    while len(random_integers) < count:
        random_int = random.randint(start, end)
        random_integers.add(random_int)
    
    return sorted(list(random_integers))
rand_list = generate_random_integers(0,12499,2500)

# choose 2500 pos reviews and 2500 neg reviews
import os
def read_files_to_list(path):
    data = []
    files = os.listdir(path)
    k = 0
    cnt = 0
    for file in files:  
        if k in rand_list:
            cnt += 1
            with open(os.path.join(path, file),encoding="utf-8") as f:
                data.append(f.read())
        k += 1
        if (cnt == 2500) : 
            break
    return data

pos_file_list = read_files_to_list(path_pos)
neg_file_list = read_files_to_list(path_neg)
pos_test_list = read_files_to_list(path_pos_test)
neg_test_list = read_files_to_list(path_neg_test)
import pandas as pd
# connect the positive sentence to the negative sentence and also add the labels(pos or neg).
imdb_df = pd.DataFrame(data = zip(pos_file_list +neg_file_list, ['pos'] * len(pos_file_list) + ['neg'] * len(neg_file_list)))
imdb_df.columns = ['text', 'label']

imdb_test = pd.DataFrame(data = zip(pos_test_list + neg_test_list, ['pos']*len(pos_test_list) + ['neg']*len(neg_test_list)))
imdb_test.columns = ['text', 'label']
print("data load successfully.\n")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# initialize the vectorizer, use CountVectorizer to implement it.
vectorizer = CountVectorizer(stop_words="english")
# stop_words="english"

# change the text into bag-of-word
text = vectorizer.fit_transform(imdb_df['text'])
X_train, X_test, y_train, y_test = train_test_split(text, imdb_df['label'], test_size=0.2, random_state=0)
string = ("1 Naive Bayes Classifier\n" + 
"2 Decision Tree Classifier")
print(string)
train_model = input("please enter the number to select the training model: ")

if(train_model == "1"):
    print("your training model is Naive Bayes Classifier\n")
    model = MultinomialNB()
elif(train_model == "2"):
    print("your training model is Decision Tree Classifier\n")
    model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

model.fit(X_train, y_train)

labels = []
for i in range(len(lis)) :
    text = vectorizer.transform([lis[i]])
    labels.append(model.predict(text)[0])

total = 0
positive = 0
for i in range(len(lis)):
    total += 1
    if(labels[i] == 'pos'):
        positive += 1

print(f'In total {total} comments, {positive} comments are positive, {100 * positive / total}% people think it is a good movie')
print(f'averge rate of {total} comments is {sum(rating) / len(rating)}')


# 等待5秒
time.sleep(5)
# 關閉瀏覽器
driver.quit()    