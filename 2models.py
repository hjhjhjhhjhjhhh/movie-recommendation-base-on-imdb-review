from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

def read_files_to_list_all(path):
    data = []
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file),encoding="utf-8") as f:
            data.append(f.read())
    return data

pos_file_list = read_files_to_list(path_pos)
neg_file_list = read_files_to_list(path_neg)
pos_test_list = read_files_to_list_all(path_pos_test)
neg_test_list = read_files_to_list_all(path_neg_test)
# print(len(pos_file_list + neg_file_list))
import pandas as pd
# connect the positive sentence to the negative sentence and also add the labels(pos or neg).
imdb_df = pd.DataFrame(data = zip(pos_file_list +neg_file_list, ['pos'] * len(pos_file_list) + ['neg'] * len(neg_file_list)))
imdb_df.columns = ['text', 'label']

imdb_test = pd.DataFrame(data = zip(pos_test_list + neg_test_list, ['pos']*len(pos_test_list) + ['neg']*len(neg_test_list)))
imdb_test.columns = ['text', 'label']
print("data load successfully.\n")
# print(imdb_df.head())
# print(imdb_df.tail())
# print(imdb_df['text'])
# initialize the vectorizer, use CountVectorizer to bag-of-word 
vectorizer = CountVectorizer(stop_words="english")
# stop_words="english"

# change the text into bag-of-word
text = vectorizer.fit_transform(imdb_df['text'])
X_train, X_test, y_train, y_test = train_test_split(text, imdb_df['label'], test_size=0.2, random_state=0)

# string = ("1 Naive Bayes Classifier\n" + 
# "2 Decision Tree Classifier")
# print(string)
# train_model = input("please enter the number to select the training model: ")

# if train_model == 1:
#     print("The training model is Naive Bayes Classifier\n")
#     model = MultinomialNB()
# elif train_model == 2:
#     print("The training model is Decision Tree Classifier\n")
#     model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model1 = MultinomialNB()
model2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
import time
start_time = time.time()

for i in range(2):
    if i == 0:
        print("The model is Naive Bayes Classifier")    
        model1.fit(X_train, y_train)
    elif i == 1:
        print("The training model is Decision Tree Classifier")
        model2.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    print("it takes %s seconds ---" % (time.time() - start_time))
    if i == 0:
        y_pred_train = model1.predict(X_train)
        y_pred_test = model1.predict(X_test)
    elif i == 1:
        y_pred_train = model2.predict(X_train)
        y_pred_test = model2.predict(X_test)
    # y_pred_train = model.predict(X_train)
    print("accuracy of train data is ", accuracy_score(y_train, y_pred_train))
    print("accuracy of validation data is ", accuracy_score(y_test, y_pred_test))
    # print("r2 score of train data is :", model.score(X_train, y_train))
    # print("r2 score of test data is : ", model.score(X_test, y_test))

    answers = []
    predicts = []
    test_data_len = len(pos_test_list+neg_test_list)
    if i == 0:
        for i in range(test_data_len) :
            if(i < len(pos_test_list)):
                answers.append('pos')
                test_text = vectorizer.transform([pos_test_list[i]])
                predicts.append(model1.predict(test_text))
            else:
                answers.append('neg')
                test_text = vectorizer.transform([neg_test_list[i-len(pos_test_list)]])
                predicts.append(model1.predict(test_text))
    elif i == 1:
        for i in range(test_data_len) :
            if(i < len(pos_test_list)):
                answers.append('pos')
                test_text = vectorizer.transform([pos_test_list[i]])
                predicts.append(model2.predict(test_text))
            else:
                answers.append('neg')
                test_text = vectorizer.transform([neg_test_list[i-len(pos_test_list)]])
                predicts.append(model2.predict(test_text))
    # print(accuracy_score(answers, predicts))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(answers, predicts)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[1][0]
    FN = cm[0][1]
    Accuracy = (TP + TN) / (TP + TN + FP + FN) 
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    print(f"accuracy of {test_data_len} test data is {Accuracy}")
    print(f"precision of {test_data_len} test data is {Precision}")
    print(f"recall of {test_data_len} test data is {Recall}")
    print(f"f1_score of {test_data_len} test data is {F1_score}\n")
    # print("confusion matrix is ", cm)
