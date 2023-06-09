# movie-recommendation-base-on-imdb-review

## Overview
This program porvide a easy way to train your own models of movie recommendation
base on imdb review in **bert** and **LSTM**. This also provide other two models to
compare the result of training. You can just use the training result to get movie rating as well.
If you want to get more information, you can check our [slide](https://docs.google.com/presentation/d/1B8eHZLpXcCA7BpNuuQ01biKqHsLOxQbO8Kihi4dgt2Y/edit#slide=id.g226f3c6c0b9_0_8) to get more information and how it works.

---

## Hints
There are some hints before if you want to use it.

1. Before you run the code, you need to run the following commend
    **pip install -r requirements.txt** 
to install the resource you need.

2. WE also trained **bert** model yet is too big. So you need to download it
on the google drive that we provide down this.
https://drive.google.com/drive/folders/1bTRAzj0lBc_Bhq7MnYWoukcDC3ebJoDn?usp=drive_link
And put it into the **train** folder if you don't want to train your own model by yourself.

---

## How to use

- If you are the normal user only want to see the result of rating.

you can just see the [main.py](https://github.com/hjhjhjhhjhjhhh/movie-recommendation-base-on-imdb-review/blob/master/main.py) file
in the line 162-169, you can put your own **url** in line 162 and run the **main.py**.
```python
url = input("please the link of a movie review: ")
```
You will get the rating result in your console.
Note that you must to check that the url has already into the user reviews page in IMDb.
This make sure the program can work successfully.


- If you are the user that want to train your own model.

You can check out[LSTM.py](https://github.com/hjhjhjhhjhjhhh/movie-recommendation-base-on-imdb-review/blob/master/train/LSTM.py)and[bert_train.py](https://github.com/hjhjhjhhjhjhhh/movie-recommendation-base-on-imdb-review/blob/master/train/bert_train.py).
Also there is an example[tempCodeRunnerFile.py](https://github.com/hjhjhjhhjhjhhh/movie-recommendation-base-on-imdb-review/blob/master/tempCodeRunnerFile.py)for you that you can set up your own parameters.

Like in the line 82-84 in **bert_train.py**
```python
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LEN = 128
```
You can change your own MAX_SEQ_LEN, EPOCHS and BATCH_SIZE.
And in the line 107-108
```python
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```
You can change the optermizer, loss and metrics you want.
You can get more model building details in the place maintioned above.
You will have trained your data, you need to put the result into the **train** folder.

---

## Reference

[2019, NLP 入門 (1) — Text Classification (Sentiment Analysis) — 極簡易情感分類器 Bag of words + Naive Bayes](https://sfhsu29.medium.com/nlp-%E5%85%A5%E9%96%80-1-text-classification-sentiment-analysis-%E6%A5%B5%E7%B0%A1%E6%98%93%E6%83%85%E6%84%9F%E5%88%86%E9%A1%9E%E5%99%A8-bag-of-words-naive-bayes-e40d61de9a7f)
[Day 16：『長短期記憶網路』(LSTM) 應用 -- 情緒分析(Sentiment Analysis)](https://ithelp.ithome.com.tw/articles/10193924)
[IMDB Movie Review Sentiment Analysis Using BERT Fine-Tuning With Tensorflow Keras](https://haren.medium.com/imdb-movie-review-sentiment-analysis-using-bert-fine-tuning-with-tensorflow-keras-1473489af306)
