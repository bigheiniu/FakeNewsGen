import os
import urllib
from sklearn.\
    model_selection import train_test_split
import multiprocessing as mp
import requests
import time
import pickle
from tqdm import tqdm
import json
from itertools import chain
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))

# the openie program will return the index of each tuple;
# and there will multiple tuples for single sentence
LATIN_1_CHARS = (
    ('\xe2\x80\x99', "'"),
    ('\xc3\xa9', 'e'),
    ('\xe2\x80\x90', '-'),
    ('\xe2\x80\x91', '-'),
    ('\xe2\x80\x92', '-'),
    ('\xe2\x80\x93', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x98', "'"),
    ('\xe2\x80\x9b', "'"),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9d', '"'),
    ('\xe2\x80\x9e', '"'),
    ('\xe2\x80\x9f', '"'),
    ('\xe2\x80\xa6', '...'),
    ('\xe2\x80\xb2', "'"),
    ('\xe2\x80\xb3', "'"),
    ('\xe2\x80\xb4', "'"),
    ('\xe2\x80\xb5', "'"),
    ('\xe2\x80\xb6', "'"),
    ('\xe2\x80\xb7', "'"),
    ('\xe2\x81\xba', "+"),
    ('\xe2\x81\xbb', "-"),
    ('\xe2\x81\xbc', "="),
    ('\xe2\x81\xbd', "("),
    ('\xe2\x81\xbe', ")")
)

def load_news(data_path, save_path):
    type = data_path.split("/")[-1]
    real_news = []
    fake_news = []
    for direc in os.listdir(data_path):
        direc = os.path.join(data_path, direc)
        if "_fake" in direc:
            for file in tqdm(os.listdir(direc)):
                with open(os.path.join(direc, file, "news_article.json"), "r", ) as f1:
                    data = json.load(f1)
                    try:
                        news_fake = data["text"].replace("\n", " ")
                        headline = data["title"].replace("\n", " ")
                        if len(news_fake.split()) < 50:
                            continue
                        for _hex, _char in LATIN_1_CHARS:
                            news_fake = news_fake.replace(_hex, _char)
                            headline = headline.replace(_hex, _char)
                    except:
                        print(os.path.join(direc, file, "news_article.json"))
                        continue
                    if len(news_fake) < 1 or len(headline) < 1:
                        continue
                fake_news.append((headline, news_fake, 0))

        elif "_real" in direc:
            for file in tqdm(os.listdir(direc)):
                with open(os.path.join(direc, file, "news_article.json"), "r") as f1:
                    data = json.load(f1)
                    try:
                        news_real = data["text"].replace("\n", " ")
                        headline = data["title"].replace("\n", " ")
                        if len(news_real.split()) < 50:
                            continue
                        for _hex, _char in LATIN_1_CHARS:
                            news_fake = news_fake.replace(_hex, _char)
                            headline = headline.replace(_hex, _char)
                    except:
                        print(os.path.join(direc, file, "news_article.json"))
                        continue
                    if len(news_real) < 1 or len(headline) < 1:
                        continue
                real_news.append((headline, news_real, 1))

    news_df = pd.DataFrame(real_news + fake_news, columns=["headline" ,'news', 'label'])
    # news_df['news'] = news_df['news'].str.encode('utf-8')
    # news_df['headline'] = news_df['headline'].str.encode('utf-8')
    news_df.to_csv("{}/news_label_{}.csv".format(save_path, type), index=None)
    return real_news + fake_news


if __name__ == '__main__':
    data_path = "/home/yichuan/fake_news_data/gossip"
    save_news_path = "./data/news_corpus"
    load_news(data_path, save_news_path)


