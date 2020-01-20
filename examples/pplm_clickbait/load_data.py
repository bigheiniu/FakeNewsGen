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
    # ATTENTION: we only keep the previous 250 words
    type = data_path.split("/")[-1]
    news_list = []
    max_news_length = 250
    for direc in os.listdir(data_path):
        direc = os.path.join(data_path, direc)
        if "_fake" in direc or "_real" in direc:
            for file in tqdm(os.listdir(direc)):
                with open(os.path.join(direc, file, "news_article.json"), "r", ) as f1:
                    data = json.load(f1)
                    if len(data) == 0:
                        continue
                    news_content = data["text"].encode('ascii', 'ignore').decode('utf-8').replace("\n", " ")
                    headline = data["title"].encode('ascii', 'ignore').decode('utf-8').replace("\n", " ")
                    for _hex, _char in LATIN_1_CHARS:
                        news_content = news_content.replace(_hex, _char)
                        headline = headline.replace(_hex, _char)
                    if len(news_content.split()) < 50:
                        continue
                    news_content = " ".join(news_content.split()[:max_news_length])
                    if len(news_content) < 1 or len(headline) < 1:
                        continue

                # ATTENTION: 0 is fake, 1 is real news
                label = 0 if "_fake" in direc else 1
                news_list.append((headline, news_content, label))

    news_df = pd.DataFrame(news_list, columns=["headline" ,'news', 'label'])
    # news_df['news'] = news_df['news'].str.encode('utf-8')
    # news_df['headline'] = news_df['headline'].str.encode('utf-8')
    news_df.to_csv("{}/news_label_{}.csv".format(save_path, type), index=None)
    return news_list


if __name__ == '__main__':
    data_path = "/home/yichuan/fake_news_data/political_fact"
    save_news_path = "./data/news_corpus"
    load_news(data_path, save_news_path)


