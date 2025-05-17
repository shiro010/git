# In[1]: 
import pandas as pd
import numpy as np
import os
import glob
import pathlib
import re
import janome
import jaconv

p_temp = pathlib.Path('text')

article_list = []

# フォルダ内のテキストファイルを全てサーチ
for p in p_temp.glob('**/*.txt'):
    #第二階層フォルダ名がニュースサイトの名前になっているので、それを取得
    media = p.parent.name
    file_name = p.name

    if file_name != 'LICENSE.txt' and file_name != 'CHANGES.txt' and file_name != 'README.txt':
        #テキストファイルを読み込む
        with open(p, 'r') as f:
            #テキストファイルの中身を一行ずつ読み込み、リスト形式で格納
            article = f.readlines()
            #不要な改行等を置換処理
            article = [re.sub(r'[\n \u3000]', '', i) for i in article]
    #ニュースサイト名・記事URL・日付・記事タイトル・本文の並びでリスト化
            article_list.append([media, article[0], article[1], article[2], ''.join(article[3:])])
    else :
        continue


article_df = pd.DataFrame(article_list)

from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *

t = Tokenizer()
char_filters = [UnicodeNormalizeCharFilter()]
analyzer = Analyzer(char_filters=char_filters, tokenizer=t)

word_lists = []
for i, row in article_df.iterrows(): 
    # print(i) 
    # print(row[3])  
    # print(row[4])
    row_lists = []
    for t in analyzer.analyze(row[4]):
        #形態素
        surf = t.surface
        #基本形
        # base = t.base_form
        #品詞
        # pos = t.part_of_speech
        #読み
        # reading = t.reading
        
        row_lists.append(t)
    word_lists.append(row_lists)


# In[4]:


article_list = []
for i in range(len(word_lists)):
    surface_words = [token.surface for token in word_lists[i]]
    joined = ' '.join(surface_words)
    article_list.append(joined)

df = pd.DataFrame(article_list, columns=["article"])


# In[5]:


df.head()


# In[6]:


df.to_csv('token.csv', encoding='utf-8')

