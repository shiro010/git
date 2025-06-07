import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"現在使用中のデバイス: {device}")

df = pd.read_csv('dataset/token.csv')
sentences = []
for text in df['article']:
    text_list = text.split(' ')
    sentences.append(text_list)

from gensim.models import Word2Vec
model = Word2Vec(sentences,  sg=1, vector_size=100, window=5, min_count=1)

for word, similarity in model.wv.most_similar('家族'):
    print(f"{word}: {similarity}")

print("\n")

results = model.wv.most_similar(positive=['人生'], negative=['幸福'])
for word, similarity in results:
    print(f"{word}: {similarity:.4f}")