import os
import sys
sys.path.append('../../04_word_embedding')
import pdb
import pandas as pd
import numpy as np
from english_preprocessing import EnglishPreprocessing
from english_tokenizer import EnglishTokenizer
from vectorizer import Vectorizer

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader
from model import TextCNN
# from model_sample import CNN_NLP

# df = pd.read_csv('eng_data/spam.csv', encoding='latin1')
# df['v1'] = df['v1'].replace(['ham','spam'],[0,1])
# df.drop_duplicates(subset=['v2'], inplace=True)

def load_text(path):
    """Load text data, lowercase text and save to a list."""

    with open(path, 'rb') as f:
        texts = []
        for line in f:
            texts.append(line.decode(errors='ignore').lower().strip())

    return texts

# Load files
neg_text = load_text('../sample/Data/rt-polaritydata/rt-polarity.neg')
pos_text = load_text('../sample/Data/rt-polaritydata/rt-polarity.pos')

# Concatenate and label data
texts = neg_text + pos_text
labels = [0]*len(neg_text) + [1]*len(pos_text)

## preprocess
pp = EnglishPreprocessing()
cleaned = pp.clean(texts)
# cleaned = pp.clean(df['v2'])

## tokenize
tokenizer = EnglishTokenizer(tokenizer='treebank')
tokenized = tokenizer.tokenize(cleaned)

## vectorize
vectorizer = Vectorizer(tokenized_texts=tokenized)
encoded = vectorizer.vectorize(tokenized)
padded = vectorizer.pad(encoded)

## make tensor
X_data = torch.LongTensor(padded) ## index를 룩업해야하는 x는 long
Y_data = torch.LongTensor(labels) ## softmax: long (index)를 의미함 /sigmoid: 나중에 output과 target을 비교해야하므로 float
# Y_data = torch.LongTensor(df['v1']) ## softmax: long (index)를 의미함 /sigmoid: 나중에 output과 target을 비교해야하므로 float
full_data = TensorDataset(X_data, Y_data)

## Dataset split
train_size = int(0.9 * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size])


## Dataloader
batch_size = 50
# train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #sampler=train_sampler,
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

## device confiuration
os.environ["CUDA_VISIBLE_DEVICES"]="4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device)

## model initialize
model = TextCNN(device, num_classes=2)
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5, rho=0.95)

## training loop
epochs = 20
for epoch in range(1,epochs+1):
    for i, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        target = target.to(device) ## target은 1차원
        optimizer.zero_grad()

        output = model(data) ## (8,2) 
        loss = loss_fn(output, target)

        # backward and optimize
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
