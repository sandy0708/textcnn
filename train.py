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
vectorizer = Vectorizer(tokenized_texts=tokenized, pretrained=True)
vocab_size = len(vectorizer.word2idx)
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
batch_size = 16
# train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #sampler=train_sampler,
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

## device confiuration
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device)

## model initialize
model = TextCNN(num_classes=2, vocab_size=vocab_size)
# model = CNN_CLS(vocab_size=len(vectorizer.word2idx), embed_dim=300)
# model = CNN_NLP(vocab_size=len(vectorizer.word2idx), embed_dim=300)
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5, rho=0.95)

## training loop
epochs = 20
for epoch in range(1,epochs+1):
    for i, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        target = target.to(device) ## target은 1차원으로 되어있음 
        optimizer.zero_grad()

        output = model(data) 
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

## test 
model.eval()
test_accuracy = []
test_loss = []
for i, (data, target) in enumerate(test_dataloader):
    data = data.to(device)
    target = target.to(device) 
    optimizer.zero_grad()

    with torch.no_grad():
        output = model(data) 
    
    loss = loss_fn(output, target)
    test_loss.append(loss.item())

    preds = torch.argmax(output, dim=1).flatten()
    accuracy = (preds == target).cpu().numpy().mean() * 100
    test_accuracy.append(accuracy)

test_loss = np.mean(test_loss)
test_accuracy = np.mean(test_accuracy)

print('Average loss :', test_loss)
print('Average accuracy :', test_accuracy)