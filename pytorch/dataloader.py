from ..vectorizer.english_preprocessing import EnglishPreprocessing
from ..vectorizer.english_tokenizer import EnglishTokenizer
from ..vectorizer.vectorizer import Vectorizer

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader


class Dataloader:
    def __init__(self, batch_size=50):
        # Load files
        neg_text = self.load_text('../sample/Data/rt-polaritydata/rt-polarity.neg')
        pos_text = self.load_text('../sample/Data/rt-polaritydata/rt-polarity.pos')

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
        vectorizer = Vectorizer(tokenized_texts=tokenized, pretrained=False)
        self.vocab_size = len(vectorizer.word2idx)
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
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #sampler=train_sampler,
        self.val_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    def load_text(self, path):
        """Load text data, lowercase text and save to a list."""
        with open(path, 'rb') as f:
            texts = []
            for line in f:
                texts.append(line.decode(errors='ignore').lower().strip())

        return texts

