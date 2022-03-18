### TextCNN model class
### pytorch version 
import sys
sys.path.append('../../04_word_embedding')
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from word_embedding import WordEmbedding
import pdb

class TextCNN(nn.Module):
    def __init__(self, model_ver='CNN-rand',
                    vocab_size=None, 
                    embedding_dim=300, 
                    window_size=[3,4,5],
                    num_filters = 100,
                    num_classes=2):

        super(TextCNN, self).__init__()
        if model_ver == 'CNN-rand':
            '''all words are randomly initialized and then modified during training.'''
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        elif model_ver == 'CNN-static':
            '''pre-trained vectors from word2vec are kept static.'''
            embedding_matrix = self.load_embedding_vector(embedding_dim=embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
            # embedding_dim = embedding_matrix.shape[1]
            # self.embedding = nn.Embedding(num_embeddings=embedding_matrix.shape[0],embedding_dim=embedding_dim) ## (batch, seq_len, 100)
            # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        elif model_ver == 'CNN-non-static':
            '''pretrained vectors are fine-tuned for each task'''
            embedding_matrix = self.load_embedding_vector(embedding_dim=embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        elif model_ver == 'CNN-multichannel':
            '''Both channels are initialized with word2vec, but gradients are backpropagated only through one of the channels'''
            embedding_matrix = self.load_embedding_vector(embedding_dim=embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding2 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)

        ## 여러개의 conv layer를 사용할 경우 : nn.ModuleList를 사용할 것
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=w) for w in window_size]) ### bigram filter를 통해 100 -> 1로 컨볼루션 
        # self.maxpool = nn.MaxPool1d() # 맥스풀 커널 사이즈는 conv의 출력물 사이즈에 의존하기 때문에 forward 내에서 받을 것.
        self.dropout = nn.Dropout(p=0.5) # 0.5 밑의 값들 drop
        self.fc = nn.Linear(num_filters * len(window_size), num_classes)

    def forward(self, x):
        em = self.embedding(x) # (8, 173, 100)
        print('embedding 1st batch:', em[0])
        em = em.transpose(1,2) # (8, 100, 173) # max_length, 즉 문장 길이를 기준으로 conv해야 bigram 적용가능 
        print('transposed embedding:', em[0])
        convs = [F.relu(cv(em)) for cv in self.conv1d_list] # (8, 100, 172) * 3
        print('1st filter conv, 1st batch:', convs[0][0])
        pooled = [F.max_pool1d(cs, cs.shape[-1]).squeeze() for cs in convs] # (8, 100) * 3
        print('1st max-pooled, 1st batch:', pooled[0][0])
        pool_cat = torch.cat(pooled, dim=1) # (8, 300) # 한 배치당 3,4,5 필터들을 거친 피쳐맵 concat 됨
        print('pool concat, 1st batch:', pool_cat[0])
        logit = self.fc(self.dropout(pool_cat)) # (8, 2) # 0,1
        print('logit from fc layer:', logit[0])
        return logit 

    def load_embedding_vector(self, embedding_dim=100):
        embed = WordEmbedding(f'glove.6B.{embedding_dim}d.txt')
        lookup_table = embed.load_word_embeddings() # load embedding matrix in np array form
        return lookup_table



if __name__ == '__main__':
    model = TextCNN()