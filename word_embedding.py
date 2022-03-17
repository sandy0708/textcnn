### word2vec, glove 등의 사전 훈련된 워드 임베딩을 훈련에 넣기 위한 처리 스크립트
### 1) word2idx = {word : idx}
### 2) embedding matrix for model embedding weight 

import pdb
import numpy as np

class WordEmbedding:
    def __init__(self, embedding_path):
        with open(embedding_path, encoding="utf8") as f:
            self.lines = f.readlines()

    def make_word2idx(self):
        print("Loading word2idx from word embeddings...")
        word2idx = dict()
        for i, line in enumerate(self.lines):
            ## id는 2부터 시작 
            word = line.split(' ')[0]
            word2idx[word] = i+2
        ### <unk>, pad 처리
        word2idx['<unk>'] = 1
        word2idx['<pad>'] = 0
        return word2idx
    
    def load_word_embeddings(self):
        vocab_size = len(self.lines) + 2
        embedding_dim = len(self.lines[0].split(' ')[1:])

        lookup_table = np.zeros((vocab_size, embedding_dim))
        for i, line in enumerate(self.lines):
            arr = np.asarray(line.split(' ')[1:], dtype='float32') 
            lookup_table[i+2] = arr

        ### unk 값 처리 : 1) random 2) average
        lookup_table[1] = np.random.uniform(-0.25, 0.25, lookup_table[1].shape)
        ####
        
        print(f"Total words in lookup table : {len(lookup_table)}")
        print(f"Shape of word embedding : {(len(lookup_table), len(lookup_table[0]))}")
        return lookup_table


if __name__ == "__main__":
    we = WordEmbedding('../04_word_embedding/glove.6B.100d.txt')
    word2idx = we.make_word2idx()
    table = we.load_word_embeddings()
