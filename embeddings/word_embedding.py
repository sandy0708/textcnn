### word2vec, glove 등의 사전 훈련된 워드 임베딩을 훈련에 넣기 위한 처리 스크립트
### 1) word2idx = {word : idx}
### 2) embedding matrix for model embedding weight 

import numpy as np
from gensim.models import KeyedVectors

class WordEmbedding:
    def __init__(self, embedding_type='word2vec' ,embedding_dim=300):
        self.embedding_type = embedding_type
        if self.embedding_type == 'word2vec':
            print("Loading word2vec vectors...")
            self.w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        elif embedding_type == 'glove':
            with open(f'glove.6B.{embedding_dim}d.txt', encoding="utf8") as f:
                self.lines = f.readlines()

    def make_word2idx(self):
        print(f"Loading word2idx from {self.embedding_type} embeddings...")

        ## id는 2부터 시작 
        if self.embedding_type == 'word2vec':
            word2idx = {token: token_index+2 for token_index, token in enumerate(self.w2v.index2word)} 
        elif self.embedding_type == 'glove':
            word2idx = dict()
            for i, line in enumerate(self.lines):
                word = line.split(' ')[0]
                word2idx[word] = i+2

        ### unk, pad 처리
        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1
        return word2idx
    
    def load_word_embeddings(self):
        if self.embedding_type == 'word2vec':
            lookup_table = self.w2v.vectors

        elif self.embedding_type == 'glove':
            vocab_size = len(self.lines) + 2
            embedding_dim = len(self.lines[0].split(' ')[1:])

            lookup_table = np.zeros((vocab_size, embedding_dim))
            for i, line in enumerate(self.lines):
                arr = np.asarray(line.split(' ')[1:], dtype='float32') 
                lookup_table[i+2] = arr

        ### unk 값 처리 방법 선택 : 1) random 2) average ###
        lookup_table[1] = np.random.uniform(-0.25, 0.25, lookup_table[1].shape)

        
        print(f"Total words in lookup table : {len(lookup_table)}")
        print(f"Shape of word embedding : {(len(lookup_table), len(lookup_table[0]))}")
        return lookup_table


if __name__ == "__main__":
    we = WordEmbedding(embedding_type='word2vec', embedding_dim=300)
    word2idx = we.make_word2idx()
    table = we.load_word_embeddings()
    pdb.set_trace() ## 일치여부 확인 ㅇ