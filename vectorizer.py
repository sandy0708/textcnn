### 토큰화된 데이터를 word embedding에 맞게 정수 인코딩 진행
### 패딩 진행

import sys
sys.path.append('../../04_word_embedding')
from word_embedding import WordEmbedding

class Vectorizer:
    def __init__(self, pretrained=True, tokenized_texts=None, embedding_dim=100):
        if pretrained == True: ## if True, use pretrained word embedding
            we = WordEmbedding(f'glove.6B.{embedding_dim}d.txt')
            self.word2idx = we.make_word2idx()
        else:
            self.word2idx = self.word2idx_from_data(tokenized_texts)
        
    def word2idx_from_data(self, tokenized_texts):
        '''Create word2idx with training dataset'''
        print('Creating word2idx from dataset...')
        word2idx = {}

        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1

        # Add new token to `word2idx`
        idx = 2
        for tokenized_sent in tokenized_texts:
            for token in tokenized_sent:
                if token not in word2idx:
                    word2idx[token] = idx
                    idx += 1      
        print(f'Vocab size is {len(word2idx)}')  

        return word2idx 

    def vectorize(self, sents):
        print("Vectorizing tokens...")
        encoded = []
        for tokens in sents:
            sent_enc = []
            for tok in tokens:
                if tok in self.word2idx:
                    sent_enc.append(self.word2idx[tok])
                else:
                    sent_enc.append(0)
            encoded.append(sent_enc)

        return encoded

    def pad(self, sents):
        max_length = len(max(sents, key=len))
        pad_id = self.word2idx['<pad>']
        print(f"Padding {pad_id} for max length {max_length}...")
        padded = []
        for ids in sents:
            pad_ids = ids + [pad_id] * (max_length - len(ids)) ## ids에 더하는 형식으로 하면 기존의 값이 바뀜
            padded.append(pad_ids)
        return padded

        

if __name__ == "__main__":
    samples = [['for', 'fear', 'of', 'fainting', 'with', 'the', 'of', 'all', 'that', 'housework', 'you', 'just', 'did'],
                ["i", 'been', 'searching', 'for', 'the', 'right', 'words', 'to', 'thank', 'you']]
    v = Vectorizer()
    token_ids = v.vectorize(samples)
    padded = v.pad(token_ids)
    print(token_ids)
    print(padded)