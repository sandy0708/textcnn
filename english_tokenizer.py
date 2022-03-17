### Tokenizer class
### 임포트 시 이름만 명시하면 사용 가능

from nltk.tokenize import TreebankWordTokenizer

class EnglishTokenizer:
    def __init__(self, tokenizer):
        '''Initializes tokenizer (Available : treebank) '''
        if tokenizer == 'treebank':
            self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, sent_arr):
        '''Returns tokenized sentence array'''
        tokenized_list = []
        for sent in sent_arr:
            tok_sent = self.tokenizer.tokenize(sent)
            tokenized_list.append(tok_sent)

        return tokenized_list


if __name__ == '__main__':
    ## 사용할 토크나이저 불러오기
    sample_data = ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
                 "FreeMsg Hey there darling it's been 3 week's now and no word back!"]

    tokenizer = EnglishTokenizer(tokenizer='treebank')
    print(sample_data)
    print(tokenizer.tokenize(sample_data))