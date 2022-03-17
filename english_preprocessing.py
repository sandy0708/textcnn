### Preprocessing class

import re

class EnglishPreprocessing:
    def __init__(self):
        pass

    def clean(self, sent_arr):
        '''Returns cleaned sentence array'''
        cleaned = []
        for sent in sent_arr:
            sent = sent.lower()
            sent = re.sub("[^ 'a-z]", '', sent)
            sent = sent.strip()
            cleaned.append(sent)
        return cleaned


if __name__ == '__main__':
    sample_data = ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
                 "FreeMsg Hey there darling it's been 3 week's now and no word back!"]

    pp = EnglishPreprocessing()
    cleaned = pp.clean(sample_data)
    print(cleaned)