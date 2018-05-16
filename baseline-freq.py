from sklearn.linear_model import LogisticRegression
from collections import Counter

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        words=word
        return [len_chars, len_tokens,words]

    def train(self, trainset,testset):
        X = []
        y = []
        x1=[]
        for sent in trainset:
            x1.append(self.extract_features(sent['target_word'])[2])
        for sent in testset:
            x1.append(self.extract_features(sent['target_word'])[2])
        x2=Counter(x1)
        
        for sent in trainset:
            X.append(np.hstack((self.extract_features(sent['target_word'])[:2],x2[self.extract_features(sent['target_word'])[2]])))
            y.append(sent['gold_label'])
        self.model.fit(X, y)

    def test(self, testset):
        x1=[]
        X=[]
        for sent in testset:
            x1.append(self.extract_features(sent['target_word'])[2])
        x2=Counter(x1)
        
        for sent in testset:

            X.append(np.hstack((self.extract_features(sent['target_word'])[:2],x2[self.extract_features(sent['target_word'])[2]])))

        return self.model.predict(X)
