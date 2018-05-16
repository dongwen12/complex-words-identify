from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
torch.manual_seed(8)

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
        words=word
        len_tokens = len(word.split(' '))

        return [len_chars,words, len_tokens]
    
    def dict1(self, trainset,testset):
        X1 = []
        dictvec={}
        for sent in trainset+testset:
            X1.append(self.extract_features(sent['target_word'])[1])
        vocab = set(X1)
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        embeds = nn.Embedding(len(vocab), 8)
        for word in vocab:
            lookup_tensor = torch.tensor([word_to_ix[word]], dtype=torch.long)
            dictvec[word]=np.hstack((embeds(lookup_tensor).detach().numpy()[0],self.extract_features(sent['target_word'])[0],self.extract_features(sent['target_word'])[2]))
        return dictvec
            
            
    def train(self, trainset,testset):
        X1 = self.dict1(trainset,testset)
        y = []
        X = []
        for sent in trainset:
            y.append(sent['gold_label'])
            X.append(np.hstack((X1[self.extract_features(sent['target_word'])[1]],self.extract_features(sent['target_word'])[0],self.extract_features(sent['target_word'])[2])))            
        self.model.fit(X, y)

    def test(self, trainset,testset):
        X1 = self.dict1(trainset,testset)
        X = []
        for sent in testset:
            X.append(np.hstack((X1[self.extract_features(sent['target_word'])[1]],self.extract_features(sent['target_word'])[0],self.extract_features(sent['target_word'])[2])))            
        return self.model.predict(X)
