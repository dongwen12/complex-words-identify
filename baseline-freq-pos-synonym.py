from sklearn.linear_model import LogisticRegression
from collections import Counter
import nltk
import numpy as np
from sklearn import svm, naive_bayes
from nltk.corpus import wordnet as wn
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model =AdaBoostClassifier()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        words=word
        synonym=len(wn.synsets(re.sub("[^\w]"," ",words)))
        pos=nltk.pos_tag([words])
        return [len_chars, len_tokens,pos,words,synonym]

    def train(self, trainset,testset):
        X = []
        y = []
        x1=[]
        x3=[]
        for sent in trainset+testset:
            x1.append(self.extract_features(sent['target_word'])[3])
            x3.append(self.extract_features(sent['target_word'])[2][0][1])
        x2=Counter(x1)
        vocab = set(x3)
        pos_to_ix = {word: i for i, word in enumerate(vocab)}
        for sent in trainset:
            X.append(np.hstack((self.extract_features(sent['target_word'])[:2], pos_to_ix[self.extract_features(sent['target_word'])[2][0][1]],x2[self.extract_features(sent['target_word'])[3]],self.extract_features(sent['target_word'])[4])))
            y.append(sent['gold_label'])
        self.model.fit(X, y)
        plt.figure()
        plt.title("Learning Curves ")
        plt.xlabel("Training samples")
        plt.ylabel("Score")
        estimator = self.model
        train_sizes = np.linspace(.1, 1.0,5)
        train_sizes, train_scores,test_scores= learning_curve(estimator, X, y, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
        plt.show()
    
                       
        
    def test(self,trainset, testset):
        x1=[]
        X=[]
        x3=[]
        for sent in trainset+testset:
            x1.append(self.extract_features(sent['target_word'])[3])
            x3.append(self.extract_features(sent['target_word'])[2][0][1])
        x2=Counter(x1)
        vocab = set(x3)
        pos_to_ix = {word: i for i, word in enumerate(vocab)}
        for sent in testset:
            X.append(np.hstack((self.extract_features(sent['target_word'])[:2],pos_to_ix[self.extract_features(sent['target_word'])[2][0][1]],x2[self.extract_features(sent['target_word'])[3]],self.extract_features(sent['target_word'])[4])))

        return self.model.predict(X)