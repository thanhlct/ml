#the source mainly refers to:
#- https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
#- https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-use-conll-2002-data-to-build-a-ner-system 

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from data_reader import read_vlsp_data

import pdb

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, chunk_label, label, _ in sent]


if __name__ == '__main__':
    print('Loading corpus...')
    train, valid, test = read_vlsp_data('../../corpora/ner/vlsp/')

    print('Extracting features...')
    trainx = [sent2features(sent) for sent in train]
    trainy = [sent2labels(sent) for sent in train]

    testx = [sent2features(sent) for sent in test]
    testy = [sent2labels(sent) for sent in test]

    print('Training model...')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True
    )
    crf.fit(trainx, trainy)
    
    print('Evaluating model...')
    labels = list(crf.classes_)
    labels.remove('O')#only evaluate all labels except O - other
    
    predict_y = crf.predict(trainx)
    print('Train: ', metrics.flat_f1_score(trainy, predict_y,
                      average='weighted', labels=labels))
   
    predict_y = crf.predict(testx)
    print('Test: ', metrics.flat_f1_score(testy, predict_y,
                      average='weighted', labels=labels))
'''=======Results notes:
- Oct 20: train: 0.994, test: 0.889
'''
