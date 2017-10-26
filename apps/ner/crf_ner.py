#the source mainly refers from:
#- https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
#- https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-use-conll-2002-data-to-build-a-ner-system 

import sklearn_crfsuite
from sklearn_crfsuite import metrics

import pickle
import scipy.stats

from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV

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


def run1(trainx, trainy, validx, validy, testx, testy):
    print('Training model...')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        #c1=0.1, 
        #c2=0.1,
        c1=0.045677,
        c2=0.069943,
        max_iterations=100, 
        all_possible_transitions=True
    )
    
    #append train and valid set
    trainx.extend(validx)
    trainy.extend(validy)

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

    #Insepect per-class results
    print(metrics.flat_classification_report(testy, predict_y))

    #save/load model
    save_model(crf, 'ner_crf.stand.pkl')
    load_model('ner_crf.stand.pkl', testx, testy, labels)

def save_model(model, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(model, f)

def load_model(fpath, testx, testy, labels):
    with open(fpath, 'rb') as f:
        crf = pickle.load(f)

        predict_y = crf.predict(testx)
        print('Test when reload: ', metrics.flat_f1_score(testy, predict_y,
                      average='weighted', labels=labels))

def run2(trainx, trainy, validx, validy, testx, testy):
    print('Optimizing hyperparameters...')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        max_iterations=100, 
        all_possible_transitions=True
    )
    
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    labels = ['B-PER', 'I-PER', 'B-LOC', 'B-ORG', 'I-LOC', 'B-MISC', 'I-MISC', 'I-ORG', 'B-NP', 'B-MICS', 'I-MICS'] 
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
    
    rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)

    trainx.extend(validx)
    trainy.extend(validy)
    trainx.extend(testx)
    trainy.extend(testy)

    rs.fit(trainx, trainy)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

def main():
    print('Loading corpus...')
    train, valid, test = read_vlsp_data('../../corpora/ner/vlsp/')

    print('Extracting features...')
    trainx = [sent2features(sent) for sent in train]
    trainy = [sent2labels(sent) for sent in train]

    validx = [sent2features(sent) for sent in valid]
    validy = [sent2labels(sent) for sent in valid]

    testx = [sent2features(sent) for sent in test]
    testy = [sent2labels(sent) for sent in test]

    run1(trainx, trainy, validx, validy, testx, testy)#normal
    #run2(trainx, trainy, validx, validy, testx, testy)#optimize hyperparameters
   

if __name__ == '__main__':
    main()

'''=======Results notes:
- Oct 20: train: 0.994, test: 0.897725
- Oct 26: 3-fold optimizing hyperparameters got: 0.9039 with c1=0.045677 and c2=0.069943
- Oct 26: Run best hyperparameters, got model: 0.8983624 

CNN-biLSTM-CRF, https://arxiv.org/pdf/1705.04044.pdf, got 88.38 with complexity and 80.
'''
