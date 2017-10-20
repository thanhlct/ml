import glob
import random
from collections import defaultdict

import pdb

random.seed(1)

def read_vlsp_data(data_dir):
    '''Read all data to a list of sentences where each sentece is a list of words and a word is atuple including word, .... and ner labels.
    '''
    #load all data to memory
    sents = []
    for filename in glob.iglob(data_dir + '*.txt'):
        print('Read vlsp file: ', filename)
        with open(filename, 'r') as f:
            f.readline()#keep titles and etc.
            f.readline()
            f.readline()
            for line in f:
                line = line.strip()
                if line == '<s>':
                    sent = []
                    continue
                elif line == '</s>':
                    sents.append(sent)
                else:
                    ms = line.split('\t')
                    if len(ms)==5:
                        ms[3] = 'O' if ms[3].strip() in ['Ô', 'o', 'OR', 'Cc', 'R', '0', 'Oc', 'P', 'Y', 'Ob', 'Os', 'B', ''] else ms[3].strip()#fill in missing values
                        sent.append(ms)
                    elif line!='':
                        print('Incorrect format:', line)

    #divide train/valid/test set: rate train 60%, valid 20%, test 20%
    num_sent = len(sents)
    random.shuffle(sents)
    train_to_id = round(num_sent*0.6)
    valid_to_id = round(num_sent*0.8)
    train = sents[:train_to_id]
    valid = sents[train_to_id:valid_to_id]
    test = sents[valid_to_id:]

    #corpus statistic
    print('=======VLSP statistics:')
    print('Total sentences:', num_sent)
    print('Train contains {} sents, {} tokens'.format(len(train), _count_token(train)))
    print('Valid contains {} sents, {} tokens'.format(len(valid), _count_token(valid)))
    print('Test contains {} sents, {} tokens'.format(len(test), _count_token(test)))

    #corpus analysis
    print('=======Labels statistics:')
    label_occur_count = defaultdict(int)
    for sent in sents:
        for w in sent:
            label_occur_count[w[3]]+=1
    for k, v in label_occur_count.items():
        print('{}:{}'.format(k, v))

    return train, valid, test

def _count_token(sents):
    token_count=[len(sent) for sent in sents]
    return sum(token_count)

if __name__=='__main__':
    train, valid, test = read_vlsp_data('../../corpora/ner/vlsp/')
