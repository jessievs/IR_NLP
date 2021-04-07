import pandas as pd
from tqdm import tqdm
import torch
from transformers import AlbertTokenizer, AlbertModel, DistilBertTokenizer, DistilBertModel
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine

print('Loading collection...')
passages = pd.read_csv('collections/msmarco-passage/collectionandqueries/collection.tsv', sep='\t',
                       names=['pid', 'passage'])
print('DONE')

def load_vectors(limit, fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    x=0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(tokens[1:])
        x+=1
        if x>limit:
            break
    return data

print('Reading pretrained wordvec [FASTTEXT] models (160.000 vectors)...')
# To download: curl https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip --output fasttext/vectors-english/wiki-news-300d-1M.vec.zip
# Afet downloading, unzip file
d_fast = load_vectors(160000, 'fasttext/vectors-english/wiki-news-300d-1M.vec')
print('DONE')

print('Reading pretrained wordvec [GLOVE] model (240.000 vectors of size 200)...')
# To download: curl http://nlp.stanford.edu/data/glove.6B.zip --output glove/data/glove.6B.zip
# Afer downloading, upzip file
d_glove = load_vectors(240000, 'glove/data/glove.6B.200d.txt')
d = [d_fast, d_glove]
print('DONE')

def createVec(doc, d):
    if d == d_fast:
        vecq = np.zeros(300)
    else:
        vecq = np.zeros(200)
    count = 0
    for x in doc.split():
        if d.get(x) is not None:
           count = count + 1
           vecq = np.add(vecq, np.array(d.get(x)).astype(np.float), out=vecq, casting="unsafe")
    if count<1 :
        count =1
    return vecq/count

def compute_NLP_features(data, queries):
    # Creating new features for NLP features
    NLP_data = []

    # Going through all query-document pairs
    for index in tqdm(range(len(data))):
        row = data.loc[index]

        # Identifying relevant query and passage/document
        query = queries[queries['qid'] == row['qid']]['query'].values[0]
        passage = passages[passages['pid'] == row['docid']]['passage'].values[0]
        cos = [None] * len(d)
        for idx, x in enumerate(d):
            vec = createVec(query, x)
            vecq = createVec(passage, x)
            cos[idx] = cosine(vec.reshape((1, -1)), vecq.reshape((1, -1)))[0]

        # Adding NLP features to data
        NLP_data.append([row['qid'], row['docid'], row['rating'], cos[0], cos[1]])

    NLP_data = pd.DataFrame(NLP_data, columns=['qid', 'docid', 'rating', 'FastText-COSINE', 'Glove-COSINE'])

    return NLP_data

print('Loading training data...')

training_data = pd.read_csv('features_core/training_data.csv', index_col=0)
training_data_nonrelevant = pd.read_csv('features_core/training_data_nonrelevant.csv', index_col=0)
training_data = training_data.append(training_data_nonrelevant).reset_index()
queries_train = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.train.tsv', sep='\t', names=['qid', 'query'])
del training_data_nonrelevant

print('DONE')

print('Calculating features training data...')
training_data_NLP = compute_NLP_features(training_data , queries_train)
del training_data, queries_train
print('DONE')

print('Saving training data to CSV...')
training_data_NLP.to_csv('features_word/training_data_word.csv')
del training_data_NLP
print('DONE')

print('Loading testing data...')

testing_data = pd.read_csv('features_core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('features_core/testing_data_nonrelevant.csv', index_col=0)
testing_data = testing_data.append(testing_data_nonrelevant).reset_index()
queries_test = pd.read_csv('collections/msmarco-passage/msmarco-test2019-queries.tsv', sep = '\t',
                           names=['qid', 'query'])
del testing_data_nonrelevant

print('DONE')

print('Calculating features testing data...')
testing_data_NLP = compute_NLP_features(testing_data , queries_test)
del testing_data, queries_test
print('DONE')

print('Saving testing data to CSV...')
testing_data_NLP.to_csv('features_word/testing_data_word.csv')
del testing_data_NLP
print('DONE')

print('Loading validation data....')

validation_data = pd.read_csv('features_core/validation_data.csv', index_col=0)
validation_data_nonrelevant = pd.read_csv('features_core/validation_data_nonrelevant.csv', index_col=0)
validation_data = validation_data.append(validation_data_nonrelevant).reset_index()
queries_val = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.dev.small.tsv', sep = '\t', names=['qid', 'query'])
del validation_data_nonrelevant

print('DONE')

print('Calculating features validation data...')
validation_data_NLP = compute_NLP_features(validation_data , queries_val)
del validation_data, queries_val
print('DONE')

print('Saving validation data to CSV...')
validation_data_NLP.to_csv('features_word/validation_data_word.csv')
del validation_data_NLP
print('DONE')
