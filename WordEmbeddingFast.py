import pandas as pd
from tqdm import tqdm
import torch
from transformers import AlbertTokenizer, AlbertModel, DistilBertTokenizer, DistilBertModel
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine

print('Loading collection...')

# Load collection 
passages = pd.read_csv('collections/msmarco-passage/collectionandqueries/collection.tsv', sep='\t',
                       names=['pid', 'passage'])

print('DONE')

print('Loading training data...')

# Load training data
# training_data = pd.read_csv('core/training_data.csv', index_col=0)
# training_data_nonrelevant = pd.read_csv('core/training_data_nonrelevant.csv', index_col=0)
# training_data = training_data.append(training_data_nonrelevant).reset_index()
# queries_train = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.train.tsv', sep='\t',
#                             names=['qid', 'query'])

print('DONE')

print('Loading testing data...')

# Load testing data
testing_data = pd.read_csv('core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('core/testing_data_nonrelevant.csv', index_col=0)
testing_data = testing_data.append(testing_data_nonrelevant).reset_index()
queries_test = pd.read_csv('collections/msmarco-passage/msmarco-test2019-queries.tsv', sep = '\t',
                           names=['qid', 'query'])

print('DONE')

print('Loading validation data....')

# Load validation data
# validation_data = pd.read_csv('core/validation_data.csv', index_col=0)
# validation_data_nonrelevant = pd.read_csv('core/validation_data_nonrelevant.csv', index_col=0)
# validation_data = validation_data.append(validation_data_nonrelevant).reset_index()
# queries_val = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.dev.small.tsv', sep = '\t', \
#                           names=['qid', 'query'])

print('DONE')

# print('Reading pretrained wordvec models (95.000 vectors)...')

# Store the model we want to use
# albert = "albert-base-v2"
# distilbert = "distilbert-base-uncased"

# We need to create the model and tokenizer
# t_albert = AlbertTokenizer.from_pretrained(albert)
# m_albert = AlbertModel.from_pretrained(albert)
# t_distilbert = DistilBertTokenizer.from_pretrained(distilbert)
# m_distilbert = DistilBertModel.from_pretrained(distilbert)

print('DONE')

print('Calculating features...')

# # Obtaining features of tex
def load_vectors(limit, fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    x=0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(tokens[1:])
        x+=1
        if x>limit:
            break
    return data

def createVec(doc, d):
    vecq = np.zeros(300)
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
    print('Reading pretrained wordvec models (160.000 vectors)...')
    d = load_vectors(160000, 'wiki-news-300d-1M.vec(1)/wiki-news-300d-1M.vec')
    # Going through all query-document pairs
    #for index in tqdm(range(5)):
    for index in tqdm(range(len(data))):
        row = data.loc[index]

        # Identifying relevant query and passage/document
        query = queries[queries['qid'] == row['qid']]['query'].values[0]
        passage = passages[passages['pid'] == row['docid']]['passage'].values[0]
        vec = createVec(query, d)
        vecq = createVec(passage, d)
        cos = cosine(vec.reshape((1, -1)), vecq.reshape((1, -1)))
        
        # Adding NLP features to data
        NLP_data.append(cos[0])
#        print(index,"/",len(data),"\tvalue: ", cos[0])
#        if(index > 50):
#            break
        
    NLP_data = pd.DataFrame(NLP_data, columns=["FastText-COSINE"])
    
    return NLP_data

training_data_NLP = compute_NLP_features(testing_data , queries_test)

print('DONE')

print(training_data_NLP)

print('Saving to CSV...')

# Save to csv
training_data_NLP.to_csv('testing_data_NLP_FASTTEXT_160000.csv')

print('DONE')


