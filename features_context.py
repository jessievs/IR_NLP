import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline


print('Loading collection...')

# Load collection
passages = pd.read_csv('collections/msmarco-passage/collectionandqueries/collection.tsv', sep='\t',
                       names=['pid', 'passage'])

print('DONE')

print('Creating contextual models...')

# Store the model we want to use
albert = "albert-large-v2"
distilbert = "distilbert-base-uncased"

# We need to create the pipeline
qa_albert = pipeline('question-answering', model=albert, tokenizer=albert)
qa_distilbert = pipeline('question-answering', model=distilbert, tokenizer=distilbert)

def compute_features(data, queries, data_size):
    # Creating new features for NLP features
    NLP_data = []

    # Going through all query-document pairs
    for index in tqdm(range(data_size)):
        row = data.loc[index]

        # Identifying relevant query and passage/document
        query = queries[queries['qid'] == row['qid']]['query'].values[0]
        passage = passages[passages['pid'] == row['docid']]['passage'].values[0]

        # Computing answer according to passage and its score
        result_albert = qa_albert(context=passage, question=query).get('score')
        result_distilbert = qa_distilbert(context=passage, question=query).get('score')

        # Adding NLP features to data
        NLP_data.append([row['qid'], row['docid'], row['rating'], result_albert, result_distilbert])

    NLP_data = pd.DataFrame(NLP_data, columns=['qid', 'docid', 'rating', 'score_qa_albert', 'score_qa_distilbert'])

    return NLP_data

print('DONE')

print('Loading training data...')

# Load training data
training_data = pd.read_csv('features_core/training_data.csv', index_col=0)
training_data_nonrelevant = pd.read_csv('features_core/training_data_nonrelevant.csv', index_col=0)
training_data = training_data.append(training_data_nonrelevant).reset_index()
queries_train = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.train.tsv', sep='\t',
                            names=['qid', 'query'])
del training_data_nonrelevant

print('DONE')

print('Calculating features...')

# # Obtaining features of text

training_data_NLP = compute_features(training_data, queries_train, 20000)
del training_data, queries_train

print('DONE')

print('Saving to CSV...')

# Save to csv
training_data_NLP.to_csv('features_context/training_data_NLP.csv')
del training_data_NLP

print('DONE')

print('Loading testing data...')

# Load testing data
testing_data = pd.read_csv('features_core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('features_core/testing_data_nonrelevant.csv', index_col=0)
testing_data = testing_data.append(testing_data_nonrelevant).reset_index()
queries_test = pd.read_csv('collections/msmarco-passage/msmarco-test2019-queries.tsv', sep = '\t',
                           names=['qid', 'query'])
del testing_data_nonrelevant

print('DONE')

print('Calculating features...')

testing_data_NLP = compute_features(testing_data, queries_test, len(testing_data))
del testing_data, queries_test

print('DONE')

print('Saving to CSV...')

# Save to csv
testing_data_NLP.to_csv('features_context/testing_data_NLP.csv')
del testing_data_NLP  # free up space

print('DONE')

print('Loading validation data....')

# Load validation data
validation_data = pd.read_csv('features_core/validation_data.csv', index_col=0)
validation_data_nonrelevant = pd.read_csv('features_core/validation_data_nonrelevant.csv', index_col=0)
validation_data = validation_data.append(validation_data_nonrelevant).reset_index()
queries_val = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.dev.small.tsv', sep = '\t', \
                          names=['qid', 'query'])
del validation_data_nonrelevant

print('DONE')

print('Calculating features...')

validation_data_NLP = compute_features(validation_data, queries_val, len(validation_data))
del validation_data, queries_val

print('DONE')

print('Saving to CSV...')

# Save to csv
validation_data_NLP.to_csv('features_context/validation_data_NLP.csv')
del validation_data_NLP

print('DONE')
