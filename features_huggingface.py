import pandas as pd
from tqdm import tqdm
import torch
from transformers import AlbertTokenizer, AlbertModel, DistilBertTokenizer, DistilBertModel

print('Loading collection...')

# Load collection
passages = pd.read_csv('collections/msmarco-passage/collectionandqueries/collection.tsv', sep='\t',
                       names=['pid', 'passage'])

print('DONE')

print('Creating contextual models...')

# Store the model we want to use
albert = "albert-large-v2"
distilbert = "distilbert-base-uncased"

# We need to create the model and tokenizer
t_albert = AlbertTokenizer.from_pretrained(albert)
m_albert = AlbertModel.from_pretrained(albert)
t_distilbert = DistilBertTokenizer.from_pretrained(distilbert)
m_distilbert = DistilBertModel.from_pretrained(distilbert)

def compute_features(data, queries):
    # Creating new features for NLP features
    NLP_data = []

    # Going through all query-document pairs
    for index in tqdm(range(len(data))):
        row = data.loc[index]

        # Identifying relevant query and passage/document
        query = queries[queries['qid'] == row['qid']]['query'].values[0]
        passage = passages[passages['pid'] == row['docid']]['passage'].values[0]

        # Computing model input
        input_albert = t_albert.encode_plus(query, passage, return_tensors='pt')
        input_distilbert = t_distilbert.encode_plus(query, passage, return_tensors='pt')

        # Obtaining model output probabilities
        output_albert = m_albert(**input_albert)
        output_distilbert = m_distilbert(**input_distilbert)
 
        last_hidden_state_albert = output_albert.last_hidden_state
        last_hidden_state_distilbert = output_distilbert.last_hidden_state

        # Compute the logits
        softmax = torch.nn.Softmax(dim=1)
        logits_albert = softmax(last_hidden_state_albert)
        logits_distilbert = softmax(last_hidden_state_distilbert)

        # Adding NLP features to data
        NLP_data.append([p_albert, p_distilbert])

    NLP_data = pd.DataFrame(NLP_data, columns=['albert', 'distilbert'])

    return NLP_data

print('DONE')

print('Loading training data...')

# Load training data
training_data = pd.read_csv('core/training_data.csv', index_col=0)
training_data_nonrelevant = pd.read_csv('core/training_data_nonrelevant.csv', index_col=0)
training_data = training_data.append(training_data_nonrelevant).reset_index()
del training_data_nonrelevant  # free up space
queries_train = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.train.tsv', sep='\t',
                            names=['qid', 'query'])

print('DONE')

print('Calculating features...')

# # Obtaining features of text

training_data_NLP = compute_features(training_data, queries_train)
del training_data, queries_train  # free up space

print('DONE')

print('Saving to CSV...')

# Save to csv
training_data_NLP.to_csv('training_data_NLP.csv')
del training_data_NLP  # free up space

print('DONE')

print('Loading testing data...')

# Load testing data
testing_data = pd.read_csv('core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('core/testing_data_nonrelevant.csv', index_col=0)
testing_data = testing_data.append(testing_data_nonrelevant).reset_index()
del testing_data_nonrelevant                                                             # free up space
queries_test = pd.read_csv('collections/msmarco-passage/msmarco-test2019-queries.tsv', sep = '\t',
                           names=['qid', 'query'])

print('DONE')

print('Calculating features...')

testing_data_NLP = compute_features(testing_data, queries_test)
del testing_data, queries_test  # free up space

print('DONE')

print('Saving to CSV...')

# Save to csv
testing_data_NLP.to_csv('testing_data_NLP.csv')
del testing_data_NLP  # free up space

print('DONE')

print('Loading validation data....')

# Load validation data
validation_data = pd.read_csv('core/validation_data.csv', index_col=0)
validation_data_nonrelevant = pd.read_csv('core/validation_data_nonrelevant.csv', index_col=0)
validation_data = validation_data.append(validation_data_nonrelevant).reset_index()
del validation_data_nonrelevant                                                             # free up space
queries_val = pd.read_csv('collections/msmarco-passage/collectionandqueries/queries.dev.small.tsv', sep = '\t', \
                          names=['qid', 'query'])

print('DONE')

print('Calculating features...')

validation_data_NLP = compute_features(validation_data, queries_val)
del validation_data, queries_val  # free up space

print('DONE')

print('Saving to CSV...')

# Save to csv
validation_data_NLP.to_csv('validation_data_NLP.csv')
del validation_data_NLP  # free up space

print('DONE')
