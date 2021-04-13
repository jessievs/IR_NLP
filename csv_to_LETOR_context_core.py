import pandas as pd
from tqdm import tqdm

training_data_context = pd.read_csv('features_context/training_data_context.csv', index_col=0)
training_data = pd.read_csv('features_core/training_data.csv', index_col=0)
training_data_nonrelevant = pd.read_csv('features_core/training_data_nonrelevant.csv', index_col=0)
training_data = training_data.append(training_data_nonrelevant).reset_index() # these are the core features
del training_data_nonrelevant

with open('features_context/training_data_context_albert_core.txt', 'w+') as file:
    for index in tqdm(range(len(training_data_context))):
        row_context = training_data_context.iloc[index] # NLP features
        row = training_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_context['score_qa_albert'], int(row['docid'])))

with open('features_context/training_data_context_distilbert_core.txt', 'w+') as file:
    for index in tqdm(range(len(training_data_context))):
        row_context = training_data_context.iloc[index] # NLP features
        row = training_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_context['score_qa_distilbert'], int(row['docid'])))

validation_data_context = pd.read_csv('features_context/validation_data_context.csv', index_col=0)
validation_data = pd.read_csv('features_core/validation_data.csv', index_col=0)
validation_data_nonrelevant = pd.read_csv('features_core/validation_data_nonrelevant.csv', index_col=0)
validation_data = validation_data.append(validation_data_nonrelevant).reset_index() # these are the core features
del validation_data_nonrelevant

with open('features_context/validation_data_context_albert_core.txt', 'w+') as file:
    for index in tqdm(range(len(validation_data_context))):
        row_context = validation_data_context.iloc[index] # NLP features
        row = validation_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_context['score_qa_albert'], int(row['docid'])))

with open('features_context/validation_data_context_distilbert_core.txt', 'w+') as file:
    for index in tqdm(range(len(validation_data_context))):
        row_context = validation_data_context.iloc[index] # NLP features
        row = validation_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_context['score_qa_distilbert'], int(row['docid'])))


testing_data_context = pd.read_csv('features_context/testing_data_context.csv', index_col=0)
testing_data = pd.read_csv('features_core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('features_core/testing_data_nonrelevant.csv', index_col=0)
testing_data = testing_data.append(testing_data_nonrelevant).reset_index() # these are the core features
del testing_data_nonrelevant

with open('features_context/testing_data_context_albert_core.txt', 'w+') as file:
    for index in tqdm(range(len(testing_data_context))):
        row_context = testing_data_context.iloc[index] # NLP features
        row = testing_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_context['score_qa_albert'], int(row['docid'])))


with open('features_context/testing_data_context_distilbert_core.txt', 'w+') as file:
    for index in tqdm(range(len(testing_data_context))):
        row_context = testing_data_context.iloc[index] # NLP features
        row = testing_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], int(row_context['score_qa_distilbert']), int(row['docid'])))
