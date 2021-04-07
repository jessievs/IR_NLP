import pandas as pd
from tqdm import tqdm

training_data_word = pd.read_csv('features_word/training_data_word.csv', index_col=0)
training_data = pd.read_csv('features_core/training_data.csv', index_col=0)
training_data_nonrelevant = pd.read_csv('features_core/training_data_nonrelevant.csv', index_col=0)
training_data = training_data.append(training_data_nonrelevant).reset_index() # these are the core features
del training_data_nonrelevant

with open('features_word/training_data_word_fasttext_core.txt', 'w') as file:
    for index in tqdm(range(len(training_data))):
        row_word = training_data_word.iloc[index] # NLP features
        row = training_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_word['FastText-COSINE'], int(row['docid'])))

with open('features_word/training_data_word_glove_core.txt', 'w') as file:
    for index in tqdm(range(len(training_data))):
        row_word = training_data_word.iloc[index] # NLP features
        row = training_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_word['Glove-COSINE'], int(row['docid'])))

validation_data_word = pd.read_csv('features_word/validation_data_word.csv', index_col=0)
validation_data = pd.read_csv('features_core/validation_data.csv', index_col=0)
validation_data_nonrelevant = pd.read_csv('features_core/validation_data_nonrelevant.csv', index_col=0)
validation_data = validation_data.append(validation_data_nonrelevant).reset_index() # these are the core features
del validation_data_nonrelevant

with open('features_word/validation_data_word_fasttext_core.txt', 'w') as file:
    for index in tqdm(range(len(validation_data))):
        row_word = validation_data_word.iloc[index] # NLP features
        row = validation_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_word['FastText-COSINE'], int(row['docid'])))

with open('features_word/validation_data_word_glove_core.txt', 'w') as file:
    for index in tqdm(range(len(validation_data))):
        row_word = validation_data_word.iloc[index] # NLP features
        row = validation_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_word['Glove-COSINE'], int(row['docid'])))


testing_data_word = pd.read_csv('features_word/testing_data_word.csv', index_col=0)
testing_data = pd.read_csv('features_core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('features_core/testing_data_nonrelevant.csv', index_col=0)
testing_data = testing_data.append(testing_data_nonrelevant).reset_index() # these are the core features
del testing_data_nonrelevant

with open('features_word/testing_data_word_fasttext_core.txt', 'w') as file:
    for index in tqdm(range(len(testing_data))):
        row_word = testing_data_word.iloc[index] # NLP features
        row = testing_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], row_word['FastText-COSINE'], int(row['docid'])))


with open('features_word/testing_data_word_glove_core.txt', 'w') as file:
    for index in tqdm(range(len(testing_data))):
        row_word = testing_data_word.iloc[index] # NLP features
        row = testing_data.iloc[index]      # core features
        file.write('{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), row['bm25'], row['passage_length'], row['c'], row['df'], row['cf'], row['idf'], row['c_idf'], int(row_word['Glove-COSINE']), int(row['docid'])))
