import pandas as pd
from tqdm import tqdm

training_data = pd.read_csv('features_word/training_data_word.csv', index_col=0)

with open('features_word/training_data_word_fasttext.txt', 'w') as file:
    for index in tqdm(range(len(training_data))):
        row = training_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['FastText-COSINE'], int(row['docid'])))

with open('features_word/training_data_word_glove.txt', 'w') as file:
    for index in tqdm(range(len(training_data))):
        row = training_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['Glove-COSINE'], int(row['docid'])))

validation_data = pd.read_csv('features_word/validation_data_word.csv', index_col=0)

with open('features_word/validation_data_word_fasttext.txt', 'w') as file:
    for index in tqdm(range(len(validation_data))):
        row = validation_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['FastText-COSINE'], int(row['docid'])))

with open('features_word/validation_data_word_glove.txt', 'w') as file:
    for index in tqdm(range(len(validation_data))):
        row = validation_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['Glove-COSINE'], int(row['docid'])))


testing_data = pd.read_csv('features_word/testing_data_word.csv', index_col=0)

with open('features_word/testing_data_word_fasttext.txt', 'w') as file:
    for index in tqdm(range(len(testing_data))):
        row = testing_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['FastText-COSINE'], int(row['docid'])))

with open('features_word/testing_data_word_glove.txt', 'w') as file:
    for index in tqdm(range(len(testing_data))):
        row = testing_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['Glove-COSINE'], int(row['docid'])))
