import pandas as pd
from tqdm import tqdm

training_data = pd.read_csv('features_context/training_data_NLP.csv', index_col=0)

with open('features_context/training_data_context_albert.txt', 'w') as file:
    for index in tqdm(range(len(training_data))):
        row = training_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['score_qa_albert'], int(row['docid'])))

with open('features_context/training_data_context_distilbert.txt', 'w') as file:
    for index in tqdm(range(len(training_data))):
        row = training_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['score_qa_distilbert'], int(row['docid'])))

validation_data = pd.read_csv('features_context/validation_data_NLP.csv', index_col=0)

with open('features_context/validation_data_context_albert.txt', 'w') as file:
    for index in tqdm(range(len(validation_data))):
        row = validation_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['score_qa_albert'], int(row['docid'])))

with open('features_context/validation_data_context_distilbert.txt', 'w') as file:
    for index in tqdm(range(len(validation_data))):
        row = validation_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['score_qa_distilbert'], int(row['docid'])))


testing_data = pd.read_csv('features_context/testing_data_NLP.csv', index_col=0)

with open('features_context/testing_data_context_albert.txt', 'w') as file:
    for index in tqdm(range(len(testing_data))):
        row = testing_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['score_qa_albert'], int(row['docid'])))

with open('features_context/testing_data_context_distilbert.txt', 'w') as file:
    for index in tqdm(range(len(testing_data))):
        row = testing_data.iloc[index]
        file.write('{} qid:{} 1:{} # docid = {} \n'.format(int(row['rating']), int(row['qid']), \
                                                                         row['score_qa_distilbert'], int(row['docid'])))
