import pandas as pd

training_data_fasttext = pd.read_csv('features_word/training_data_NLP_FASTTEXT_160000.csv', index_col=0)
training_data_glove = pd.read_csv('features_word/training_data_NLP_GLOVE_240000.csv', index_col=0)
training_data = pd.read_csv('features_core/training_data.csv', index_col=0)
training_data_nonrelevant = pd.read_csv('features_core/training_data_nonrelevant.csv', index_col=0)
training_data_core = training_data.append(training_data_nonrelevant).reset_index()
training_data = training_data_core[['qid', 'docid', 'rating']]
training_data['FastText-COSINE'] = training_data_fasttext['FastText-COSINE']
training_data['Glove-COSINE'] = training_data_glove['Glove-COSINE']

print(training_data.head())
training_data.to_csv('training_data_word.csv')

testing_data_fasttext = pd.read_csv('features_word/testing_data_NLP_FASTTEXT_160000.csv', index_col=0)
testing_data_glove = pd.read_csv('features_word/testing_data_NLP_GLOVE_240000.csv', index_col=0)
testing_data = pd.read_csv('features_core/testing_data.csv', index_col=0)
testing_data_nonrelevant = pd.read_csv('features_core/testing_data_nonrelevant.csv', index_col=0)
testing_data_core = testing_data.append(testing_data_nonrelevant).reset_index()
testing_data = testing_data_core[['qid', 'docid', 'rating']]
testing_data['FastText-COSINE'] = testing_data_fasttext['FastText-COSINE']
testing_data['Glove-COSINE'] = testing_data_glove['Glove-COSINE']

print(testing_data.head())
testing_data.to_csv('testing_data_word.csv')

validation_data_fasttext = pd.read_csv('features_word/validation_data_NLP_FASTTEXT_160000.csv', index_col=0)
validation_data_glove = pd.read_csv('features_word/validation_data_NLP_GLOVE_240000.csv', index_col=0)
validation_data = pd.read_csv('features_core/validation_data.csv', index_col=0)
validation_data_nonrelevant = pd.read_csv('features_core/validation_data_nonrelevant.csv', index_col=0)
validation_data_core = validation_data.append(validation_data_nonrelevant).reset_index()
validation_data = validation_data_core[['qid', 'docid', 'rating']]
validation_data['FastText-COSINE'] = validation_data_fasttext['FastText-COSINE']
validation_data['Glove-COSINE'] = validation_data_glove['Glove-COSINE']

print(validation_data.head())
validation_data.to_csv('validation_data_word.csv')
