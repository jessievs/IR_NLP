# NLP project

## MS Marco dataset

The MS Marco data needed to run this code is the collectionsandqueries file. Part of the files can be founded in this repo. Larger files need to be downloaded and added to the folder.

## Core features

In the previous project we created features for the MS Marco dataset. The CSVs corresponding to the created features can be found in the folder `features_core`. We will be creating new embedding features for the same query-passage pairs.

## Creating word embedding features

To create word embedding features, run `features_word.py`.

## Creating context embedding features

To create context embedding features, run `features_context.py`.

## Converting features to LETOR format

To convert the context embedding features and word embedding features to LETOR format `txt` files, run `csv_to_LETOR.py`.

## Creating L2R models

To create the L2R models for each of the embedding features:

### Word embedding - FastText

We investigated the following models and report their MAPs on the test data:

1. RankNet: 0.0217
2. RankBoost: 0.0168
3. AdaRank: 0.0217
4. Coordinate Ascent: 0.0217
5. LambdaMART: 0.0143
6. MART: 0.0168
7. ListNet: 0.0143
8. Random Forests: results in error

To create such a model, run the following (change the value after `-ranker` to the value corresponding to the desired model):

`java -jar RankLib-2.15.jar -train features_word/training_data_word_fasttext.txt -ranker 1 -gmax 1 -validate features_word/validation_data_word_fasttext.txt -test features_word/testing_data_word_fasttext.txt -metric2T MAP -save L2R_fasttext.txt`

We chose to create a RankNet model. The output is as follows:

```
Discard orig. features
Training data:	features_word/training_data_word_fasttext.txt
Test data:	features_word/testing_data_word_fasttext.txt
Validation data:	features_word/validation_data_word_fasttext.txt
Feature vector representation: Dense.
Ranking method:	RankNet
Feature description file:	Unspecified. All features will be used.
Train metric:	ERR@10
Test metric:	MAP
Highest relevance label (to compute ERR): 1
Feature normalization: No
Model file: L2R_fasttext.txt

[+] RankNet's Parameters:
No. of epochs: 100
No. of hidden layers: 1
No. of hidden nodes per layer: 10
Learning rate: 5.0E-5

...

Finished sucessfully.
ERR@10 on training data: 0.4967
ERR@10 on validation data: 0.4439
---------------------------------
MAP on test data: 0.0217
```

### Word embedding - GloVe

For GloVe we also create a RankNet model as follows:

`java -jar RankLib-2.15.jar -train features_word/training_data_word_glove.txt -ranker 1 -gmax 1 -validate features_word/validation_data_word_glove.txt -test features_word/testing_data_word_glove.txt -metric2T MAP -save L2R_glove.txt`

The output is as follows:

```
Discard orig. features
Training data:	features_word/training_data_word_glove.txt
Test data:	features_word/testing_data_word_glove.txt
Validation data:	features_word/validation_data_word_glove.txt
Feature vector representation: Dense.
Ranking method:	RankNet
Feature description file:	Unspecified. All features will be used.
Train metric:	ERR@10
Test metric:	MAP
Highest relevance label (to compute ERR): 1
Feature normalization: No
Model file: L2R_glove.txt

[+] RankNet's Parameters:
No. of epochs: 100
No. of hidden layers: 1
No. of hidden nodes per layer: 10
Learning rate: 5.0E-5

...

Finished sucessfully.
ERR@10 on training data: 0.4967
ERR@10 on validation data: 0.4439
---------------------------------
MAP on test data: 0.0211
```

### Context embedding - DistilBERT

### Context embedding - AlBERT

### Word embedding + core - FastText

We create a random forest model as follows:

`java -jar RankLib-2.15.jar -train features_word/training_data_word_fasttext_core.txt -ranker 8 -gmax 1 -validate features_word/validation_data_word_fasttext_core.txt -test features_word/testing_data_word_fasttext_core.txt -metric2T MAP -save L2R_fasttext_core.txt`

The output is as follows:

```
Discard orig. features
Training data:	features_word/training_data_word_fasttext_core.txt
Test data:	features_word/testing_data_word_fasttext_core.txt
Validation data:	features_word/validation_data_word_fasttext_core.txt
Feature vector representation: Dense.
Ranking method:	Random Forests
Feature description file:	Unspecified. All features will be used.
Train metric:	ERR@10
Test metric:	MAP
Highest relevance label (to compute ERR): 1
Feature normalization: No
Model file: L2R_fasttext_core.txt

[+] Random Forests's Parameters:
No. of bags: 300
Sub-sampling: 1.0
Feature-sampling: 0.3
No. of trees: 1
No. of leaves: 100
No. of threshold candidates: 256
Learning rate: 0.1

...

------------------------------------
Finished sucessfully.
ERR@10 on training data: 0.4967
ERR@10 on validation data: 0.4439
------------------------------------
MAP on test data: 0.4963
```

### Word embedding + core - GloVe

We create a random forest model as follows:

`java -jar RankLib-2.15.jar -train features_word/training_data_word_glove_core.txt -ranker 8 -gmax 1 -validate features_word/validation_data_word_glove_core.txt -test features_word/testing_data_word_glove_core.txt -metric2T MAP -save L2R_glove_core.txt`

The output is as follows:

```

```
