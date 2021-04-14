import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def process(modelname):
    # Get data
    filename = 'evaluations/{}.txt'.format(modelname)
    evaluations = pd.read_csv(filename, names=['metric', 'qid', 'value'], delim_whitespace=True)

    # Ignored combined results
    evaluations_q = evaluations[~evaluations['qid'].str.contains('all')]

    # Select MAP metric
    map = evaluations_q[(evaluations_q['metric'] == 'map')]

    # Convert to numeric
    map[['qid', 'value']] = map[['qid', 'value']].apply(pd.to_numeric, errors='ignore')

    # Return data with column indicating model name
    map = map.assign(modelname=modelname)
    return map

# Saving model names
modelnames = ['albert', 'albert_core', 'distilbert', 'distilbert_core', 'fasttext', 'fasttext_core', 'glove', 'glove_core', 'core']

# Combining data
combined = pd.DataFrame([], columns=['metric', 'qid', 'value', 'modelname'])
for modelname in modelnames:
    map_data = process(modelname)
    combined = combined.append(map_data)

# Loading query data
queries = pd.read_csv('collections/msmarco-passage/msmarco-test2019-queries.tsv', sep = '\t', names=['qid', 'query'])

# Merging query data and evaluation data
merged = combined.merge(queries, on='qid')

# Plotting distributions over MAP scores for each model
sns.displot(x='value', hue='modelname', data=merged, kind="kde")
plt.xlabel('MAP')
plt.ylabel('Frequency')
plt.savefig('error_analysis/distribution_map_combined', bbox_inches = "tight")

# Plotting markers to indicate MAP for each unique query
plt.figure(figsize=(7, 15))
sns.stripplot(x='value', y='query', data=merged, hue='modelname')
for idx in range(len(merged.qid.unique())):
    plt.axhline(y=idx, color='grey', alpha=0.5)
plt.xlabel('MAP')
plt.ylabel('')
plt.savefig('error_analysis/all_models_scatter', bbox_inches = "tight")

# Save data to csv
merged.to_csv('error_analysis/merged_data.csv')
