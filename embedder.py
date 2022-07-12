from typing import Literal
from numpy import dot
from numpy.linalg import norm
import pandas as pd


# FIRST EXECUTE TWO COMMANDS BELOW:
# analyzing to filter the more pure recordings:
# python .\BirdNET-Analyzer-main\analyze.py --i wav_files --o analyzed --min_conf 0.5 --threads 6

# generate embeddings:
# python .\BirdNET-Analyzer-main\embeddings.py --i wav_files --o embeddings --threads 6


def filter_fragment(id, species):
    try:
        file_analysis = pd.read_csv('analyzed\\' + str(id) + '.BirdNET.selection.table.txt', sep="\t")
        embeddings = pd.read_csv('embeddings\\' + str(id) + '.birdnet.embeddings.txt', sep="\t", header=None)
    except:
        print("Fragment " + str(id) + " not processed.")
        return

    for i, row in embeddings.iterrows():
        start_time = row[0]
        predictions = file_analysis[file_analysis['Begin Time (s)'] == start_time]
        
        if len(predictions) == 1 and predictions.iloc[0]['Common Name'] == species:  
            selection.write(species + "\t" + str(id) + "\t" + str(start_time) + "\t" + row[2] + "\n")


selection = open("embeddings.txt", "w+")

recordings = pd.read_csv('filtered_recs.txt', sep="\t", encoding="latin-1")
for i, row in recordings.iterrows():
    filter_fragment(row['id'], row['name'])

selection.close()
