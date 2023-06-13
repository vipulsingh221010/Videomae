import numpy as np
import torch
import av
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import random
import shutil
from huggingface_hub import hf_hub_download
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(0)
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import csv
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
row1=[]
row2=[]
with open('labels_data.csv', 'r') as file:
    # Create a CSV reader object
   csv_reader = csv.reader(file)

   for row in csv_reader:
        row1.append(row[0])
        row2.append(row[1])

sim_score=[]
classes_similarityscore={}
num=1
for element1, element2 in zip(row1, row2):
    print(element1)
    embedding_1 = model1.encode(element1, convert_to_tensor=True)
    embedding_2 = model1.encode(element2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    if element1 in classes_similarityscore:
        classes_similarityscore[element1].append(similarity_score.item())
    else:
        classes_similarityscore[element1]=[similarity_score.item()]
    print(num)
    num+=1
precision=[]
recall=[]
f1=[]
accuracy=[]

for key in classes_similarityscore:
    simlist=classes_similarityscore[key]
    truth_labels= np.ones(len(simlist))
    precision.append(precision_score(truth_labels, [int(score >= 0.4) for score in simlist]))
    recall.append(recall_score(truth_labels, [int(score >= 0.4) for score in simlist]))
    f1.append(f1_score(truth_labels, [int(score >= 0.4) for score in simlist]))
    accuracy.append(accuracy_score(truth_labels, [int(score >= 0.4) for score in simlist]))



data = [precision, recall, f1, accuracy]

data_transposed = list(map(list, zip(*data)))

# Create a DataFrame from the transposed data
df = pd.DataFrame(data_transposed, columns=['Precision', 'Recall', 'F1-Score', 'Accuracy'])

# Save the results to ancsv file
df.to_csv('kinetic400_dataset_results.csv', index=False)


filename = 'similarity_scorekinetic400.csv'

# Extract the keys from the dictionary
keys = classes_similarityscore.keys()

# Determine the maximum length of the lists
max_length = max(len(lst) for lst in  classes_similarityscore.values())

# Create a list of lists, each representing a column
columns = [[ classes_similarityscore[key][i] if i < len( classes_similarityscore[key]) else '' for key in keys] for i in range(max_length)]

# Open the CSV file in write mode
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(keys)

    # Write the data rows
    writer.writerows(columns)
