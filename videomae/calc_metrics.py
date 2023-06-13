
import numpy as np
import torch
import av
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
torch.set_num_threads(32)
import time

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices




test_set_dir ="kinetic_dataset"
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-huge-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-huge-finetuned-kinetics")

ground_truth_labels = {}


predicted_label=[]
ground_truth_label=[]
x=1
accuracy=[]
precision=[]
recall=[]
f1=[]
for class_folder in os.listdir(test_set_dir):
    class_folder_path=os.path.join(test_set_dir,class_folder)
    predicted_list = []
    num=1
    for video_file in os.listdir(class_folder_path):

        print(x)
        x+=1
        num+=1
        if num==16:
            break
        print(time.perf_counter())
        video_name=os.path.splitext(video_file)[0]
        ground_truth_labels[video_name]=class_folder
        video_path = os.path.join(class_folder_path,video_file)


        try:
            container = av.open(video_path)

            indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container, indices)
            video1 = np.transpose(video, (0, 3, 1, 2))

            inputs = processor(list(video1), return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()

            word1 =model.config.id2label[predicted_class_idx]
            word2 =ground_truth_labels[video_name]
            predicted_label.append(word1)
            ground_truth_label.append(word2)
            if word1==word2:
              predicted_list.append(1)
            else:
                predicted_list.append(0)
        
        except Exception as e:
            print(f"error processing video'{video_file}':{str(e)}")
            continue  
    truth_list = np.ones(len(predicted_list))

    precision.append(precision_score(truth_list, predicted_list))
    recall.append(recall_score(truth_list, predicted_list))
    f1.append(f1_score(truth_list, predicted_list))
    accuracy.append(accuracy_score(truth_list, predicted_list))



data = [precision, recall, f1, accuracy]
print(data)
# Transpose the data
data_transposed = list(map(list, zip(*data)))

# Create a DataFrame from the transposed data
df = pd.DataFrame(data_transposed, columns=['Precision', 'Recall', 'F1-Score', 'Accuracy'])

# Save the results to an Excel file
df.to_csv('kinetic_new_results.csv', index=False)

labels=[ground_truth_label,predicted_label]
labels_transpose=list(map(list, zip(*labels)))
labels_data=pd.DataFrame(labels_transpose,columns=['truth_class','Predicted_class'])
labels_data.to_csv('kinetic_new_labels_data.csv',index=False)

