import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
import librosa
import random
import pandas as pd
import argparse

import numpy as np
from bc_resnet import BCResNets

accuracy = evaluate.load("accuracy")

def compute_metrics(logits, label_ids):
    #print(logits)
    predictions = torch.argmax(logits, dim=-1)
    #print(predictions)
    #print(label_ids)
    return accuracy.compute(predictions=predictions, references=label_ids)

labels = ['background','speech']
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print(id2label, '\n\n', label2id)



def create_segments(wav_data, sr, segment_length, overlap):
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    step = segment_samples - overlap_samples
    segments = []
    
    for start in range(0, len(wav_data) - segment_samples + 1, step):
        segment = wav_data[start:start + segment_samples]
        segments.append(segment)
    
    return segments

# Step 3: Create a custom Dataset class for DataLoader
class WavDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx]

# Step 4: Main function to load the file, create segments, and return DataLoader
def create_dataloader(file_path, segment_length=0.42, overlap=0.1, batch_size=1, num_workers = 4, sr=None):
    wav_data, sr = librosa.load(file_path, sr)
    wav_data = wav_data.astype(np.float32)/32768
    segments = create_segments(wav_data, sr, segment_length, overlap)
    dataset = WavDataset(segments)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    return dataloader


def segments_time(filepath ,segment_length=0.42):
    segments = []
    with torch.no_grad():
        features = create_dataloader(filepath)
        for feature in features:
            logits = model(feature)
            logits = logits.numpy()
            pre = np.argmax(logits[0])
            preds = id2label[pre]
            segments.append(preds)
    start_time = None
    wav_data,_ = librosa.load(filepath, sr=16000)
    time_segments = []
    for idx, value in enumerate(segments):
        current_time = idx * segment_length
        
        if value == 1 and start_time is None:
            start_time = current_time
        elif value == 0 and start_time is not None:
            end_time = current_time
            time_segments.append([start_time, end_time])
            start_time = None

    if start_time is not None:
        time_segments.append([start_time, current_time + segment_length])
    
    if segments[-1] ==1:
        time_segments[-1][1] = len(wav_data)/16000
    return time_segments

class BC_ResNetModel(LightningModule):
    def __init__(self, width = 3):#width choices=[1, 1.5, 2, 3, 6, 8]
        super(BC_ResNetModel, self).__init__()
        self.bc_resnet = BCResNets(width*8)
    def forward(self,x):
        x = self.bc_resnet(x)
        return x

model = BC_ResNetModel.load_from_checkpoint('/VAD/bc_experiments/24-04/checkpoint_epoch=47-step=573792-val_loss=0.03823.ckpt')
model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the speech recognition engine in terminal.")
    parser.add_argument('--wav_file', type=str, default=None, required=False,
                        help='path of wav file')

    args = parser.parse_args()
    vad = segments_time(args.wav_file)
    print(vad)