import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import wandb
import librosa
import random
import pandas as pd
import evaluate
import numpy as np
from audiomentations import TimeMask, TimeStretch

from model import BCResNets

accuracy = evaluate.load("accuracy")

def compute_metrics(logits, label_ids):

    predictions = torch.argmax(logits, dim=-1)
    return accuracy.compute(predictions=predictions, references=label_ids)

labels = ['background','speech']
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print(id2label, '\n\n', label2id)

def data_processing(file_path,sample_rate,segment_size):
    wav_form, _ = librosa.load(file_path,sr = sample_rate,mono=True)
    segment = segment_size*sample_rate
    if len(wav_form)-segment > 0:
        n = random.randint(0,int(len(wav_form)-segment))
        wav_form = wav_form[int(n):int(n+segment)]
    i = random.randint(0,2)
    if i==1:
        transform = TimeMask(min_band_part=0.2, max_band_part=0.3,p=1.0)
        wav_form = transform(wav_form, sample_rate=16000)
    if i==2: 
        transform = TimeStretch(min_rate=0.5, max_rate=1.5, p=1.0)
        wav_form = transform(wav_form, sample_rate=16000)
    wav_form = wav_form.astype(np.float32)/32768
    return wav_form

def padding(batch):
    wavforms = []
    labels = []
    for wavform, label in batch:
        wavforms.append(torch.tensor(wavform))
        labels.append(torch.tensor(label))
    wavforms = nn.utils.rnn.pad_sequence(wavforms, batch_first=True,padding_value=0)
    return wavforms, labels

class CustomDataSet(Dataset):
    def __init__(self, csv_file,sample_rate,segment_size, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.segment_size = segment_size
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        file_path = self.df.file_path[index]
        label = label2id[self.df.label[index]]
  
        segment = data_processing(file_path,self.sample_rate,self.segment_size)
        return segment, label

class BC_ResNetModel(LightningModule):
    def __init__(self, width = 3):#width choices=[1, 1.5, 2, 3, 6, 8]
        super(BC_ResNetModel, self).__init__()
        self.bc_resnet = BCResNets(width*8)
        self.csv_train = 'train.csv'
        self.csv_val = 'valid.csv'
        self.csv_test = 'test.csv'
        self.sample_rate = 16000
        self.num_workers = 4
        self.frame_size = 0.42
        #self.learning_rate = 0.0005,
        self.lr = 0.0001,

    def forward(self,x):
        x = self.bc_resnet(x)
        return x
    
    def training_step(self,batch,batch_idx):
        loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self,batch,batch_idx):
        loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self,batch,batch_idx):
         loss, acc = self._get_preds_loss_accuracy(batch)
         self.log("test_loss", loss, prog_bar=True)
         self.log("test_acc", acc, prog_bar=True)
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        wavform, labels = batch
        pred = self(wavform)
        labels = torch.tensor(labels).cuda()
        loss = F.cross_entropy(pred,labels)
        acc = compute_metrics(pred,labels)
        return loss, acc
    # def configure_optimizers(self):
    #     """
    #     Return whatever optimizers and learning rate schedulers you want here.
    #     At least one optimizer is required.
    #     """
    #     optimizer = optim.AdamW(self.parameters(), lr=list(self.learning_rate).pop(0)/10)
    #     lr_scheduler = {'scheduler':optim.lr_scheduler.CyclicLR(optimizer,base_lr=list(self.base_learning_rate).pop(0),max_lr=list(self.learning_rate).pop(0),step_size_up=2000,cycle_momentum=False),}


    #     return [optimizer], [lr_scheduler]
    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=list(self.lr).pop(0))
    
    def prepare_data(self):
        pass
    
    def setup(self,stage):
        self.dataset_train = CustomDataSet(self.csv_train, self.sample_rate, self.frame_size)
        self.dataset_val = CustomDataSet(self.csv_val, self.sample_rate, self.frame_size)
        self.dataset_test = CustomDataSet(self.csv_test, self.sample_rate, self.frame_size)
    
    def train_dataloader(self):
        return DataLoader(
            dataset = self.dataset_train,
            batch_size = 32,
            shuffle = False,
            collate_fn = lambda x: padding(x),
            num_workers = self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset = self.dataset_val,
            batch_size = 32,
            shuffle = False,
            collate_fn = lambda x: padding(x),
            num_workers = self.num_workers,
        )

    def test_dataloader(self):
         return DataLoader(
             dataset = self.dataset_test,
             batch_size = 32,
             shuffle = False,
             collate_fn = lambda x: padding(x),
             num_workers = self.num_workers,
         )

#train model
import warnings
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='VAD', # group runs in "MNIST" project
                            name="exp1",
                            log_model='all') # log all new checkpoints during training



checkpoint_callback = ModelCheckpoint(dirpath = 'bc_experiment/24-04',
                                      filename= 'checkpoint_{epoch:02d}-{step:02.0f}-{val_loss:02.5f}',
                                      auto_insert_metric_name = True,
                                      save_top_k = 3,
                                      monitor='val_loss', 
                                      mode='min',)

model = BC_ResNetModel()
trainer = Trainer(max_epochs=50, devices=1,logger = wandb_logger, accelerator="gpu", callbacks = [checkpoint_callback])
trainer.fit(model)
wandb.finish()
