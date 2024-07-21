import pandas as pd
import os
import glob
from tqdm import tqdm

#create csv file for train and test data
if os.path.exists('train_dataset.csv'):
    print('file test_dataset.csv already exists. Skipping...')
    df_train = pd.read_csv('train_dataset.csv')
else:
    df_train = pd.DataFrame(columns=["file_path","label"])
    folder = "/VAD/vad_data" #replace your data folder
# Note that the code below is for my dataset structure.
# Please edit the code to match your dataset structure or change your dataset structure 
    i=0
    for sub_folder in tqdm(os.listdir(folder)):
        for sub_sub_folder in os.listdir(os.path.join(folder,sub_folder)):
            for sub_sub_sub_folder in os.listdir(os.path.join(folder,sub_folder,sub_sub_folder)):
                for file_name in glob.glob(os.path.join(folder,sub_folder,sub_sub_folder,sub_sub_sub_folder, "*.wav")):
                    file_path = os.path.join(folder,sub_folder,sub_sub_folder,sub_sub_sub_folder,file_name)
                    df_train.loc[i] = [file_path,sub_folder]
                    i += 1
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_train.to_csv('train_dataset.csv')

# if os.path.exists('test.csv'):
#     print('file test.csv already exists. Skipping...')
#     df_test = pd.read_csv('test.csv')
# else:
#     df_test = pd.DataFrame(columns=["file_path","label"])
#     folder = "/VAD/vad/audio_test"
#     i=0
#     for sub_folder in tqdm(os.listdir(folder)):
#         for sub_sub_folder in os.listdir(os.path.join(folder,sub_folder)):
#             for sub_sub_sub_folder in os.listdir(os.path.join(folder,sub_folder,sub_sub_folder)):
#                 for file_name in glob.glob(os.path.join(folder,sub_folder,sub_sub_folder,sub_sub_sub_folder, "*.wav")):
#                     file_path = os.path.join(folder,sub_folder,sub_sub_folder,sub_sub_sub_folder,file_name)
#                     df_test.loc[i] = [file_path,sub_folder]
#                     i += 1
#     df_test = df_test.sample(frac=1).reset_index(drop=True)
#     df_test.to_csv('test.csv')

print(df_train['label'].value_counts())
#print(df_test['label'].value_counts())

#balance training data using oversampling method
import numpy as np

X = df_train['file_path']
y = df_train['label']

minority_samples = X[y == 'background']

# balance datasets
num_samples_to_generate = len(X) - 2*len(minority_samples)
new_minority_samples = np.random.choice(minority_samples, num_samples_to_generate,replace=True)

X_balanced = pd.concat([X, pd.Series(new_minority_samples)], ignore_index=True)
y_balanced = pd.concat([y, pd.Series(['background'] * num_samples_to_generate)], ignore_index=True)


data_resampled = pd.DataFrame({'file_path':X_balanced})
data_resampled['label'] = y_balanced

data_resampled.to_csv('balanced_train_dataset.csv', index=False)
data_resampled = data_resampled.sample(frac=1).reset_index(drop=True)
print(data_resampled['label'].value_counts())
#split to train and valid file
import math
train_num = math.ceil(len(data_resampled['label'])*0.95)
data_resampled_train = data_resampled[:train_num]
data_resampled_val = data_resampled[train_num:]
data_resampled_train.to_csv('train.csv')
data_resampled_val.to_csv('valid.csv')