import os
import numpy as np
import pathlib
import pandas as pd
from collections import defaultdict, Counter
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
from nsml import DATASET_PATH

def make_folds(df, n_folds: int) -> pd.DataFrame:
    
    cls_counts = Counter([classes for classes in df['tag']])
    fold_cls_counts = defaultdict()
    for class_index in cls_counts.keys():
        fold_cls_counts[class_index] = np.zeros(n_folds, dtype=np.int)

    df['fold'] = -1
    pbar = tqdm.tqdm(total=len(df))

    def get_fold(row):
        class_index = row['tag']
        counts = fold_cls_counts[class_index]
        fold = np.argmin(counts)
        counts[fold] += 1
        fold_cls_counts[class_index] = counts
        row['fold']=fold
        pbar.update()
        return row
    
    df = df.apply(get_fold, axis=1)
    return df


image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
meta_data = pd.read_csv(meta_path, delimiter=',', header=0)

label_matrix = np.load(label_path)

b = []
for i in range(label_matrix.shape[0]):
    tag = np.argmax(label_matrix[i])
    b.append(tag)

tag_df = pd.DataFrame(b, columns=['tag'])
del meta_data['tags']
df = pd.concat([meta_data, tag_df], axis=1)
print(df.head())
print()

print("start making folds df")
folds = make_folds(df, n_folds=5)
print()

print(folds.head())
print()
fold0 = folds.loc[folds['fold'] != 0]
fold1 = folds.loc[folds['fold'] != 1]
fold2 = folds.loc[folds['fold'] != 2]
fold3 = folds.loc[folds['fold'] != 3]
fold4 = folds.loc[folds['fold'] != 4]
print()
print(len(fold0), len(fold1), len(fold2), len(fold3), len(fold4))





