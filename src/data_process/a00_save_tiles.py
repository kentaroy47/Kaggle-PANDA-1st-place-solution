#!/usr/bin/env python
# coding: utf-8

# In[1]:


kernel_type = 'resnet34'
modelname = kernel_type

fold = 0
tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 1
num_workers = 8
out_dim = 5
init_lr = 3e-4
warmup_factor = 10

kernel_type += "v2_tile{}_imsize{}".format(n_tiles, image_size)


# In[2]:


DEBUG = False


# In[3]:


import os
import sys


# In[4]:


import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm_notebook as tqdm
import torchvision


# # Config

# In[5]:


data_dir = '../input/'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'train')

warmup_epo = 1
n_epochs = 1 if DEBUG else 30
df_train = df_train.sample(100).reset_index(drop=True) if DEBUG else df_train

device = torch.device('cuda')

print(image_folder)


# In[6]:


df_train.head()


# # Create Folds

# In[7]:


def erase(df_train):
    df_train2 = df_train
    erase = []
    for i, id in enumerate(df_train2["image_id"].to_numpy()):
        if not os.path.isfile(os.path.join(image_folder, f'{id}.png')):
            erase.append(i)
            pass
        #img = cv2.imread(os.path.join(image_folder, f'{id}.png'))
        
    return df_train.drop(erase)

df_train = erase(df_train).reset_index()


# In[8]:


len(df_train)
df_train = df_train.drop("index", 1)


# In[9]:


skf = StratifiedKFold(5, shuffle=True, random_state=42)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
    df_train.loc[valid_idx, 'fold'] = i

df_train.tail()


# In[10]:


train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]


# In[11]:


transforms_val = albumentations.Compose([])


# # Dataset

# In[12]:


def get_tiles(img, mode=0, transform=None):
        result = []
        h, w, c = img.shape
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

        img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
        img3 = img2.reshape(
            img2.shape[0] // tile_size,
            tile_size,
            img2.shape[1] // tile_size,
            tile_size,
            3
        ).astype(np.float32)

        img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
        if len(img3) < n_tiles:
            img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
        idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
        img3 = img3[idxs]
        img3 = (img3*255).astype(np.uint8)
        for i in range(len(img3)):
            if transform is not None:
                img3[i] = transform(image=img3[i])['image']
            result.append({'img':img3[i], 'idx':i})
        return result, n_tiles_with_info >= n_tiles


class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                 show=False
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform
        self.show = show

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        if not os.path.isfile(os.path.join(image_folder, f'{img_id}.tiff')):
            pass
        
        # Load images as tiles
        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        
		times = 1
        for iii in range(times):
            tiles, OK = get_tiles(image, self.tile_mode, self.transform)

            if self.rand:
                idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
            else:
                idxes = list(range(self.n_tiles))

            # Concat tiles into a single image
            n_row_tiles = int(np.sqrt(self.n_tiles))
            images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
            for h in range(n_row_tiles):
                for w in range(n_row_tiles):
                    i = h * n_row_tiles + w

                    if len(tiles) > idxes[i]:
                        this_img = tiles[idxes[i]]['img']
                    else:
                        this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                    #this_img = 255 - this_img
                    h1 = h * image_size
                    w1 = w * image_size
                    images[h1:h1+image_size, w1:w1+image_size] = this_img

            # Make image augumentations
            #if self.transform is not None:
            #    images = self.transform(image=images)['image']
            images = images.astype(np.float32)
            images /= 255
            images = images.transpose(2, 0, 1)

            img = (images*255).astype("uint8").transpose(1, 2, 0) # [H,C,W] order.. why did I do this. this is brought back to [C,W,H] in train scripts..
            np.savez(os.path.join(data_dir, "train_{}_{}/{}".format(image_size, n_tiles, img_id, iii), img)) # we save by npz
        
        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label), img_id


# # Augmentations


import os
os.makedirs(os.path.join(data_dir, "train_{}_{}".format(image_size, n_tiles)), exist_ok=True)
import cv2
import matplotlib.pyplot as plt

dataset_show = PANDADataset(df_train, image_size, n_tiles, transform=None)
train_loader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size, num_workers=8)


# In[20]:


bar = tqdm(train_loader)
for (data, target, id) in bar:
    pass


# In[ ]:




