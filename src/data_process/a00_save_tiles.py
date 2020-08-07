tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 1
num_workers = 8

import os
import sys
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm_notebook as tqdm

# # Config
data_dir = '../input/prostate-cancer-grade-assessment/' # where you place the train_images
out_dir = "../input/" # please place the tiles in input.
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'train_images')

print(image_folder)

# # Dataset
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
            np.savez_compressed(os.path.join(out_dir, "train_{}_{}/{}".format(image_size, n_tiles, img_id, iii)), img) # we save by npz
        
        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label), img_id


import os
os.makedirs(os.path.join(out_dir, "train_{}_{}".format(image_size, n_tiles)), exist_ok=True)
import cv2
import matplotlib.pyplot as plt

# declare dataloader
dataset_show = PANDADataset(df_train, image_size, n_tiles, transform=None)
train_loader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size, num_workers=num_workers)

# Generate npz files
bar = tqdm(train_loader)
for (data, target, id) in bar:
    pass
