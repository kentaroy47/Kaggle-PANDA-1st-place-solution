import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NOT_GOOD_IMAGE_IDS = {
    "3790f55cad63053e956fb73027179707",  # all white
    # "040b2c98538ec7ead1cbd6daacdb3f64",  # radboud, small
    # "07a14fa5b8f74272e4cc0a439dbc8f7f",  # radboud, part of half...
    # "0da0915a236f2fc98b299d6fdefe7b8b",  # radboud, part of half...
    # "0e65f90aa2b3fd3e49982839aa381583",  # too small..., negative
    # "10c7898fb3c68c0e1b268ddb57590eb3",  # little small...
    # "",
}


def tile_augmentation(
    tiles,
    masks,
    random_tile: bool = False,
    random_rotate: bool = False,
    random_tile_white: bool = False,
):
    """
    tiles: (num_tile, h, w)
    masks: (num_tile, mask_ch, h, w)
    """
    # Tile augmentation
    if random_rotate and np.random.rand() < 0.5:
        indxs = np.random.randint(0, 4, len(tiles))
        for i, indx in enumerate(indxs):
            if indx == 3:
                # cv2.rotate(t, 0 or 1 or 2)
                continue
            tiles[i] = cv2.rotate(tiles[i], indx)
            if masks is not None:
                masks[i] = cv2.rotate(masks[i], indx)
                assert 1 == 2, "not implement !!"

    if random_tile and np.random.rand() < 0.5:
        indxs = np.random.permutation(len(tiles))
        tiles = tiles[indxs]
        if masks is not None:
            masks = masks[indxs]

    if random_tile_white and np.random.rand() < 0.5:
        indx = np.random.randint(len(tiles))
        tiles[indx][:] = 255  # White
        if masks is not None:
            masks[indx][:] = 0  # Black
    return tiles, masks


def get_spiral_shape(width: int, height: int) -> np.ndarray:
    assert width > 1 and height > 1
    north, south, west, east = (0, -1), (0, 1), (-1, 0), (1, 0)  # directions
    turn_left = {north: west, west: south, south: east, east: north}

    # Start near center
    x, y = width // 2 - 1, height // 2 - 1

    # initial direction
    d = south

    # nonvisit: -1
    matrix = np.zeros((height, width), dtype=np.int) - 1
    count = 0

    while count < (width * height):
        # visit
        matrix[y, x] = count

        # new place
        x, y = x + d[0], y + d[1]

        # judge next direction(straight or turn left)
        d_new = turn_left[d]
        x_new, y_new = x + d_new[0], y + d_new[1]
        if 0 <= x_new < width and 0 <= y_new < height and matrix[y_new, x_new] == -1:
            # enable_turn_left
            d = d_new
        count += 1
    return matrix


def concat_imgs(tiles, tile_shape):
    image = cv2.hconcat([cv2.vconcat(tiles[ts]) for ts in tile_shape])
    return image


def concat_imgs_ch(tiles, tile_shape):
    image = np.concatenate(tiles, axis=2)
    return image


def concat_tiles_with_augmentation(
    tiles,
    masks=None,
    tile_shape=None,
    mask_ch: int = 3,
    random_tile: bool = False,
    random_rotate: bool = False,
    random_tile_white: bool = False,
    concat_ch: bool = False,
):
    tiles, masks = tile_augmentation(
        tiles,
        masks,
        random_tile=random_tile,
        random_rotate=random_rotate,
        random_tile_white=random_tile_white,
    )

    # concat
    if concat_ch:
        concat_fn = concat_imgs_ch
    else:
        concat_fn = concat_imgs

    image = concat_fn(tiles=tiles, tile_shape=tile_shape)
    mask = None
    if masks is not None:
        masks_each_ch = [np.array([m[ch] for m in masks]) for ch in range(mask_ch)]
        mask = [concat_fn(tiles=m, tile_shape=tile_shape) for m in masks_each_ch]
    return image, mask


class TrainDataset(Dataset):
    def __init__(self, conf_dataset, phase, out_ch, transform):
        assert phase in {"train", "valid"}
        self.conf_dataset = conf_dataset
        self.phase = phase
        self.transform = transform
        self.out_ch = out_ch

        # Get from config
        self.img_size = conf_dataset.img_size
        self.tile_option = conf_dataset.tile_option
        self.num_tile = conf_dataset.num_tile
        self.tile_size = conf_dataset.tile_size
        self.kfold = conf_dataset.kfold
        self.df_path = conf_dataset.train_df
        self.mode_aug = conf_dataset.mode_aug
        self.spiral_tile = conf_dataset.spiral_tile
        self.nms = conf_dataset.nms_dataset
        self.mix_nms_normal_tile = conf_dataset.mix_nms_normal_tile
        self.concat_ch = conf_dataset.concat_ch
        self.cycle_gan_aug = conf_dataset.cycle_gan_aug
        self.softmax = conf_dataset.softmax
        self.use_clean_label = conf_dataset.use_clean_label
        self.clean_label_type = conf_dataset.clean_label_type  # pbc, pbnr, both
        assert self.clean_label_type in {"pbc", "pbnr", "both"}
        assert not (self.nms and self.mix_nms_normal_tile)

        # Tile augmentation
        if phase == "train":
            self.random_tile = conf_dataset.random_tile
            self.random_rotate_tile = conf_dataset.rotate_tile
            self.random_tile_white = conf_dataset.random_tile_white
        elif phase == "valid":
            self.random_tile = False
            self.random_rotate_tile = False
            self.random_tile_white = False

        # Tile shape
        n = int(self.num_tile ** 0.5)
        if self.spiral_tile:
            self.tile_shape = get_spiral_shape(n, n)
        else:
            self.tile_shape = np.array(range(self.num_tile)).reshape((n, n))

        # Adaptive generate
        print(f"df_path: {self.df_path}")
        self.df = pd.read_csv(self.df_path)
        if self.use_clean_label:
            print(f"Use clean label: {self.clean_label_type}")
            self.df = self.df[~self.df[f"is_error_cl_{self.clean_label_type}"]]

        if phase == "train":
            self.df = self.df[self.df["kfold"] != self.kfold].reset_index(drop=True)
        elif phase == "valid":
            self.df = self.df[self.df["kfold"] == self.kfold].reset_index(drop=True)

        # Remove not good img
        self.df = self.df[~self.df.image_id.isin(NOT_GOOD_IMAGE_IDS)].reset_index(drop=True)
        print(f"{phase} dataset: {len(self.df)}")
        if self.nms:
            data_dirs = [
                # ex. numtile-64-tilesize-160-nms-striderate-8
                f"../input/numtile-{self.num_tile}-tilesize-{self.tile_size}-nms-striderate-{sr}"
                for sr in [2, 8]
            ]
        elif self.mix_nms_normal_tile:
            data_dirs = [
                # f"../input/numtile-{self.num_tile}-tilesize-mixed-nms-normal-mode-{m}"
                # f"../input/numtile-{self.num_tile}-tilesize-mixed-nms-normal-mode-{m}"
                f"../input/numtile-{self.num_tile}-tilesize-{self.tile_size}-nms-striderate-{sr}"
                for sr in [2, 8]
                # for m in [0, 2]
            ]
        else:
            data_dirs = [
                f"../input/numtile-{self.num_tile}-tilesize-{self.tile_size}-res-1-mode-{m}"
                for m in [0, 2]
            ]
        self.img_dir = os.path.join(data_dirs[0], "train")
        self.img_dir_mode2 = os.path.join(data_dirs[1], "train")
        self.img_dir_gan = "../input/tile-64-size-192-res-1-forGAN"
        print(f"input dir: {self.img_dir}")
        if self.cycle_gan_aug:
            print(f"gan aug dir: {self.img_dir_gan}")

        self.labels = self.df["isup_grade"]
        self.df.gleason_score = self.df.gleason_score.map(lambda x: "0+0" if x == "negative" else x)
        self.gleason_first = [int(x.split("+")[0]) for x in self.df.gleason_score]
        self.gleason_second = [int(x.split("+")[1]) for x in self.df.gleason_score]
        self.data_provider = (self.df.data_provider == "karolinska").astype(np.float).values
        self.data_provider_raw = self.df.data_provider.values

    def __len__(self):
        return len(self.df)

    def make_label_softmax(self, idx):
        return self.labels[idx]

    def make_label(self, idx, out_ch):
        if self.softmax:
            return self.make_label_softmax(idx)

        if out_ch == 1:
            return torch.tensor(self.labels[idx]).float()

        label = np.zeros(out_ch).astype(np.float32)
        if out_ch == 2:
            # isup_grade, second gleason
            label[0] = self.labels[idx]
            label[1] = self.gleason_first[idx]
        elif out_ch == 3:
            # isup_grade, second gleason
            label[0] = self.labels[idx]
            label[1] = self.gleason_first[idx]
            label[2] = self.data_provider[idx]
        elif out_ch == 5:
            label[: self.labels[idx]] = 1.0
        elif out_ch == 10:
            label[: self.labels[idx]] = 1.0
            label[5 : 5 + self.gleason_first[idx]] = 1.0
        elif out_ch == 11:
            label[: self.labels[idx]] = 1.0
            label[5 : 5 + self.gleason_first[idx]] = 1.0
            label[10] = self.data_provider[idx]
        elif out_ch == 15:
            label[: self.labels[idx]] = 1.0
            label[5 : 5 + self.gleason_first[idx]] = 1.0
            label[10 : 10 + self.gleason_second[idx]] = 1.0
        elif out_ch == 16:
            label[: self.labels[idx]] = 1.0
            label[5 : 5 + self.gleason_first[idx]] = 1.0
            label[10 : 10 + self.gleason_second[idx]] = 1.0
            label[15] = self.data_provider[idx]
        return label

    def img_preprocess(self, img):
        # resize and normalization
        # TODO same img preprocess in kernel.py
        img = img.astype(np.float32)
        img = 255 - img  # this task imgs has many whites(255)
        img /= 255  # ToTensorV2 has no normalize !
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def get_tile_from_paths(self, paths):
        # BGR
        tiles = np.array([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths])
        return tiles

    def get_tiles_ganaug(self, idx):
        file_name = self.df["image_id"].values[idx]
        d = {"karolinska": "radboud", "radboud": "karolinska"}
        dp = self.data_provider_raw[idx]

        if self.mode_aug and np.random.rand() < 0.5:
            mode = 2
        else:
            mode = 0

        # ex.  fake_radboud_mode_0/0018ae58b01bdadc8e347995b69f99aa_0_fake.png
        base_path = os.path.join(self.img_dir_gan, f"fake_{d[dp]}_mode_{mode}")
        paths = [
            os.path.join(base_path, f"{file_name}_{idx}_fake.png") for idx in range(self.num_tile)
        ]
        return self.get_tile_from_paths(paths)

    def get_normal_tile(self, idx):
        file_name = self.df["image_id"].values[idx]
        paths = [
            os.path.join(self.img_dir, f"{file_name}_{idx}.png") for idx in range(self.num_tile)
        ]
        return self.get_tile_from_paths(paths)

    def get_tiles(self, idx):
        if self.phase == "valid":
            return self.get_normal_tile(idx)

        # TODO cyclegan aug only karolinska
        dp = self.data_provider_raw[idx]
        if self.cycle_gan_aug and dp == "karolinska" and np.random.rand() < 0.5:
            return self.get_tiles_ganaug(idx)

        if not (self.mode_aug and np.random.rand() < 0.5):
            return self.get_normal_tile(idx)

        # mode aug
        file_name = self.df["image_id"].values[idx]
        paths = [
            os.path.join(self.img_dir_mode2, f"{file_name}_{idx}.png")
            for idx in range(self.num_tile)
        ]
        return self.get_tile_from_paths(paths)

    def __getitem__(self, idx):
        tiles = self.get_tiles(idx)
        image, _ = concat_tiles_with_augmentation(
            tiles=tiles,
            tile_shape=self.tile_shape,
            random_tile=self.random_tile,
            random_rotate=self.random_rotate_tile,
            random_tile_white=self.random_tile_white,
            concat_ch=self.concat_ch,
        )

        # pre-process and augmentation
        image = self.img_preprocess(image)

        # HWC -> CHW
        augmented = self.transform(image=image)
        image = augmented["image"]

        label = self.make_label(idx, self.out_ch)
        return image, label, self.data_provider[idx]
