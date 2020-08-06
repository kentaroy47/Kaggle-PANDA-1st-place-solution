import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage.io
from skimage import morphology
from sklearn.cluster import KMeans
from tqdm import tqdm


def thinning_img_new(img, median=False, resize=False):
    """
    No use opencv-contrib(for kernel competition)
    But little slow and results changed from Opencv
    TODO: optimizer when resize
    """
    height = img.shape[0]
    width = img.shape[1]
    img2 = img.copy()

    if resize:
        # For speed
        r = 3
        img2 = cv2.resize(img2, (width // r, height // r))

    img2 = img2.mean(axis=2)
    img3 = ((img2 < 220) * 255).astype(np.uint8)
    img4 = img3.copy()

    if median:
        img4 = cv2.medianBlur(img4, 5)
    kernel = np.ones((5, 5), np.uint8)
    img5 = cv2.erode(img4, kernel, iterations=1)
    img5 = cv2.dilate(img5, kernel, iterations=4)
    img5 = morphology.thin(img5 // 255)
    img5 = (img5 * 255).astype(np.uint8)

    if resize:
        img5 = cv2.resize(img5, (width, height))
        _, img5 = cv2.threshold(img5, 127, 255, cv2.THRESH_BINARY)
    return img5


def search_coords_by_kmeans(thinned_img, num_clusters=16, size=(128, 128)):
    # 座標が y, x 方向であることに注意
    xs = np.array(list(zip(*np.where(thinned_img > 0))))

    cluster = KMeans(n_clusters=num_clusters)
    _ = cluster.fit_predict(xs)
    height, width = thinned_img.shape

    # クラスタのセントロイド (重心) を描く
    centers = cluster.cluster_centers_

    # 各センターに対しクラスタ内で最も遠い点を探す
    # これうまくいけば順番に探せるのでは？ 最短経路問題として
    # 縦横でアスペクト比を変えるか変えないほうが良い場合がある
    # 長方形で切り取ってから正方形にするか、正方形で切り出すかの2種類ある
    # best r は マスクの総ピクセル数を割って作って良いのでは？
    # 画像って tiff から読みだした方が良いのでは？
    # TODO: must size (128, 128, 3)
    bboxes_coords = list()
    for i in range(num_clusters):
        c_y, c_x = centers[i]
        r1 = size[0] // 2
        r2 = size[1] // 2

        xmin, xmax = max(0, c_x - r1), min(width, c_x + r1)
        ymin, ymax = max(0, c_y - r2), min(height, c_y + r2)
        bboxes_coords.append(np.round([xmin, ymin, xmax, ymax]).astype(np.int))
    return bboxes_coords


def crop_imgmask_tile(img, mask=None, coords=None):
    tiles = list()
    for i, coord in enumerate(coords):
        xmin, ymin, xmax, ymax = coord
        img_tmp = img[ymin:ymax, xmin:xmax]
        mask_tmp = None
        if mask is not None:
            mask_tmp = mask[ymin:ymax, xmin:xmax]
        tiles.append({"img": img_tmp, "mask": mask_tmp, "idx": i})
    return tiles


def tile(img, mask, sz=128, num=16):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(
        img,
        [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
        constant_values=255,
    )
    mask = np.pad(
        mask,
        [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
        constant_values=0,
    )
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    mask = mask.reshape(mask.shape[0] // sz, sz, mask.shape[1] // sz, sz, 3)
    mask = mask.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < num:
        mask = np.pad(mask, [[0, num - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=0)
        img = np.pad(img, [[0, num - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({"img": img[i], "mask": mask[i], "idx": i})
    return result


def main():
    root = Path("../input/")
    img_dir = root / "train_images"
    mask_dir = root / "train_label_masks"

    train = pd.read_csv(root / "train.csv")
    np.random.seed(1222)

    num_cluster = 16

    out_dir = root / "panda-16x128x128-tiles-kmeans"
    out_dir.mkdir(exist_ok=True)
    out_train_zip = str(out_dir / "train.zip")
    out_masks_zip = str(out_dir / "mask.zip")

    x_tot, x2_tot = [], []

    img_ids_error = list()

    with zipfile.ZipFile(out_train_zip, "w") as img_out, zipfile.ZipFile(
        out_masks_zip, "w"
    ) as mask_out:
        for img_id in tqdm(train.image_id):
            img_path = img_dir / (img_id + ".tiff")
            mask_path = mask_dir / (img_id + "_mask.tiff")

            # Load img and mask
            img = skimage.io.MultiImage(str(img_path))[-1]
            mask = None
            if mask_path.exists():
                mask = skimage.io.MultiImage(str(mask_path))[-1]

            # Thinning
            # median にしないとスカスカのやつで引っかかる
            thinned_img = thinning_img_new(img, median=True, resize=True)
            if thinned_img.sum() < (255 * num_cluster):
                # 薄い場合は resize すると点が消える場合がある
                thinned_img = thinning_img_new(img, median=True, resize=False)
            if thinned_img.sum() >= (255 * num_cluster):
                coords = search_coords_by_kmeans(thinned_img, num_clusters=num_cluster)
                tiles = crop_imgmask_tile(img, mask, coords)
            else:
                print(img_id)
                img_ids_error.append(img_id)
                tiles = tile(img, mask)

            for t in tiles:
                img, mask, idx = t["img"], t["mask"], t["idx"]
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
                # if read with PIL RGB turns into BGR
                img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f"{img_id}_{idx}.png", img)

                if mask is not None:
                    mask = cv2.imencode(".png", mask[:, :, 0])[1]
                    mask_out.writestr(f"{img_id}_{idx}.png", mask)

    # print error imgids
    print(img_ids_error)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))


if __name__ == "__main__":
    main()
