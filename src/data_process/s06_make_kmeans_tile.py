import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.io
from sklearn.cluster import KMeans
from tqdm import tqdm


"""
Make tile dir by adaptive

Usage:
    python data_process/s05_make_highres_tile_adaptive.py --num-tile 16 --box-scale 2
"""


def get_tiles(img, tile_size, n_tiles, mask, mode=0):
    t_sz = tile_size
    h, w, c = img.shape
    pad_h = (t_sz - h % t_sz) % t_sz + ((t_sz * mode) // 2)
    pad_w = (t_sz - w % t_sz) % t_sz + ((t_sz * mode) // 2)

    img2 = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img3 = img2.reshape(img2.shape[0] // t_sz, t_sz, img2.shape[1] // t_sz, t_sz, 3)
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    mask3 = None
    if mask is not None:
        mask2 = np.pad(
            mask,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=0,
        )
        mask3 = mask2.reshape(
            mask2.shape[0] // t_sz, t_sz, mask2.shape[1] // t_sz, t_sz, 3
        )
        mask3 = mask3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    n_tiles_with_info = (
        img3.reshape(img3.shape[0], -1).sum(1) < t_sz ** 2 * 3 * 255
    ).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(
            img3,
            [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]],
            constant_values=255,
        )
        if mask is not None:
            mask3 = np.pad(
                mask3,
                [[0, n_tiles - len(mask3)], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
            )
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    if mask is not None:
        mask3 = mask3[idxs]
    return img3, mask3, n_tiles_with_info >= n_tiles


def get_mask_from_img(img):
    img2 = img.mean(axis=2).copy()
    img3 = ((img2 < 240) * 255).astype(np.uint8)
    img4 = cv2.medianBlur(img3, 5)
    return img4, (img4 > 0).sum()


def search_center_by_kmeans(thinned_img, num_clusters):
    # 座標が y, x 方向であることに注意
    xs = np.array(list(zip(*np.where(thinned_img > 0))))
    cluster = KMeans(n_clusters=num_clusters)
    _ = cluster.fit_predict(xs)
    centers = cluster.cluster_centers_
    return centers


def search_coords_withoutthinning(img, resize_ratio, num_clusters):
    h, w, _ = img.shape
    img2 = cv2.resize(img, (w // resize_ratio, h // resize_ratio))
    img_thinned, mask_pixels = get_mask_from_img(img2)

    if mask_pixels < (num_clusters * 10):
        return None, None

    coords = search_center_by_kmeans(img_thinned, num_clusters=num_clusters)
    coords = [[tmp[1] * resize_ratio, tmp[0] * resize_ratio] for tmp in coords]
    return coords, mask_pixels * (resize_ratio ** 2)


def coords_to_bbox(coords, img_width, img_height, box_width, box_height):
    bboxes_coords = list()

    w = box_width // 2
    h = box_height // 2

    for center in coords:
        c_x, c_y = center
        xmin, xmax = max(0, c_x - w), min(img_width, c_x + w)
        ymin, ymax = max(0, c_y - h), min(img_height, c_y + h)
        bboxes_coords.append(np.round([xmin, ymin, xmax, ymax]).astype(np.int))
    return bboxes_coords


def crop_tile(img, mask, coords, tile_size):
    tile_imgs = list()
    tile_masks = None
    if mask is not None:
        tile_masks = list()
    sums = list()

    for i, coord in enumerate(coords):
        xmin, ymin, xmax, ymax = coord
        img_tmp = img[ymin:ymax, xmin:xmax]
        img_tmp = cv2.resize(make_square_white(img_tmp), (tile_size, tile_size))

        if mask is not None:
            mask_tmp = mask[ymin:ymax, xmin:xmax]
            mask_tmp = cv2.resize(make_square_black(mask_tmp), (tile_size, tile_size))
            tile_masks.append(mask_tmp)

        sums.append(img_tmp.sum())
        tile_imgs.append(img_tmp)

    indxs = np.argsort(sums)
    tile_imgs = np.array(tile_imgs)[indxs]
    if mask is not None:
        tile_masks = np.array(tile_masks)[indxs]
    return tile_imgs, tile_masks


def get_tile_kmeans(
    img_path: str, mask_path: str, resize_ratio: int, n_tiles: int, tile_size: int
):
    level_low = 2
    level_middle = 1
    level_high = 0
    ratio_high = 16

    img = skimage.io.MultiImage(img_path)[level_low]  # Low res
    mask = None

    centers_new, pixels = search_coords_withoutthinning(
        img, resize_ratio=resize_ratio, num_clusters=n_tiles
    )

    if centers_new is None:
        img_middle = skimage.io.MultiImage(img_path)[level_middle]
        if mask_path is not None:
            mask = skimage.io.MultiImage(mask_path)[level_middle]  # Low res
        tiles, masks, _ = get_tiles(
            img=img_middle, tile_size=tile_size, n_tiles=n_tiles, mask=mask, mode=0
        )
        return tiles, masks, None

    # Calc box size and tile coords
    box_size = ((pixels / n_tiles) ** 0.5) // 2 * 2
    tile_coords = coords_to_bbox(
        centers_new,
        img_width=img.shape[1],
        img_height=img.shape[0],
        box_width=box_size,
        box_height=box_size,
    )

    img_high = skimage.io.MultiImage(img_path)[level_high]
    if mask_path is not None:
        mask = skimage.io.MultiImage(mask_path)[level_high]  # Low res
    tile_coords_high = [coords * ratio_high for coords in tile_coords]
    tiles, masks = crop_tile(
        img=img_high, mask=mask, coords=tile_coords_high, tile_size=tile_size
    )
    return tiles, masks, tile_coords_high


def make_square_white(img):
    h, w, _ = img.shape
    if h == w:
        return img
    white = [255, 255, 255]

    top = max(0, (w - h) // 2)
    bottom = max(0, w - h - top)
    left = max(0, (h - w) // 2)
    right = max(0, h - w - left)

    img_tmp = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white
    )
    return img_tmp


def make_square_black(img):
    h, w, _ = img.shape
    if h == w:
        return img
    black = [0, 0, 0]

    top = max(0, (w - h) // 2)
    bottom = max(0, w - h - top)
    left = max(0, (h - w) // 2)
    right = max(0, h - w - left)

    img_tmp = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black
    )
    return img_tmp


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--num-tile", type=int, default=64)
    return parser.parse_args()


def main():
    args = make_parse()
    root = Path("../input/")
    img_dir = root / "train_images"
    mask_dir = root / "train_label_masks"
    train = pd.read_csv(root / "train.csv")

    np.random.seed(1222)
    num_tile = args.num_tile
    img_size = 256

    out_dir = root / f"kmeans-numtile-{num_tile}-res-0"
    out_dir.mkdir(exist_ok=True)
    out_train_zip = str(out_dir / "train.zip")
    out_mask_zip = str(out_dir / "mask.zip")

    x_tot, x2_tot = [], []

    with zipfile.ZipFile(out_train_zip, "w") as img_out, zipfile.ZipFile(
        out_mask_zip, "w"
    ) as mask_out:
        for img_id in tqdm(train.image_id):
            img_path = str(img_dir / (img_id + ".tiff"))
            mask_path = mask_dir / (img_id + "_mask.tiff")

            if mask_path.exists():
                mask_path = str(mask_path)
            else:
                mask_path = None

            # calc resize_ratio
            # num_cluster * 10 pixel ほしい
            _, pixels = get_mask_from_img(skimage.io.MultiImage(img_path)[2])
            resize_ratio_tmp = (pixels / (15 * num_tile)) ** 0.5
            resize_ratio_tmp = max(1, int(resize_ratio_tmp))

            # RGB
            tiles, masks, _ = get_tile_kmeans(
                img_path, mask_path, resize_ratio_tmp, num_tile, tile_size=img_size
            )

            for idx, img in enumerate(tiles):
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))

                # if read with PIL RGB turns into BGR
                img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f"{img_id}_{idx}.png", img)

                # mask[:, :, 0] has value in {0, 1, 2, 3, 4, 5}, other mask is 0 only
                if masks is not None:
                    mask = masks[idx]
                    mask = cv2.imencode(".png", mask[:, :, 0])[1]
                    mask_out.writestr(f"{img_id}_{idx}.png", mask)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))


if __name__ == "__main__":
    main()
