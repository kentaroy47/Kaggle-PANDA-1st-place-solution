import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage.io
import tifffile
from skimage import morphology
from sklearn.cluster import KMeans
from tqdm import tqdm


def search_center_by_kmeans(thinned_img, num_clusters=16):
    # 座標が y, x 方向であることに注意
    xs = np.array(list(zip(*np.where(thinned_img > 0))))
    cluster = KMeans(n_clusters=num_clusters)
    _ = cluster.fit_predict(xs)
    centers = cluster.cluster_centers_
    return centers


def my_thinning(img, median=True, gray=False):
    if gray:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB2〜 でなく BGR2〜 を指定
    else:
        img2 = img.mean(axis=2)

    img3 = ((img2 < 220) * 255).astype(np.uint8)
    img4 = img3.copy()

    if median:
        img4 = cv2.medianBlur(img4, 5)
    kernel = np.ones((5, 5), np.uint8)
    img5 = cv2.erode(img4, kernel, iterations=1)
    img5 = cv2.dilate(img5, kernel, iterations=4)
    img5 = morphology.thin(img5 // 255)
    img5 = (img5 * 255).astype(np.uint8)
    return img5, (img2 < 250).sum()


def thinning_and_search_coords(img, resize_ratio=1, num_clusters=16):
    h, w, _ = img.shape
    if resize_ratio != 1:
        img2 = cv2.resize(img, (w // resize_ratio, h // resize_ratio))
    else:
        img2 = img.copy()

    img_thinned, mask_pixels = my_thinning(img2)

    # thin img
    if img_thinned.sum() < (255 * 100):
        resize_ratio = 1
        img_thinned, mask_pixels = my_thinning(img)

    # Like white img
    if img_thinned.sum() < 255 * num_clusters:
        print(img_thinned.sum())
        return None, mask_pixels * (resize_ratio ** 2)

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


def make_square_white(img):
    h, w, _ = img.shape
    white = [255, 255, 255]

    if h == w:
        return img

    top = max(0, (w - h) // 2)
    bottom = max(0, w - h - top)
    left = max(0, (h - w) // 2)
    right = max(0, h - w - left)

    img_tmp = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white)
    return img_tmp


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
    train = pd.read_csv(root / "train.csv")

    np.random.seed(1222)
    num_cluster = 16
    img_size = 256
    resize_ratio = 3

    # For high res img
    level = 1
    level_ratio = 4

    # TODO: mask
    out_dir = root / "panda-16x256x256-tiles-kmeans"
    out_dir.mkdir(exist_ok=True)
    out_train_zip = str(out_dir / "train.zip")

    x_tot, x2_tot = [], []
    img_ids_error = list()

    with zipfile.ZipFile(out_train_zip, "w") as img_out:
        for img_id in tqdm(train.image_id):
            img_path = img_dir / (img_id + ".tiff")
            img = skimage.io.MultiImage(str(img_path))[-1]

            centers_new, pixels = thinning_and_search_coords(
                img, resize_ratio=resize_ratio, num_clusters=num_cluster
            )

            if centers_new is None:
                print(img_id)
                img_ids_error.append(img_id)
                mask_dummy = np.ones(img.shape).astype(np.uint8)
                tiles = tile(img, mask_dummy)
            else:
                # Tuned at June 1st.
                box_size = ((pixels * 2.0 / num_cluster) ** 0.5) // 2 * 2
                box_size = max(40, box_size)
                tile_coords = coords_to_bbox(
                    centers_new,
                    img_width=img.shape[1],
                    img_height=img.shape[0],
                    box_width=box_size,
                    box_height=box_size,
                )
                tile_coords = [c * level_ratio for c in tile_coords]
                img_tiff = tifffile.imread(str(img_path), key=level)
                tiles = crop_imgmask_tile(img_tiff, coords=tile_coords)
            tiles = [
                {
                    "idx": d["idx"],
                    "img": cv2.resize(make_square_white(d["img"]), (img_size, img_size)),
                }
                for d in tiles
            ]

            for t in tiles:
                img, idx = t["img"], t["idx"]
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
                # if read with PIL RGB turns into BGR
                img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f"{img_id}_{idx}.png", img)

    # print error imgids
    print(img_ids_error)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))


if __name__ == "__main__":
    main()
