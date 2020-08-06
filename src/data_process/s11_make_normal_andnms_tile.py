import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from tqdm import tqdm


def get_tiles(img, tile_size, n_tiles, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img3 = img2.reshape(
        img2.shape[0] // tile_size, tile_size, img2.shape[1] // tile_size, tile_size, 3
    )

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (
        img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255
    ).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(
            img3,
            [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]],
            constant_values=255,
        )
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({"img": img3[i], "idx": i})
    return result


def crop_imgmask_tile(img, coords, num_tile, tile_size):
    tiles = list()
    i = 0
    for coord in coords:
        xmin, ymin, xmax, ymax = coord
        img_tmp = img[ymin:ymax, xmin:xmax]
        tiles.append({"img": img_tmp, "idx": i})
        i += 1

    num_tile_tmp = len(tiles)
    if num_tile_tmp < num_tile:
        img_white = np.ones((tile_size, tile_size, 3))
        for j in range(num_tile_tmp, num_tile):
            tiles.append({"img": img_white.copy(), "idx": j})
    return tiles


def calc_coord_sum(img, tile_size, stride_rate, cuda=False):
    img_height, img_width, _ = img.shape

    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    img = torch.Tensor(img)
    img = 255 - img

    img = img.unsqueeze(0)
    in_ch = 3
    stride = tile_size // stride_rate

    conv2d = torch.nn.Conv2d(in_ch, 1, kernel_size=tile_size, stride=stride, bias=False)
    conv2d.weight = torch.nn.Parameter(torch.ones((1, in_ch, tile_size, tile_size)))
    if cuda:
        img = img.cuda()
        conv2d = conv2d.cuda()
        sums = conv2d(img).squeeze().cpu().detach().numpy()
    else:
        sums = conv2d(img).squeeze().detach().numpy()

    all_boxes = list()
    all_scores = list()

    height, width = sums.shape
    for h in range(height):
        for w in range(width):
            # xmin, ymin, xmax, ymax
            xmin = w * stride
            ymin = h * stride
            all_boxes.append(np.array([xmin, ymin, xmin + tile_size, ymin + tile_size]))
            all_scores.append(sums[h, w])
    return np.array(all_boxes), all_scores


def non_max_suppression(boxes, scores, max_boxes, overlap_thresh=0.01):
    # float 型に変換する。
    boxes = boxes.astype("float")
    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(scores)  # スコアを降順にソートしたインデックス一覧
    selected = []  # NMS により選択されたインデックス一覧
    selected_scores = []

    # indices がなくなるまでループする。
    while len(indices) > 0:
        # indices は降順にソートされているので、一番最後の要素の値 (インデックス) が
        # 残っている中で最もスコアが高い。
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)
        selected_scores.append(scores[selected_index])

        if len(selected) >= max_boxes:
            return boxes[selected].astype("int"), selected_scores

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        overlap = (i_w * i_h) / area[remaining_indices]

        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(
            indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    # 選択された短形の一覧を返す。
    return boxes[selected].astype("int"), selected_scores


def split_tile_nms(
    img, tile_size, stride_rate, n_tile, mode=0, overlap_thresh=0.01, cuda=False
):
    h, w, _ = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )

    # Split tile by NMS
    boxes, scores = calc_coord_sum(
        img, tile_size=tile_size, stride_rate=stride_rate, cuda=cuda
    )
    boxes_selected, scores_selected = non_max_suppression(
        boxes, scores, overlap_thresh=overlap_thresh, max_boxes=n_tile
    )
    return boxes_selected


def split_tile_nms_resize(
    img, tile_size, stride_rate, n_tile, mode=0, overlap_thresh=0.01, cuda=False
):
    r = 4
    h, w, _ = img.shape

    img_small = cv2.resize(img, (w // r, h // r))

    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)
    img = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    boxes_selected = split_tile_nms(
        img_small, tile_size // r, stride_rate, n_tile, mode, overlap_thresh, cuda=cuda
    )
    boxes_selected = [b * r for b in boxes_selected]
    tiles = crop_imgmask_tile(
        img, coords=boxes_selected, num_tile=n_tile, tile_size=tile_size
    )
    return tiles


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--tile-num", type=int, default=64)
    arg("--tile-size-k", type=int, default=192)
    arg("--tile-size-r", type=int, default=512)
    arg("--mode", type=int, default=0)
    return parser.parse_args()


def main():
    args = make_parse()
    root = Path("../input/")
    img_dir = root / "train_images"
    train = pd.read_csv(root / "train.csv")

    num_cluster = args.tile_num
    tile_size_k = args.tile_size_k
    tile_size_r = args.tile_size_r
    overlap_thresh = 0.05
    mode = args.mode
    cuda = True

    out_dir = root / f"numtile-{num_cluster}-tilesize-mixed-nms-normal-mode-{mode}"
    out_dir.mkdir(exist_ok=True)
    print(f"output: {out_dir}")
    out_train_zip = str(out_dir / "train.zip")
    img_ids_error = list()

    np.random.seed(1222)

    img_ids = train.image_id
    data_providers = train.data_provider

    with zipfile.ZipFile(out_train_zip, "w") as img_out:
        for img_id, dp in tqdm(zip(img_ids, data_providers), total=len(train)):
            assert dp in {"karolinska", "radboud"}

            img_path = str(img_dir / (img_id + ".tiff"))

            if dp == "karolinska":
                img = skimage.io.MultiImage(img_path)[1]
                tiles = get_tiles(img, tile_size_k, num_cluster, mode=mode)
            elif dp == "radboud":
                ts = tile_size_r
                # Read level 1 tiff
                img = skimage.io.MultiImage(img_path)[0]
                h, w, _ = img.shape

                tiles = split_tile_nms_resize(
                    img,
                    tile_size=ts,
                    stride_rate=8 if mode == 0 else 2,
                    n_tile=num_cluster,
                    mode=0,
                    overlap_thresh=overlap_thresh,
                    cuda=cuda,
                )

            assert len(tiles) == num_cluster

            for t in tiles:
                img, idx = t["img"], t["idx"]
                # if read with PIL RGB turns into BGR
                img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f"{img_id}_{idx}.png", img)

    # print error imgids
    print(img_ids_error)


if __name__ == "__main__":
    main()
