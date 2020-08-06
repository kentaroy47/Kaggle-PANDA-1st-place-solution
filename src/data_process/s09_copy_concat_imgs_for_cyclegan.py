import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


"""
Copy and concat images for CycleGAN
"""


def concat_imgs(tiles, tile_shape):
    image = cv2.hconcat([cv2.vconcat(tiles[ts]) for ts in tile_shape])
    return image


def main():
    root = Path("../input/")

    data_roots = [
        "numtile-64-tilesize-192-res-1-mode-0/train",
        "numtile-64-tilesize-192-res-1-mode-2/train",
    ]
    data_roots = [root / dr for dr in data_roots]

    out_dir = root / "tile-64-size-192-res-1-forGAN"
    out_dir.mkdir(exist_ok=True)
    print(f"out_dir: {out_dir}")

    train = pd.read_csv(root / "train.csv")
    image_ids = train.image_id.tolist()
    data_providers = train.data_provider.tolist()

    provider_to_output_dir = {"karolinska": "trainA", "radboud": "trainB"}
    for provider, dir_tmp in provider_to_output_dir.items():
        # (out_dir / dir_tmp).mkdir(exist_ok=True)
        for m in [0, 2]:
            (out_dir / f"test_{provider}_mode_{m}").mkdir(exist_ok=True)

    n_tiles = 64
    n = int(n_tiles ** 0.5)
    tile_shape = np.array(range(n_tiles)).reshape((n, n))
    for img_id, provider in tqdm(zip(image_ids, data_providers), total=len(train)):
        for i, dr in enumerate(data_roots):
            # Copy
            for idx in range(n_tiles):
                fname = f"{img_id}_{idx}.png"
                src_path = str(dr / fname)
                target_path = str(out_dir / f"test_{provider}_mode_{i*2}" / fname)
                shutil.copyfile(src_path, target_path)

            # # Read before imgs(tiles)
            # # BGR
            # tiles = np.array(
            #     [cv2.imread(str(dr / f"{img_id}_{idx}.png")) for idx in range(n_tiles)]
            # )
            #
            # # Concat imgs
            # img = concat_imgs(tiles, tile_shape=tile_shape)
            #
            # # Save new path
            # new_dir = out_dir / provider_to_output_dir[provider]
            # img_path_new = new_dir / f"{img_id}-mode-{i*2}.png"
            # cv2.imwrite(str(img_path_new), img)


if __name__ == "__main__":
    main()
