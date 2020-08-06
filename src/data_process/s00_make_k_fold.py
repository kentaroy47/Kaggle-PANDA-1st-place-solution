import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


"""Decription
Make kfold csv from train.csv

Usage:
    $ python data_process/s00_make_k_fold.py
"""


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--kfold", type=int, default=5)
    arg("--seed", type=int, default=1222)
    arg("--train-df", type=str, default="../input/train.csv")
    arg(
        "--duplicated-imgids",
        type=str,
        default="../input/duplicate_imgids_imghash_thres_090.csv",
    )
    return parser.parse_args()


def main():
    args = make_parse()
    df = pd.read_csv(args.train_df)
    df_origin = df.copy()

    print(f"Split {args.kfold} kfold")

    # Extract duplicated imgids
    df_duplicated = pd.read_csv(args.duplicated_imgids)
    img_ids_excluded = set(df_duplicated[df_duplicated.index_in_group != 0].image_id)

    # Remove duplicated image ids
    df = df[~df.image_id.isin(img_ids_excluded)].copy().reset_index(drop=True)

    # Label encode
    label_unique = set(df.gleason_score.unique())
    label2id = dict(zip(label_unique, range(len(label_unique))))
    y_gleason = df.gleason_score.apply(lambda x: label2id[x]).values

    privider_unique = set(df.data_provider.unique())
    privider2id = dict(zip(privider_unique, range(len(privider_unique))))
    y_provider = df.data_provider.apply(lambda x: privider2id[x]).values

    y = np.stack([y_gleason, y_provider], axis=1)
    print(label2id)
    print(privider2id)
    print(y[:10])

    # Make kfold
    df["kfold"] = -1
    indxs = list(range(len(df)))

    mskf = MultilabelStratifiedKFold(
        n_splits=args.kfold, random_state=args.seed, shuffle=True
    )
    for i, (train_index, test_index) in enumerate(mskf.split(indxs, y)):
        df.loc[test_index, "kfold"] = i + 1

    print(df.head(10))

    # Fill duplicated image id to same group's kfold
    df_tmp = df[["image_id", "kfold"]].copy()
    df2 = pd.merge(df_origin, df_tmp, how="left", on="image_id").copy()
    df2.kfold = df2.kfold.fillna(-1)
    new_kfolds = list()

    for row in df2.itertuples():
        kfold_old = int(row.kfold)
        img_id = row.image_id
        if kfold_old == -1:
            group_id = (
                df_duplicated[df_duplicated.image_id == img_id]
                .copy()
                .group_id.values[0]
            )

            row_top_in_group = df_duplicated[
                (df_duplicated.group_id == group_id)
                & (df_duplicated.index_in_group == 0)
            ].copy()
            img_id_top = row_top_in_group.image_id.values[0]
            kfold_new = df[df.image_id == img_id_top].copy().kfold.values[0]
            new_kfolds.append(kfold_new)
        else:
            new_kfolds.append(kfold_old)
    df2 = df2.drop(columns="kfold")
    df2["kfold"] = new_kfolds
    print(df2.kfold.value_counts())
    df = df2.copy()

    # Save
    p_tmp = Path(args.train_df)
    new_path = Path(args.train_df).parent / (p_tmp.stem + f"-{args.kfold}kfold.csv")
    print(f"Save to: {new_path}")
    df.to_csv(new_path, index=False)


if __name__ == "__main__":
    main()
