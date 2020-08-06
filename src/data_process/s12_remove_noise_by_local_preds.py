from pathlib import Path

import numpy as np
import pandas as pd


# Base arutema method
def remove_noisy(df, thresh):
    gap = np.abs(df["isup_grade"] - df["probs_raw"])
    df_removed = df[gap > thresh].reset_index(drop=True)
    df_keep = df[gap <= thresh].reset_index(drop=True)
    return df_keep, df_removed


def remove_noisy2(df, thresholds):
    print(f"  thresholds : {thresholds}")
    gap = np.abs(df["isup_grade"] - df["probs_raw"])

    df_keeps = list()
    df_removes = list()

    for label, thresh in enumerate(thresholds):
        df_tmp = df[df.isup_grade == label].reset_index(drop=True)
        gap_tmp = gap[df.isup_grade == label].reset_index(drop=True)

        df_remove_tmp = df_tmp[gap_tmp > thresh].reset_index(drop=True)
        df_keep_tmp = df_tmp[gap_tmp <= thresh].reset_index(drop=True)

        df_removes.append(df_remove_tmp)
        df_keeps.append(df_keep_tmp)

    df_keep = pd.concat(df_keeps, axis=0)
    df_removed = pd.concat(df_removes, axis=0)
    return df_keep, df_removed


def remove_noisy3(df, thresholds_rad, thresholds_ka):
    print(f"  threshold_rad: {thresholds_rad}")
    print(f"  threshold_ka : {thresholds_ka}")
    df_r = df[df.data_provider == "radboud"].reset_index(drop=True)
    df_k = df[df.data_provider != "radboud"].reset_index(drop=True)

    dfs = [df_r, df_k]
    thresholds = [thresholds_rad, thresholds_ka]
    df_keeps = list()
    df_removes = list()

    for df_tmp, thresholds_tmp in zip(dfs, thresholds):
        df_keep_tmp, df_remove_tmp = remove_noisy2(df_tmp, thresholds_tmp)
        df_keeps.append(df_keep_tmp)
        df_removes.append(df_remove_tmp)

    df_keep = pd.concat(df_keeps, axis=0)
    df_removed = pd.concat(df_removes, axis=0)
    return df_keep, df_removed


def show_keep_remove(df, df_removed):
    cuts = len(df_removed)
    cuts_rad = len(df_removed[df_removed.data_provider == "radboud"])
    cuts_ka = len(df_removed[df_removed.data_provider != "radboud"])
    print("  remove ratio[%]:", cuts / len(df) * 100)
    print("  number of reduced:", cuts)
    print("  number of reduced radboud :", cuts_rad)
    print("  number of reduced karolinska :", cuts_ka)
    print()


def main():
    # Read Local Predicts
    base_path = Path("../output/model/final_1")
    local_preds_paths = [
        base_path
        / f"local_preds_final_1_efficientnet-b1_kfold_{k}_latest_kfold_{k}.csv"
        for k in range(1, 6)
    ]
    for p in local_preds_paths:
        print(f"Read path: {p}")
    df = pd.concat([pd.read_csv(p) for p in local_preds_paths], axis=0)
    df.to_csv(base_path / "local_preds_final_1_efficientnet-b1.csv", index=False)

    # Base arutema method
    print("** Base arutema removing noise **")
    df_keep, df_remove = remove_noisy(df, thresh=1.6)
    show_keep_remove(df, df_remove)
    fname = "local_preds_final_1_efficientnet-b1_removed_noise_thresh_16.csv"
    df_keep.to_csv(base_path / fname, index=False)

    print("** fam_taro removing noise **")
    thresholds_rad = [1.3, 0.8, 0.8, 0.8, 0.8, 1.3]
    thresholds_ka = [1.5, 1.0, 1.0, 1.0, 1.0, 1.5]

    df_keep, df_removed = remove_noisy3(
        df, thresholds_rad=thresholds_rad, thresholds_ka=thresholds_ka
    )
    show_keep_remove(df, df_removed)
    fname = "local_preds_final_1_efficientnet-b1_removed_noise_thresh_rad_13_08_ka_15_10.csv"
    df_keep.to_csv(base_path / fname, index=False)


if __name__ == "__main__":
    main()
