#!/usr/bin/env python

import os
import glob
from tqdm.auto import tqdm
# TODO: Generate txt file for full dataset

videoPaths = glob.glob("/data2/raghav/datasets/Pixel4_3DP/rectified/B/Video_data/*")
unrectified_videoPaths = "/data2/raghav/datasets/Pixel4_3DP/unrectified/B/dp_data"

val_files = ["20230306_163832_130", "20230320_161947_394"]
test_files = os.listdir("/data2/raghav/datasets/Pixel4_3DP/rectified/B/Video_data/")


def gen_train(val_files, test_files):
    with open("train_files.txt", "w+") as f:
        for videoDir in tqdm(videoPaths):
            videoName = videoDir.split("/")[-1]
            # print(videoName)
            # Check extension of files in unrectified folder
            if videoName in val_files or videoName in test_files:
                continue

            urect_vid = sorted(glob.glob(os.path.join(unrectified_videoPaths, videoName) + "/*"))
            # print(urect_vid)
            fileCheckName =  urect_vid[0].split("/")[-1]
            if "right" in urect_vid[0] or "left" in urect_vid[0]:
                for img in sorted(os.listdir(videoDir)):
                    if img.endswith(".jpg"):
                        img_name = img.split(".")[0]
                        img_number = int(img_name.split("_")[-1])
                        f.write(f"{videoName}/{img};{videoName}/{img_name}_left.jpg;{videoName}/{img_name}_right.jpg;{videoName}/{img}\n")


def gen_val_or_test(split_name, files_lst):

    with open(f"{split_name}.txt", "w+") as f:
        for videoDir in tqdm(videoPaths):
            videoName = videoDir.split("/")[-1]
            
            if videoName not in files_lst:
                continue

            urect_vid = sorted(glob.glob(os.path.join(unrectified_videoPaths, videoName) + "/*"))
            # fileCheckName =  urect_vid[0].split("/")[-1]
            
            if "right" in urect_vid[0] or "left" in urect_vid[0]:
                for img in sorted(os.listdir(videoDir)):
                    if img.endswith(".jpg"):
                        img_name = img.split(".")[0]
                        img_number = int(img_name.split("_")[-1])
                        f.write(f"{videoName}/{img};{videoName}/{img_name}_left.jpg;{videoName}/{img_name}_right.jpg;{videoName}/{img}\n")


def count_files(filename):
    train_len = {}
    total_train = 0
    with open(f"{filename}", "r") as f:
        for line in f:
            video_name = line.split("/")[0]
            if video_name in train_len:
                train_len[video_name] += 1
            else:
                train_len[video_name] = 1
            total_train += 1

    return train_len, total_train


def get_dataset_split_lengths():
    train_len, total_train = count_files("train_files.txt")
    val_len, total_val = count_files("val_files.txt")
    test_len, total_test = count_files("test_files.txt") 

    import pandas as pd
    train_df = pd.DataFrame(train_len.items(), columns=["video_name", "train_len"])
    val_df = pd.DataFrame(val_len.items(), columns=["video_name", "val_len"])
    test_df = pd.DataFrame(test_len.items(), columns=["video_name", "test_len"])

    # save dataframe
    train_df.to_csv("train.csv", index=False) # will add to markdown later
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"Total:\nTrain: {total_train} | Val: {total_val} | Test: {total_test}\n")
    print("---------------------------------------------------------------------------")
    print(f"Train (head):\n {train_df.head()}\n\nVal (all):\n {val_df}\n\nTest (all):\n {test_df}")
    print("-------------------------------------x-------------------------------------")
    


if __name__ == "__main__":
    # gen_train(val_files, test_files)
    # gen_val_or_test("val_files", val_files)
    gen_val_or_test("test_files", test_files)
    # get_dataset_split_lengths()
    # print(f"Currently no function is being called. Please uncomment the function you want to run.")