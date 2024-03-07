#!/usr/bin/env python

import os
from tqdm import tqdm

val_files = ["20230306_163832_130", "20230320_161947_394"]
test_files = ["20230320_162454_268", "20230320_155248_635", "20230320_154606_631", "20230320_154400_119"]


def remove_val_test():
    # open train_files.txt and remove val and test files
    with open("train_files.txt", "r") as f:
        lines = f.readlines()

    with open("train_files.txt", "w") as f:
        for line in tqdm(lines):
            line = line.strip()
            if line.split("/")[0] not in val_files and line.split("/")[0] not in test_files:
                # print(f"[+] Writing {line}")
                f.write(line + "\n")


def keep_every_third_frame(filename):
    print(f"[+] Processing {filename}...")
    # open train_files.txt and remove val and test files
    with open(f"{filename}", "r") as f:
        lines = f.readlines()

    with open(f"{filename}", "w") as f:
        for line in tqdm(lines):
            line = line.strip()
            if int(line.split("/")[-1].split("_")[-1].split(".")[0]) % 3 == 0:
                # print(f"[+] Writing {line}")
                f.write(line + "\n")


if __name__ == "__main__":
    # remove_val_test()
    for fn in ['train_files.txt', 'val_files.txt', 'test_files.txt']:
        keep_every_third_frame(fn)
    # print(f"Currently no function is being called. Please uncomment the function you want to run.")
