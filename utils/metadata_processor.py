#!/usr/bin/env python

import os
import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import glob

rectified_dirs = glob.glob("/data2/raghav/datasets/Pixel4_3DP/rectified/*")

### WARNING: These functions will operate directly on the dataset ###

def clean(metadata):
    for dirBig in rectified_dirs:
        
        video_data_Dir = dirBig + "/Video_data/"
        total_frames = 0

        for videoDir in os.listdir(video_data_Dir):
            total_frames += len(os.listdir(video_data_Dir + videoDir))
            if videoDir in metadata["Video"].values:
                # get row where videoDir is present
                row = metadata.loc[metadata["Video"] == videoDir]
                # get the stopwatch column value
                removeFramesTill = int(row["stopwatch_or_hand_visible_till"].values[0])
                
                print("Inside:", dirBig.split('/')[-1], videoDir, removeFramesTill)
                ### TODO: Clean frames till removeFramesTill ###

                for frame in os.listdir(video_data_Dir + videoDir):
                    frameNum = int(frame.split('_')[1].split('.')[0])
                    if frameNum < removeFramesTill:
                        print(f"Removing {frame}")
                        os.remove(video_data_Dir + videoDir + "/" + frame)

        # print(f"Total frames in {dirBig.split('/')[-1]}: {total_frames}")
        

def process(metadata):
    pass


def augment(metadata):
    pass

###############################################################################

def get_metadata_from_csv(csv_path):
    metadata = pd.read_csv(csv_path)
    return metadata


if __name__ == '__main__':
    metadata_dir = "metadata/"
    stopwatchMeta = get_metadata_from_csv(metadata_dir + "metadata_stopwatch.csv")
    clean(stopwatchMeta)