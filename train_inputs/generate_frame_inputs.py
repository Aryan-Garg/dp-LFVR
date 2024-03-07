#!/usr/bin/env python3
import os
import json
import random

def generate_metadata_file(
    videos, skip, rgb_path, rectified_base_path, metadata_file_path
):
    def get_dp_path_from_rgb(rgb_frame):
        base_name = rgb_frame.split(".")[0]
        left_name = base_name + "_left.jpg"
        right_name = base_name + "_right.jpg"
        return left_name, right_name
    frames = []
    for video_name in videos:
        video_name_beginning = video_name.split("_")[0]
        if video_name_beginning == "20230212" or video_name_beginning == "20230301":
            continue
        frames_B = sorted(
            os.listdir(os.path.join(rectified_base_path, "B", rgb_path, video_name))
        )
        # frames_A = sorted(
        #     os.listdir(os.path.join(rectified_base_path, "A", rgb_path, video_name))
        # )
        # frames_C = sorted(
        #     os.listdir(os.path.join(rectified_base_path, "C", rgb_path, video_name))
        # )
        for i in range(0, len(frames_B), skip):
            left_dpB, right_dpB = get_dp_path_from_rgb(frames_B[i])
            frames.append((
                os.path.join(video_name, frames_B[i]),
                os.path.join(video_name, left_dpB),
                os.path.join(video_name, right_dpB),
                os.path.join(video_name, frames_B[i]),
            ))
    with open(metadata_file_path, 'w') as f:
        for frame in frames:
            f.write(f"{frame[0]};{frame[1]};{frame[2]};{frame[3]}\n")
        
        
if __name__ == "__main__":
    rectified_base_path = "/data2/raghav/datasets/Pixel4_3DP/rectified/"
    rgb_path = "Video_data/"
    dp_path = "dp_data/"
    dpt_path = "dptdepth/"

    # TODO: try a higher skip once you have more inputs
    skip = 100
    # metadata_file_name = f"dpvideos_skip{skip}.json"

    videos = os.listdir(os.path.join(rectified_base_path, "B", rgb_path))
    print("Number of videos:", len(videos))
    random.Random(42).shuffle(videos) # Setting seed as 42

    train_videos = videos[: int(0.8 * len(videos))]
    val_videos = videos[int(0.8 * len(videos)) : int(0.9 * len(videos))]
    test_videos = videos[int(0.9 * len(videos)) :]

    os.makedirs(f"./Pixel4_3DP_frame_skip{skip}", exist_ok=True)
    generate_metadata_file(
        train_videos,
        skip,
        rgb_path=rgb_path,
        rectified_base_path=rectified_base_path,
        metadata_file_path=f"./Pixel4_3DP_frame_skip{skip}/train_files.txt",
    )
    generate_metadata_file(
        test_videos,
        skip,
        rgb_path=rgb_path,
        rectified_base_path=rectified_base_path,
        metadata_file_path=f"./Pixel4_3DP_frame_skip{skip}/test_files.txt",
    )
    generate_metadata_file(
        val_videos,
        skip,
        rgb_path=rgb_path,
        rectified_base_path=rectified_base_path,
        metadata_file_path=f"./Pixel4_3DP_frame_skip{skip}/val_files.txt",
    )
