import os
import json
import random


def generate_metadata_file(
    videos, skip, rgb_path, dp_path, dpt_path, rectified_base_path, metadata_file_path
):
    metadata = {}

    metadata["videos"] = []
    metadata["skip"] = skip
    metadata["rgb_path"] = rgb_path
    metadata["dp_path"] = dp_path
    metadata["dpt_path"] = dpt_path
    metadata["video_length"] = 8

    def get_dp_path_from_rgb(rgb_frame):
        base_name = rgb_frame.split(".")[0]
        left_name = base_name + "_left.jpg"
        right_name = base_name + "_right.jpg"
        return left_name, right_name

    for video_name in videos:
        video_name_beginning = video_name.split("_")[0]
        if video_name_beginning == "20230212" or video_name_beginning == "20230301":
            continue
        frames_B = sorted(
            os.listdir(os.path.join(rectified_base_path, "B", rgb_path, video_name))
        )
        frames_A = sorted(
            os.listdir(os.path.join(rectified_base_path, "A", rgb_path, video_name))
        )
        frames_C = sorted(
            os.listdir(os.path.join(rectified_base_path, "C", rgb_path, video_name))
        )
        video = {}
        video["name"] = video_name
        video["frames"] = []
        ctr = 0
        
        for i in range(0, len(frames_B), skip):
            # print(frames_A[i])
            rgb_views = {"A": frames_A[i], "B": frames_B[i], "C": frames_C[i]}
            left_dpA, right_dpA = get_dp_path_from_rgb(frames_A[i])
            left_dpB, right_dpB = get_dp_path_from_rgb(frames_B[i])
            left_dpC, right_dpC = get_dp_path_from_rgb(frames_C[i])

            left_dp_views = {"A": left_dpA, "B": left_dpB, "C": left_dpC}
            right_dp_views = {"A": right_dpA, "B": right_dpB, "C": right_dpC}

            video["frames"].append(
                {
                    "rgb": rgb_views,
                    "left_dp": left_dp_views,
                    "right_dp": right_dp_views,
                    "dptdepth": rgb_views,
                }
            )
            
            if len(video["frames"]) == metadata["video_length"]:
                ctr+=1
                metadata["videos"].append(video)
                del video
                video = {}
                video["name"] = video_name
                video["frames"] = []
        print(f"{ctr} sample videos generated from {video_name}")

    print(f"Dataset size: {len(metadata['videos'])} (video length: {metadata['video_length']})")
    metadata_str = json.dumps(metadata, indent=4)

    with open(metadata_file_path, "w") as f:
        f.write(metadata_str)


if __name__ == "__main__":
    rectified_base_path = "/data2/raghav/datasets/Pixel4_3DP/rectified/"
    rgb_path = "Video_data/"
    dp_path = "dp_data/"
    dpt_path = "./disparity_maps/disp_pixel4_BA"

    skip = 10
    # metadata_file_name = f"dpvideos_skip{skip}.json"

    videos = os.listdir(os.path.join(rectified_base_path, "B", rgb_path))
    print("Number of videos:", len(videos))
    random.shuffle(videos)

    train_videos = videos[: int(0.8 * len(videos))]
    val_videos = videos[int(0.8 * len(videos)) : int(0.9 * len(videos))]
    test_videos = videos[int(0.9 * len(videos)) :]

    if not os.path.exists(f"Pixel4_3DP_skip{skip}"):
        os.makedirs(f"./Pixel4_3DP_skip{skip}")
        
    generate_metadata_file(
        train_videos,
        skip,
        rgb_path=rgb_path,
        dp_path=dp_path,
        dpt_path=dpt_path,
        rectified_base_path=rectified_base_path,
        metadata_file_path=f"./Pixel4_3DP_skip{skip}/train_files.json",
    )
    generate_metadata_file(
        test_videos,
        skip,
        rgb_path=rgb_path,
        dp_path=dp_path,
        dpt_path=dpt_path,
        rectified_base_path=rectified_base_path,
        metadata_file_path=f"./Pixel4_3DP_skip{skip}/test_files.json",
    )
    generate_metadata_file(
        val_videos,
        skip,
        rgb_path=rgb_path,
        dp_path=dp_path,
        dpt_path=dpt_path,
        rectified_base_path=rectified_base_path,
        metadata_file_path=f"./Pixel4_3DP_skip{skip}/val_files.json",
    )
