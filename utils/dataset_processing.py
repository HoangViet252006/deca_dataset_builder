import os
import shutil
from collections import defaultdict
from glob import glob
import cv2
import random

def reorganize_dataset(root_folder):
    """
    Group files in each subfolder by their base prefix and move them into new subfolders.
    Suffixes like '_cropped', '_mask', '_lmk' are ignored when grouping.
    """
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Group files by prefix
        file_groups = defaultdict(list)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if not os.path.isfile(file_path):
                continue

            prefix = os.path.splitext(file)[0]
            while any(prefix.endswith(suffix) for suffix in ["_cropped", "_mask", "_lmk"]):
                prefix = prefix.rsplit("_", 1)[0]

            file_groups[prefix].append(file_path)

        # Move files into new folders
        for prefix, files in file_groups.items():
            prefix_folder = os.path.join(subfolder_path, prefix)
            os.makedirs(prefix_folder, exist_ok=True)
            for file_path in files:
                new_path = os.path.join(prefix_folder, os.path.basename(file_path))
                if not os.path.exists(new_path):
                    shutil.move(file_path, new_path)
            print(f"Moved {len(files)} files into {prefix_folder}")


import os
import cv2
from glob import glob

def video_folder_to_frames(video_folder, output_root, sample_step=10,
                           extensions=(".mp4", ".vid", ".ebm", ".mpg"),
                           start_minute=None, duration_minute=None):
    """
    Extract frames from videos in a folder.
    Each video gets its own subfolder. Frames are sampled every `sample_step` frames.

    If `start_minute` and `duration_minute` are None, extract the entire video.
    """
    os.makedirs(output_root, exist_ok=True)

    # Collect video files
    video_files = []
    for ext in extensions:
        video_files.extend(glob(os.path.join(video_folder, f"*{ext}")))

    all_frames = {}

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(output_root, video_name)
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If start/duration are provided, restrict range; otherwise use full video
        if start_minute is not None and duration_minute is not None:
            start_frame = int(start_minute * 60 * fps)
            end_frame = min(int((start_minute + duration_minute) * 60 * fps), total_frames)
        else:
            start_frame = 0
            end_frame = total_frames

        frame_list = []

        for count in range(total_frames):
            success, frame = cap.read()
            if not success:
                break
            if start_frame <= count < end_frame and (count - start_frame) % sample_step == 0:
                frame_path = os.path.join(output_folder, f"{video_name}_frame{count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_list.append(frame_path)
            if count >= end_frame:
                break

        cap.release()
        print(f"Saved {len(frame_list)} frames from {video_path} to {output_folder}")
        all_frames[video_path] = frame_list

    return all_frames



def remove_small_subfolders(dataset_path, min_files=4):
    """
    Delete subfolders with fewer than `min_files` files.
    """
    for main_folder in os.listdir(dataset_path):
        main_folder_path = os.path.join(dataset_path, main_folder)
        if not os.path.isdir(main_folder_path):
            continue

        for subfolder in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                file_count = sum(os.path.isfile(os.path.join(subfolder_path, f))
                                 for f in os.listdir(subfolder_path))
                if file_count < min_files:
                    print(f"Removing folder: {subfolder_path} ({file_count} files)")
                    shutil.rmtree(subfolder_path)


def split_dataset(src_root, dst_root, train_ratio=0.9, seed=42):
    random.seed(seed)

    # make train and val dirs
    train_root = os.path.join(dst_root, "train")
    val_root = os.path.join(dst_root, "val")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # iterate over each top-level sequence
    for seq_name in os.listdir(src_root):
        seq_path = os.path.join(src_root, seq_name)
        if not os.path.isdir(seq_path):
            continue

        # all subfolders (frames)
        frame_folders = [f for f in os.listdir(seq_path)
                         if os.path.isdir(os.path.join(seq_path, f))]

        # shuffle and split
        random.shuffle(frame_folders)
        split_idx = int(len(frame_folders) * train_ratio)
        train_folders = frame_folders[:split_idx]
        val_folders = frame_folders[split_idx:]

        # copy train
        for f in train_folders:
            src = os.path.join(seq_path, f)
            dst = os.path.join(train_root, seq_name, f)
            shutil.copytree(src, dst)

        # copy val
        for f in val_folders:
            src = os.path.join(seq_path, f)
            dst = os.path.join(val_root, seq_name, f)
            shutil.copytree(src, dst)

        print(f"Processed {seq_name}: {len(train_folders)} train, {len(val_folders)} val")


def is_blurry(image_path, threshold=100):
    """
    Return True if the image is blurry.
    threshold: smaller value = stricter check, larger value = looser check
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True  # treat unreadable images as "blurry"
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    # if laplacian_var < threshold:
    #     print(f"Removed blurry image: {image_path} with laplacian_var: {laplacian_var}")

    return laplacian_var < threshold


def remove_blurry_cropped_images(root_dir, threshold=20):
    """
    Traverse the entire dataset and delete images ending with *_cropped.png if they are blurry.
    """
    removed_files = []
    total = 0
    removed = 0
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith("_cropped.png"):
                total += 1
                file_path = os.path.join(subdir, filename)
                if is_blurry(file_path, threshold=threshold):
                    removed += 1
                    os.remove(file_path)
                    removed_files.append(file_path)

    # print(f"Percentage of blurry images: {removed / total * 100:.2f}%")
    return removed_files


if __name__ == "__main__":
    video_folder_to_frames("cheo_videos", "dataset_DECA_cheo", sample_step=12)
    #
    # remove_blurry_cropped_images("dataset_DECA_cheo")
    #
    # remove_small_subfolders("dataset_DECA_cheo")
    #
    # src_root = "dataset_DECA_cheo"  # original dataset
    # dst_root = "dataset_DECA_cheo_split"  # same root, will create train/ and val/ inside
    # split_dataset(src_root, dst_root)