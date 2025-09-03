import os
import shutil
from collections import defaultdict
from glob import glob
import cv2


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


def video_folder_to_frames(video_folder, output_root, sample_step=10,
                           extensions=(".mp4", ".vid", ".ebm", ".mpg"),
                           start_minute=1, duration_minute=3):
    """
    Extract frames from videos in a folder.
    Each video gets its own subfolder. Frames are sampled every `sample_step` frames.

    Returns a dictionary mapping video paths to lists of saved frame paths.
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

        start_frame = int(start_minute * 60 * fps)
        end_frame = min(int((start_minute + duration_minute) * 60 * fps), total_frames)

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


if __name__ == "__main__":
    remove_small_subfolders("dataset_DECA_cheo")
