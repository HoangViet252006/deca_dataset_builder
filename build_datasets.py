import os
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from utils.visualization import save_mask_vis, save_landmark_vis
import FAN
from face_seg import face_seg


def get_args():
    parser = argparse.ArgumentParser(description="Build datasets from images or folders")
    parser.add_argument("--input_path", "-i", type=str,
                        default="TestSamples/AFLW2000/image00302.jpg",
                        help="Path to a single image or a folder of images")
    parser.add_argument("--crop", "-c", action="store_true", default=True,
                        help="Whether to crop detected faces")
    parser.add_argument("--landmarks", "-l", action="store_true", default=True,
                        help="Whether to save facial landmarks")
    parser.add_argument("--output_dir", "-o", type=str, default="My_datasets",
                        help="Directory to save processed images and masks")
    return parser.parse_args()


def process_image(image_path, fan_model, face_seg_model, crop_size=224, scale=1.25):
    """
    Process a single image:
    1. Detect face & crop
    2. Save landmarks
    3. Generate and save segmentation mask
    """
    prefix, _ = os.path.splitext(image_path)
    cropped_path = f"{prefix}_cropped.png"
    lmk_path = f"{prefix}_lmk.npy"
    mask_path = f"{prefix}_mask.npy"

    if os.path.exists(cropped_path):
        return  # skip already processed images

    image = cv2.imread(image_path)
    result = fan_model.detect_and_crop(image, crop_size=crop_size, scale=scale)
    if result is None:
        return  # skip images without detected faces

    # Save landmarks
    np.save(lmk_path, result["lmk"])

    # Save cropped image
    img_np = result["image_cropped"].cpu().numpy().transpose(1, 2, 0)
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    cv2.imwrite(cropped_path, img_np)

    # Generate and save face mask
    if not os.path.exists(mask_path):
        mask = face_seg_model.run(cropped_path)
        np.save(mask_path, mask)


def process_folder(root_folder, fan_model=None, face_seg_model=None, crop_size=224, scale=1.25):
    """
    Process all images in a folder (recursively by subfolder):
    - Crop faces
    - Save landmarks
    - Generate segmentation masks
    """
    fan_model = fan_model or FAN.FAN()
    face_seg_model = face_seg_model or face_seg.FaceSegmentation()

    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    for subfolder_name in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(root_folder, subfolder_name)
        image_files = glob(os.path.join(subfolder_path, "*.[jp][pn]g")) + glob(os.path.join(subfolder_path, "*.bmp"))

        for image_path in tqdm(image_files, desc=subfolder_name, leave=False):
            process_image(image_path, fan_model, face_seg_model, crop_size=crop_size, scale=scale)


if __name__ == "__main__":
    args = get_args()

    fan_model = FAN.FAN()
    face_seg_model = face_seg.FaceSegmentation()

    if os.path.isdir(args.input_path):
        process_folder(args.input_path, fan_model, face_seg_model)
    else:
        process_image(args.input_path, fan_model, face_seg_model)

    # Example visualization
    image_path = "data/images/Alison_Lohman_0001.jpg"
    prefix, _ = os.path.splitext(image_path)
    cropped_path = f"{prefix}_cropped.png"
    mask_path = f"{prefix}_mask.npy"
    lmk_path = f"{prefix}_lmk.npy"

    save_mask_vis(img_path=cropped_path, mask_path=mask_path,
                  save_path=f"{prefix}_mask_vis.png")
    # Uncomment to visualize landmarks
    # save_landmark_vis(img_path=cropped_path, lmk_path=lmk_path,
    #                   save_path=f"{prefix}_lmk_vis.png")
