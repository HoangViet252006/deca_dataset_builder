import numpy as np
import cv2
import os


def save_mask_vis(img_path, mask_path, save_path, class_id=1, color=(0, 0, 255), alpha=0.4):
    """
    Overlay a segmentation mask on the original image and save the result.

    Args:
        img_path (str): Path to the original image.
        mask_path (str): Path to the .npy mask file.
        save_path (str): Path to save the visualized image.
        class_id (int): Mask class to highlight.
        color (tuple): BGR color for overlay (default red).
        alpha (float): Transparency factor (0=transparent, 1=opaque).
    """
    # Load original image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Load mask
    mask = np.load(mask_path)
    if mask.shape[:2] != image.shape[:2]:
        # Resize mask if it doesn't match image size
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply overlay for the selected class
    overlay = image.copy()
    mask_pixels = (mask == class_id)
    overlay[mask_pixels] = ((1 - alpha) * overlay[mask_pixels] + alpha * np.array(color)).astype(np.uint8)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the visualized image
    cv2.imwrite(save_path, overlay)
    print(f"Saved mask visualization to {save_path}")


def save_landmark_vis(img_path, lmk_path, save_path="landmarks_vis.png"):
    """
    Draw facial landmarks on an image and save the result.

    Args:
        img_path (str): Path to the original image.
        lmk_path (str): Path to the .npy file containing landmarks.
        save_path (str): Path to save the visualized image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Load landmarks
    lmk = np.load(lmk_path, allow_pickle=True)
    if lmk is None or len(lmk) == 0:
        raise ValueError(f"No landmarks found in {lmk_path}")

    # Handle multiple faces: use the first set of landmarks
    if isinstance(lmk, (list, tuple)):
        lmk = lmk[0]

    # Draw each landmark point
    for (x, y) in lmk.astype(np.int32):
        cv2.circle(img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(save_path, img)
    print(f"Saved landmark visualization to {save_path}")


if __name__ == "__main__":
    # Example usage
    image_path = "dataset_DECA_cheo/trichdoan1/a_cropped.png"
    mask_path = "dataset_DECA_cheo/trichdoan1/a_cropped_mask.npy"
    vis_path = "dataset_DECA_cheo/trichdoan1/a_cropped_vis.png"

    save_mask_vis(image_path, mask_path, vis_path, class_id=1, color=(0, 0, 255), alpha=0.5)

    # Optional: visualize landmarks
    # lmk_path = "dataset_DECA_cheo/trichdoan1/a_cropped_lmk.npy"
    # save_landmark_vis(image_path, lmk_path, "dataset_DECA_cheo/trichdoan1/a_cropped_lmk_vis.png")
