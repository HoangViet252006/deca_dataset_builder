import numpy as np
import torch
from skimage.transform import estimate_transform, warp


def bbox2point(left: float, right: float, top: float, bottom: float):
    """Convert bounding box to center and scale."""
    old_size = (right - left + bottom - top) / 2 * 1.1
    center = np.array([
        right - (right - left) / 2.0,
        bottom - (bottom - top) / 2.0
    ])
    return old_size, center


class FAN(object):
    def __init__(self, device='cuda'):
        import face_alignment
        if device == 'cpu':
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=device
            )
        else:
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=device
            )

    def run(self, image: np.ndarray):
        """
        Run landmark detection.

        Args:
            image (np.ndarray): RGB image, uint8, shape [H, W, 3].

        Returns:
            tuple: (bbox, landmarks) where
                bbox: [x_min, y_min, x_max, y_max] or [0] if no face.
                landmarks: np.ndarray of shape (68, 2) or None if no face.
        """
        out = self.model.get_landmarks(image)
        if not out:  # handles None and empty list
            return [0], None

        kpt = out[0]  # (68, 2)
        x_min, y_min = np.min(kpt, axis=0)
        x_max, y_max = np.max(kpt, axis=0)
        bbox = [x_min, y_min, x_max, y_max]

        return bbox, kpt

    def detect_and_crop(self, image: np.ndarray, crop_size=224, scale=1.25):
        """
        Detect face, crop, and align.

        Args:
            image (np.ndarray): RGB image [H,W,3].
            crop_size (int): output image size.
            scale (float): scaling factor for bbox.

        Returns:
            dict or None: {
                "image_cropped": torch.FloatTensor [3,H,W],
                "lmk": np.ndarray of landmarks (aligned)
            }
        """
        h, w, _ = image.shape
        bbox, lmk = self.run(image)

        if bbox is None or len(bbox) < 4:
            return None

        left, top, right, bottom = bbox
        old_size, center = bbox2point(left, right, top, bottom)
        size = int(old_size * scale)

        src_pts = np.array([
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2]
        ])
        dst_pts = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, dst_pts)

        dst_image = warp(image / 255., tform.inverse, output_shape=(crop_size, crop_size))
        dst_image = dst_image.transpose(2, 0, 1)  # [3,H,W]

        lmk_aligned = tform(lmk) if lmk is not None else None
        return {"image_cropped": torch.tensor(dst_image).float(), "lmk": lmk_aligned}
