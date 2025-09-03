import os
import numpy as np
from PIL import Image

import os
os.environ['GLOG_minloglevel'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

import caffe
from tqdm import tqdm


class FaceSegmentation:
    def __init__(self, model_prototxt='data/face_seg_fcn8s_300_deploy.prototxt', model_weights='data/face_seg_fcn8s_300.caffemodel', image_size=(300, 300),
                 mean_bgr=np.array([104.00698793, 116.66876762, 122.67891434]),
                 use_gpu=False, gpu_id=0):
        """
        Initialize the segmentation model.
        """
        if use_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(model_prototxt, model_weights, caffe.TEST)
        self.image_size = image_size
        self.mean_bgr = mean_bgr

    def preprocess(self, image_path):
        """
        Load image, resize, convert RGB->BGR, subtract mean, transpose.
        """
        im = Image.open(image_path).resize(self.image_size)
        im_np = np.array(im, dtype=np.float32)
        im_np = im_np[:, :, ::-1]  # RGB -> BGR
        im_np -= self.mean_bgr
        im_np = im_np.transpose((2, 0, 1))  # CxHxW
        return im_np

    def run(self, image_path):
        """
        Run forward pass and return segmentation mask.
        """
        input_blob = self.preprocess(image_path)
        self.net.blobs['data'].reshape(1, *input_blob.shape)
        self.net.blobs['data'].data[...] = input_blob
        self.net.forward()
        mask = self.net.blobs['score'].data[0].argmax(axis=0)
        return mask


if __name__ == "__main__":

    ROOT_FOLDER = "dataset_DECA_cheo"

    seg_model = FaceSegmentation()

    # Walk through subfolders
    for subdir, _, files in os.walk(ROOT_FOLDER):
        for f in tqdm(files, desc=f"Processing {subdir}"):
            if f.endswith("_cropped.png"):
                img_path = os.path.join(subdir, f)
                mask_path = os.path.splitext(img_path)[0] + "_mask.npy"

                if os.path.exists(mask_path):
                    continue

                try:
                    mask = seg_model.run(img_path)  # only return mask
                    np.save(mask_path, mask)        # save outside the class
                except Exception as e:
                    print(f"Failed {img_path}: {e}")
