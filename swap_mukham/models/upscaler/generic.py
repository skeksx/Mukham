import cv2
import torch
import onnxruntime
import numpy as np
import threading
import time
from swap_mukham.utils.device import OnnxInferenceSession

lock = threading.Lock()

class GenericUpscaler(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.resolution = self.session.get_inputs()[0].shape[-2:]

    def preprocess(self, img):
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)[:, :, ::-1] / 255.0
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, img):
        img = (img.transpose(1, 2, 0))
        img = (img * 255)[:, :, ::-1]
        img = img.clip(0, 255)
        return img

    def enhance(self, img):
        img = self.preprocess(img)
        with lock:
            output = self.session.run(None, {"input": img})[0][0]
        output = self.postprocess(output)
        return output
