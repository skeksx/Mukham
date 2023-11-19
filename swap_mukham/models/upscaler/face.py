import cv2
import torch
import onnxruntime
import numpy as np
import threading
import time
from swap_mukham.utils.device import OnnxInferenceSession

lock = threading.Lock()
semaphore = threading.Semaphore()

class FaceUpscaler(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.n_inputs = len(self.input_names)
        self.resolution = self.session.get_inputs()[0].shape[-2:]
        self.codeformer_fidelity = 0.9

    def preprocess(self, img):
        img = cv2.resize(img, self.resolution)
        img = img.astype(np.float32)[:, :, ::-1] / 127.5 - 1
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, img):
        img = (img.transpose(1, 2, 0).clip(-1, 1) + 1) * 0.5
        img = (img * 255)[:, :, ::-1]
        img = img.clip(0, 255)
        return img

    def enhance(self, img):
        img = self.preprocess(img)
        with semaphore:
            if self.n_inputs == 1: # gfpgan, gpen, restoreformer
                output = self.session.run(None, {
                    self.input_names[0]: img
                })[0][0]
            elif self.n_inputs == 2: # codeformer
                output = self.session.run(None, {
                    self.input_names[0]: img,
                    self.input_names[1]: np.array([self.codeformer_fidelity], dtype=np.double)
                })[0][0]
            else:
                raise ValueError("Unknown Face Enhancer!")

        output = self.postprocess(output)
        return output
