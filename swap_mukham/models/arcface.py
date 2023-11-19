import os
import cv2
import numpy as np
from numpy.linalg import norm
from swap_mukham.utils.face_alignment import align_crop
from swap_mukham.utils.device import OnnxInferenceSession

class ArcFace(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.session.get_inputs()
        self.input_size = tuple(self.inputs[0].shape[2:4][::-1])

    def compute_similarity(self, feat1, feat2):
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        similarity = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return similarity

    def get(self, img, kps, top_fill=0):
        aimg, matrix = align_crop(img, kps, self.input_size[0], method="arcface")
        if top_fill > 0:
            top_pixels = int((aimg.shape[1] / 2) * top_fill)
            # aimg[:top_pixels, :, :] = np.mean(aimg[-top_pixels:, :, :], axis=(0, 1))
            aimg[:top_pixels, :, :] = (127,127,127)
        blob = aimg.astype("float32") / 127.5 - 1
        blob = blob[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        out = self.session.run(None, {self.inputs[0].name: blob})[0]
        return out.ravel(), aimg
