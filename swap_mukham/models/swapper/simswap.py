import cv2
import onnxruntime
import numpy as np
from swap_mukham.utils.device import OnnxInferenceSession


class SimSwap(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = self.session.get_inputs()[0].shape[-1]
        if self.crop_size == 512:
            self.align_template = "ffhq"
        else:
            self.align_template = "arcface"

    def forward(self, target, source):
        latent = source["embedding"].reshape(1, -1)
        latent /= np.linalg.norm(latent)

        blob = cv2.resize(target, (self.crop_size, self.crop_size))
        blob = blob.astype("float32") / 255
        blob = blob[:, :, ::-1]
        if self.crop_size == 256:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            blob = (blob - mean) / std
        blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2).astype("float32")

        blob = self.session.run(None, {self.session.get_inputs()[0].name: blob, self.session.get_inputs()[1].name: latent})[0]

        out = blob[0].transpose((1, 2, 0))
        out = (out * 255).clip(0, 255)

        return out[:, :, ::-1]


class SimSwapUnofficial(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = self.session.get_inputs()[0].shape[-1]
        self.align_template = "arcface"

    def forward(self, target, source):
        latent = source["embedding"].reshape(1, -1)
        latent /= np.linalg.norm(latent)

        blob = cv2.resize(target, (self.crop_size, self.crop_size))
        blob = blob.astype("float32") / 127.5 - 1
        blob = blob[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        blob = self.session.run(None, {self.session.get_inputs()[0].name: blob, self.session.get_inputs()[1].name: latent})[0]

        out = blob[0].transpose(1, 2, 0)
        out = out * 127.5 + 127.5
        out = out.clip(0, 255)

        return out[:, :, ::-1]
