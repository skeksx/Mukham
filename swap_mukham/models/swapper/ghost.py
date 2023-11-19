import cv2
import onnxruntime
import numpy as np
from swap_mukham.utils.device import OnnxInferenceSession


class Ghost(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = 224
        self.align_template = "set2"

    def forward(self, target, source):
        latent = source["embedding"].reshape(1, -1)

        blob = cv2.resize(target, (256, 256))
        blob = blob.astype("float32") / 127.5 - 1
        blob = blob[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        blob = self.session.run(None, {"target": blob, "source_embedding": latent})[0]

        out = blob[0].transpose(1, 2, 0)
        out = out * 127.5 + 127.5

        return out[:, :, ::-1]
