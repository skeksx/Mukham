import cv2
import onnxruntime
import numpy as np
from swap_mukham.utils.device import OnnxInferenceSession


class BlendSwap(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = self.session.get_inputs()[0].shape[-1]
        self.align_template = "ffhq"

    def preprocess(self, img, crop_size):
        blob = cv2.resize(img, crop_size)
        blob = blob.astype("float32") / 255
        blob = blob[:, :, ::-1]
        blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2)
        return blob

    def forward(self, target, source):
        target = self.preprocess(target, (self.crop_size, self.crop_size))
        source = self.preprocess(source, (112, 112))

        target = self.session.run(None, {"target": target, "source": source})[0]

        out = target[0].transpose((1, 2, 0))
        out = (out * 255).clip(0, 255)
        out = out.astype("uint8")[:, :, ::-1]

        return out


# with separate embedding

# def forward(self, target, source, n_pass=1):
#     latent = source['embedding'].reshape(1,-1)
#     latent /= np.linalg.norm(latent)

#     blob = cv2.resize(target, self.crop_size)
#     blob = blob.astype('float32') / 255
#     blob = blob[:, :, ::-1]
#     blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2)

#     for _ in range(max(int(n_pass),1)):
#         blob, mask = self.session.run(None, {'target': blob, 'source_embedding': latent})

#     out = blob[0].transpose((1, 2, 0))
#     out = (out * 255).clip(0,255)
#     out = out.astype('uint8')[:, :, ::-1]

#     return out
