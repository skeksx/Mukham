import cv2
import numpy as np
import onnxruntime
from swap_mukham.utils.device import OnnxInferenceSession


class Occluder(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name

        if "in_face" in self.input_name:
            # xseg occluder
            self.crop_size = inputs[0].shape[1:3]
            self.pre_order = (0, 1, 2, 3)
            self.post_order = (0, 1, 2)
        else:
            # generic occluder
            self.crop_size = (256, 256)
            self.pre_order = (0, 3, 1, 2)
            self.post_order = (1, 2, 0)

    def predict(self, img):
        img_size = img.shape[:2]
        img = cv2.resize(img, self.crop_size)
        img = np.expand_dims(img, axis=0).astype("float32") / 255
        img = img.transpose(*self.pre_order)

        mask = self.session.run(None, {self.input_name: img})[0][0]
        mask = mask.transpose(*self.post_order)
        mask = mask.clip(0, 1).astype("float32")
        mask = cv2.resize(mask, img_size)

        return mask

    def delete_session(self):
        if hasattr(self, "session"):
            del self.session
