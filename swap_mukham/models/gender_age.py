import cv2
import numpy as np
import onnxruntime
from swap_mukham.utils.face_alignment import align_crop
from swap_mukham.utils.device import OnnxInferenceSession


class GenderAge(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def predict(self, img, kps):
        aimg, matrix = align_crop(img, kps, 128)

        blob = cv2.resize(aimg, (62, 62), interpolation=cv2.INTER_AREA)
        blob = np.expand_dims(blob, axis=0).astype("float32")

        _prob, _age = self.session.run(None, {"data": blob})
        prob = _prob[0][0][0]
        age = round(_age[0][0][0][0] * 100)
        gender = np.argmax(prob)

        return gender, age
