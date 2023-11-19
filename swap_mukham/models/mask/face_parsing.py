import cv2
import onnxruntime
import numpy as np

bisenet_face_regions = {
    "Background": 0,
    "Skin": 1,
    "Left-Eyebrow": 2,
    "Right-Eyebrow": 3,
    "Left-Eye": 4,
    "Right-Eye": 5,
    "Eye-Glasses": 6,
    "Left-Ear": 7,
    "Right-Ear": 8,
    "Earring": 9,
    "Nose": 10,
    "Mouth": 11,
    "Upper-Lip": 12,
    "Lower-Lip": 13,
    "Neck": 14,
    "Necklace": 15,
    "Cloth": 16,
    "Hair": 17,
    "Hat": 18,
}

parsenet_face_regions = {
    "Background": 0,
    "Skin": 1,
    "Nose": 2,
    "Eye-Glasses": 3,
    "Right-Eye": 4,
    "Left-Eye": 5,
    "Right-Eyebrow": 6,
    "Left-Eyebrow": 7,
    "Right-Ear": 8,
    "Left-Ear": 9,
    "Mouth": 10,
    "Upper-Lip": 11,
    "Lower-Lip": 12,
    "Hair": 13,
    "Hat": 14,
    "Earring": 15,
    "Necklace": 16,
    "Neck": 17,
    "Cloth": 18,
}

default_face_regions = [
    "Skin",
    "Nose",
    "Right-Eye",
    "Left-Eye",
    "Right-Eyebrow",
    "Left-Eyebrow",
    "Mouth",
    "Upper-Lip",
    "Lower-Lip",
    "Neck"
]

import time
class FaceParser:
    def __init__(
        self, model_path=None, provider=["CPUExecutionProvider"], session_options=None
    ):
        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=self.session_options, providers=provider
        )
        self.inputs = self.session.get_inputs()


    def parse(self, img, regions=[1, 2, 3, 4, 5, 10, 11, 12, 13]):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:, :, ::-1] / 127.5 - 1
        img = np.expand_dims(img.transpose((2, 0, 1)), axis=0).astype(np.float32)

        out = self.session.run(None, {self.inputs[0].name: img})[0]
        out = out.squeeze(0).argmax(0)
        out = np.isin(out, regions).astype("float32")

        return out.clip(0, 1)


def mask_regions_to_list(values, model_type="bisenet"):
    out_ids = []
    if model_type == "bisenet":
        mask_regions = bisenet_face_regions
    elif model_type == "parsenet":
        mask_regions = parsenet_face_regions
    else:
        raise ValueError("Unknown face parse model type!")
    for value in values:
        if value in mask_regions.keys():
            out_ids.append(mask_regions.get(value))
    return out_ids
