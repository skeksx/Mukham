import cv2
import numpy as np
import onnxruntime
from swap_mukham.utils.device import OnnxInferenceSession


class DFM(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        inputs = self.session.get_inputs()
        self.model_type = len(inputs)

        if self.model_type not in [1, 2]:
            raise Exception(f"Invalid dfm model {self.model_file}")
        else:
            if "in_face" not in inputs[0].name:
                raise Exception(f"Invalid dfm model {self.model_file}")
            if self.model_type == 2 and "morph_value" not in inputs[1].name:
                raise Exception(f"Invalid dfm model {self.model_file}")
            else:
                self.input_height, self.input_width = inputs[0].shape[1:3]
                if self.input_height != self.input_width:
                    raise Exception(f"Unsupported dfm model size {self.model_file}")
        self.crop_size = self.input_width
        self.align_template = "ffhq"

    def predict(self, target, morph_factor=0.75, use_celeb_mask=False, use_src_mask=False):
        target = cv2.resize(target, (self.input_width, self.input_height))

        src = np.expand_dims(target, axis=0).astype("float32") / 255
        src_mask, celeb_mask = None, None

        if self.model_type == 1:
            src_mask, src, celeb_mask = self.session.run(None, {"in_face:0": src})
        elif self.model_type == 2:
            src_mask, src, celeb_mask = self.session.run(None, {"in_face:0": src, "morph_value:0": np.float32([morph_factor])})

        masks = []

        if use_celeb_mask:
            masks.append(celeb_mask[0])
        if use_src_mask:
            masks.append(src_mask[0])

        mask = None
        if len(masks) > 0:
            mask = np.prod(masks, axis=0).clip(0, 1)

        return (src[0] * 255).clip(0, 255), mask
