import os
import cv2
import swap_mukham.default_paths as dp
from swap_mukham.models.upscaler.face import FaceUpscaler
from swap_mukham.models.upscaler.generic import GenericUpscaler

def load_upscaler(name, **kwargs):
    for category_name, upscaler in dp.UPSCALERS.items():
        if name in upscaler.keys():
            if category_name == "face":
                return FaceUpscaler(model_file=upscaler[name], **kwargs)
            elif category_name == "generic":
                return GenericUpscaler(model_file=upscaler[name], **kwargs)
            else:
                ValueError(f"Unknown Upscaler Model {name}")
