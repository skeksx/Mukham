import gradio
from swap_mukham.utils import image as image_util
gradio.processing_utils.encode_pil_to_base64 = image_util.fast_pil_encode
gradio.processing_utils.encode_array_to_base64 = image_util.fast_numpy_encode

from collections import OrderedDict
from swap_mukham import global_variables as gv
from swap_mukham.models.mask.face_parsing import mask_regions_to_list
from swap_mukham.gui import controls, face_id, input, output, preview, save, settings, update, theme

class Components:
    def __init__(self):
        pass

    @property
    def processed_source_data(self):
        source_data_all = self.source_data.copy()
        return source_data_all

    @property
    def processed_filter_data(self):
        filters = self.target_selection_data.copy()
        processed = {}
        for key, data in filters.items():
            if isinstance(data, str):
                processed[key] = data.lower()
        return processed

    @property
    def processed_target_data(self):
        target_data_all = self.target_data.copy()
        return target_data_all

    @property
    def processed_control_data(self):
        face_control_data = self.face_control_data.copy()
        for face_id, data in face_control_data.items():
            control_data = dict(face_control_data[face_id])

            crop_top, crop_bott = control_data["crop_top"] / 100, control_data["crop_bott"] / 100
            crop_left, crop_right = control_data["crop_left"] / 100, control_data["crop_right"] / 100
            if crop_top > crop_bott:
                crop_top, crop_bott = crop_bott, crop_top
            if crop_left > crop_right:
                crop_left, crop_right = crop_right, crop_left
            crop_mask = (crop_top, 1 - crop_bott, crop_left, 1 - crop_right)

            control_data["border_crop_values"] = crop_mask
            control_data["border_fade_amount"] /= 100
            control_data['face_parse_regions'] = mask_regions_to_list(control_data["mask_regions"])

            face_control_data[face_id] = control_data
        frame_control_data = self.frame_control_data.copy()
        return face_control_data, frame_control_data

    @property
    def processed_settings_data(self):
        settings_data = self.settings_data.copy()
        return settings_data

    @property
    def processed_output_data(self):
        output_data = self.output_data.copy()
        return output_data


components = Components()
components.realtime_update_components = []

components.face_control_data = dict(zip(gv.FACE_IDS, [gv.FACE_CONTROL_DEFAULTS] * gv.NUMBER_OF_IDS))
components.source_data = dict(zip(gv.FACE_IDS, [gv.SOURCE_DEFAULTS] * gv.NUMBER_OF_IDS))
components.target_selection_data = gv.TARGET_SELECTION_DEFAULTS
components.frame_control_data = gv.FRAME_CONTROL_DEFAULTS
components.settings_data = gv.SETTINGS_DEFAULTS
components.target_data = gv.TARGET_DEFAULTS
