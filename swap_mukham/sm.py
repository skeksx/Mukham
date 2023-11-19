import os
import cv2
import time
import numpy as np

import swap_mukham.default_paths as dp
import swap_mukham.global_variables as gv

from swap_mukham.utils import misc as misc
from swap_mukham.utils import image as image_util
from swap_mukham.utils.device import get_onnx_provider
from swap_mukham.utils.face_alignment import align_crop
from swap_mukham.utils.face_alignment import get_cropped_head

from swap_mukham.models.mask.occluder import Occluder
from swap_mukham.models.upscaler import load_upscaler
from swap_mukham.models.mask.face_parsing import FaceParser
from swap_mukham.models.swapper import load_face_swapper, DFM
from swap_mukham.models.detector import Detector

from swap_mukham.face_analyser import (
    AnalyseFace,
    get_single_face,
    is_similar_face,
    filter_face_by_age,
    filter_face_by_gender,
)


class SwapMukham:
    def __init__(self, device="cpu"):
        self.preferred_device = device
        self.face_parser = None
        self.face_swapper = None
        self.face_upscaler = None
        self.current_face_swapper_name = ""
        self.current_face_upscaler_name = ""
        self.occluder_model = None
        self.current_face_id = '1'
        self.use_control_id = False
        self.dfm_sessions = {}

        self.load_detector(device=device)
        self.load_face_analyser(device=device)
        self.load_face_swapper(gv.DEFAULT_SWAPPER, device=device)

    def set_attributes(self, filters, settings, face_control_data, frame_control_data):
        for key, value in filters.items():
            setattr(self, key, value)
        for key, value in settings.items():
            setattr(self, key, value)

        self.frame_controls = frame_control_data
        self.post_process_settings = face_control_data

        self.analyser.detection_size = (int(self.face_detection_size), int(self.face_detection_size))
        self.analyser.detection_threshold = self.face_detection_threshold
        self.analyser.adaptive_threshold = False

        self.load_face_swapper(self.face_swapper_name, device=self.preferred_device)

        check_face_upscaler = False

        for face_id, data in self.post_process_settings.items():
            if data['pre_upscale'] or data['post_upscale']:
                check_face_upscaler = True

        if check_face_upscaler:
            self.load_face_upscaler(self.face_upscaler_name, device=self.preferred_device)



    def load_detector(self, device="cpu"):
        self.detector = Detector(model_file=dp.DETECTOR_PATH, **get_onnx_provider(device=device))

    def load_face_analyser(self, device="cpu"):
        self.analyser = AnalyseFace(**get_onnx_provider(device=device))

    def load_face_swapper(self, name, device="cpu"):
        if self.current_face_swapper_name != name:
            if hasattr(self.face_swapper, 'delete_session'):
                self.face_swapper.delete_session()
            if hasattr(self.analyser.recognizer, 'delete_session'):
                self.analyser.recognizer.delete_session()
            self.face_swapper, self.analyser.recognizer = load_face_swapper(name, **get_onnx_provider(device=device))
            self.current_face_swapper_name = name

    def load_face_upscaler(self, name, device="cpu"):
        if self.current_face_upscaler_name != name:
            self.current_face_upscaler_name = name
            if hasattr(self.face_upscaler, 'delete_session'):
                self.face_upscaler.delete_session()
            self.face_upscaler = load_upscaler(name, **get_onnx_provider(device=device))

    def load_face_parser(self, device="cpu"):
        self.face_parser = FaceParser(model_path=dp.FACE_PARSER_PATH, **get_onnx_provider(device=device))

    def load_occluder_model(self, device="cpu"):
        if not os.path.exists(self.occluder_model_path):
            self.occluder_model = None
            return
        self.occluder_model = Occluder(model_file=self.occluder_model_path, **get_onnx_provider(device=device))



    def create_dfm_session(self, model_path, device="cuda"):
        if model_path in self.dfm_sessions.keys():
            return self.dfm_sessions[model_path]
        dfm_session = DFM(model_file=model_path, **get_onnx_provider(device=device))
        self.dfm_sessions[model_path] = dfm_session
        return dfm_session

    def delete_all_dfm_sessions(self):
        while self.dfm_sessions:
            model_path, session = self.dfm_sessions.popitem()
            session.delete_session()



    def collect_heads(self, frame):
        faces = self.analyser.get_faces(frame, skip_task=["embedding", "gender_age"])
        return [
            get_cropped_head(frame, face.kps)
            for face in faces
            if face["det_score"] > 0.5
        ]



    def analyse_source_faces(self, source_data, current_face_id):
        analysed_source_specific = []

        if self.filter_type != "by specific face":
            source_data = {current_face_id: source_data.get(current_face_id)}

        for face_id, data in source_data.items():

            source_type = data['source_type']
            source_images = data['source_image_input']
            dfm_name = data['source_dfm_input']
            specific_image = data['specific_image_input']

            if source_type == "Image":
                if not source_images:
                    continue

                settings = self.post_process_settings[face_id]
                source_top_fill = 1 - settings["source_forehead_influence"]
                averaged_source = self.analyser.get_face_averaged(source_images, method=self.face_averaging_method, top_fill=source_top_fill)
                analysed_source = [averaged_source, "embedding"]

            elif source_type == "DFM":
                if not dfm_name:
                    continue
                dfm_path = gv.DFM_MODELS.get(dfm_name, None)
                if dfm_path is None:
                    raise ValueError(f"{dfm_name} not found!")

                analysed_source = [self.create_dfm_session(dfm_path, device=self.preferred_device), "dfm"]

            analysed_specific =  self.analyser.get_face_averaged(specific_image, method=self.face_averaging_method) if specific_image else None
            analysed_source_specific.append((analysed_source, analysed_specific, face_id))

        self.analysed_source_specific = analysed_source_specific



    def filter_faces(self, analysed_target_faces):
        filtered_data = []

        if len(analysed_target_faces) == 0:
            return filtered_data

        if self.filter_type == "by first detected":
            analysed_target = analysed_target_faces[0]
            src_face = self.analysed_source_specific[0][0]
            face_id = self.current_control_id
            settings = self.post_process_settings[face_id]
            filtered_data.append([analysed_target, src_face, settings])

        elif self.filter_type == "by condition":
            if self.filter_condition_gender != "any":
                analysed_target_faces = filter_face_by_gender(analysed_target_faces, self.filter_condition_gender)
            if self.filter_condition_age_group != "any":
                analysed_target_faces = filter_face_by_age(analysed_target_faces, self.filter_condition_age_group)
            if self.filter_condition_position != "any":
                analysed_target_faces = [get_single_face(analysed_target_faces, method=self.filter_condition_position)]

            # TODO track face

            src_face = self.analysed_source_specific[0][0]
            face_id = self.current_control_id
            settings = self.post_process_settings[face_id]

            for analysed_target in analysed_target_faces:
                filtered_data.append([analysed_target, src_face, settings])

        elif self.filter_type == "by specific face":
            for analysed_target in analysed_target_faces:
                for analysed_source, analysed_specific, face_id in self.analysed_source_specific:
                    if not self.use_control_id:
                        face_id = self.current_control_id
                    if is_similar_face(analysed_specific, analysed_target, threshold=self.face_similarity_threshold):
                        settings = self.post_process_settings[face_id]
                        filtered_data.append([analysed_target, analysed_source, settings])

        return filtered_data


    #@misc.measure_execution_time
    def process_frame(self, frame, custom_mask):
        if not self.analysed_source_specific:
            raise ValueError("Source not analysed")
            return frame

        frame, alpha = image_util.convert_to_3_channel(frame)
        original_frame = frame.copy()

        skip_task = []

        if self.filter_type != "by specific face":
            skip_task.append("embedding")

        if self.filter_type == "by condition" and self.filter_condition_age_group == 'any' and self.filter_condition_gender == 'any':
            skip_task.append("gender_age")

        if self.filter_type == "by first detected":
            skip_task.append("gender_age")

        analysed_target_faces = self.analyser.get_faces(frame, skip_task=skip_task)

        filtered_faces = self.filter_faces(analysed_target_faces)

        if len(filtered_faces) == 0:
            return frame, None, None

        swapped_mask = []
        keypoints = []
        for data in filtered_faces:
            _, _, settings = data
            iterations = max(settings["whole_process_iteration"], 1)
            for i in range(iterations):
                frame, _mask, kps = self.process_face(frame, *data)
            keypoints.append(kps)
            swapped_mask.append(_mask)

        # TODO frame processing

        if custom_mask is not None:
            mask = cv2.resize(custom_mask, frame.shape[:2][::-1])
            original_frame = original_frame.astype("float32")
            frame = frame.astype("float32")
            frame = mask * original_frame + (1 - mask) * frame
            frame = frame.clip(0, 255).astype("uint8")

        if alpha is not None:
            frame = np.dstack((frame, alpha))

        return frame, swapped_mask, keypoints


    #@misc.measure_execution_time
    def process_face(self, frame, target, source_and_type, settings):
        source, source_type = source_and_type
        crop_size = source.crop_size if source_type == "dfm" else self.face_swapper.crop_size
        alignment_method = settings["face_alignment_method"]
        if alignment_method == "auto":
            alignment_method = "generic" if source_type == "dfm" else self.face_swapper.align_template
        default_interpolation = gv.INTERPOLATION_MAP.get(settings['default_interpolation'], cv2.INTER_LINEAR)

        masks = []

        # align and crop
        kps = target["kps"]
        target_cropped, matrix = align_crop(frame, kps, crop_size, method=alignment_method, scale=settings["landmark_scale"],)

        # pre upscale
        if settings["pre_upscale"]:
            target_cropped = self.face_upscale(target_cropped, 1.0)

        reference = target_cropped.copy()

        # pre process
        target_cropped = image_util.blur(target_cropped, settings["pre_blur_amount"])
        target_cropped = image_util.sharpen(target_cropped, settings["pre_sharpen_size"], settings["pre_sharpen_amount"])

        # swap face
        target_cropped, mask = self.face_swap(target_cropped, source, source_type, settings)
        masks.append(mask)

        # re-align with ffhq
        if settings["re_align"]:
            _frame, foreground, _mask = image_util.paste_back(
                cv2.resize(target_cropped, (512,512)),
                frame.copy(),
                matrix * 512 / crop_size,
                custom_mask=None,
                border_crop=(0,0,0,0),
                border_fade_amount=0.2,
                blend_method="alpha blend",
                interpolation=default_interpolation
            )

            new_face = self.analyser.get_face(foreground, skip_task=['gender_age', 'embedding'])

            if new_face is not None:
                crop_size = 512
                kps = new_face["kps"]
                reference, _ = align_crop(frame, kps, crop_size, method="ffhq", scale=1)
                target_cropped, matrix = align_crop(_frame, kps, crop_size, method="ffhq", scale=1)
            else:
                print("FFHQ alignment failed")

        # post upscale
        if settings["post_upscale"] and settings["face_upscaler_opacity"] > 0:
            upscaled_face = self.face_upscale(target_cropped, settings["face_upscaler_opacity"])
        else:
            upscaled_face = cv2.resize(target_cropped, (512, 512), interpolation=default_interpolation)

        # post process
        upscaled_face = image_util.median_blur(upscaled_face, settings["median_blur"])
        upscaled_face = image_util.jpeg_compress(upscaled_face, settings["jpeg_compression"])
        upscaled_face = image_util.downsample(upscaled_face, settings["downsample"])

        # face parse
        if settings["face_parse_from_target"]:
            masks.append(self.face_parse(reference, settings))
        if settings["face_parse_from_source"]:
            masks.append(self.face_parse(upscaled_face, settings))

        # occluder
        if settings["occlusion_from_target"]:
            masks.append(self.face_occlusion(reference))
        if settings["occlusion_from_source"]:
            masks.append(self.face_occlusion(upscaled_face))

        # final matrix size
        final_size = upscaled_face.shape[1]
        matrix *=  final_size / crop_size

        # merge mask
        masks = list(filter(lambda x: x is not None, masks))
        if len(masks) > 0:
            masks = [cv2.resize(mask, (final_size, final_size)).astype("float32") for mask in masks]
            final_mask = np.minimum.reduce(masks)
            final_mask = image_util.smooth_mask_edges(final_mask, radius=settings["mask_smooth_radius"], iterations=settings["mask_smooth_iteration"],)
            final_mask = image_util.erode_blur(final_mask, settings["mask_erode"], settings["mask_blur"])
        else:
            final_mask = None

        # color match
        if settings["use_color_match"]:
            upscaled_face = image_util.color_match(upscaled_face, reference, weights=final_mask)

        # paste back to original
        result, foreground, mask = image_util.paste_back(
            upscaled_face,
            frame,
            matrix,
            custom_mask=final_mask,
            border_crop=settings["border_crop_values"],
            border_fade_amount=settings["border_fade_amount"],
            blend_method=settings["blending_mode"],
            interpolation=default_interpolation
        )

        return result, mask, kps

    def face_swap(self, target, source, source_type, settings):
        mask = None
        if source_type == "dfm":
            target, mask = source.predict(target, use_celeb_mask=settings["dfm_celeb_mask"], use_src_mask=settings["dfm_src_mask"])
        elif source_type == "embedding":
            target = self.face_swapper.forward(target, source)
        target = target.clip(0, 255)
        return target, mask


    def face_upscale(self, face, opacity):
        if self.face_upscaler is None:
            raise Exception("Face Upscaler not loaded")
        before_face = face.copy()
        upscaled_face = self.face_upscaler.enhance(face)
        upscaled_face = image_util.mix_two_image(before_face, upscaled_face, opacity)
        return upscaled_face

    def face_parse(self, face, settings):
        if self.face_parser is None:
            self.load_face_parser(device=self.preferred_device)
        mask = self.face_parser.parse(face, regions=settings["face_parse_regions"])
        mask = image_util.erode_blur(mask, settings["face_parse_erode"], 0)
        if settings["face_parse_invert"]:
            return 1.0 - mask
        return mask

    def face_occlusion(self, face):
        if self.occluder_model is None:
            self.load_occluder_model(device=self.preferred_device)
        if self.occluder_model.model_file != self.occluder_model_path:
            self.occluder_model.delete_session()
            self.load_occluder_model(device=self.preferred_device)
        return self.occluder_model.predict(face)
