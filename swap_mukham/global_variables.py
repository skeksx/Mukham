import os
import cv2
from collections import OrderedDict
import swap_mukham.default_paths as dp
from swap_mukham.models.mask.face_parsing import bisenet_face_regions, default_face_regions
from swap_mukham.utils.image import resolution_map


def get_available_upscaler_names():
    available_upscalers = []
    for category_name, upscaler in dp.UPSCALERS.items():
        for name, path in upscaler.items():
            if os.path.exists(path):
                available_upscalers.append(name)
    return available_upscalers


def get_available_swapper_names():
    available_swappers = []
    for category_name, swapper in dp.EMBEDDING_BASED_SWAPPERS.items():
        if os.path.exists(swapper["backbone"]):
            for name, path in swapper.items():
                if name == "backbone":
                    continue
                if os.path.exists(path):
                    available_swappers.append(name)
    return available_swappers

NUMBER_OF_IDS = 10
FACE_IDS = [str(i) for i in range(1, NUMBER_OF_IDS + 1)]

DEFAULT_OUTPUT_DIRECTORY = os.getcwd()

MASK_REGIONS_DEFAULT = default_face_regions
MASK_REGIONS = list(bisenet_face_regions.keys())

NSFW_DETECTOR = None

FACE_SWAPPER_LIST = get_available_swapper_names()
DEFAULT_SWAPPER = FACE_SWAPPER_LIST[0]

FACE_ENHANCER_LIST = list(get_available_upscaler_names())

MAX_THREADS = 4

VIDEO_FPS_LIST = ["Original", "10", "15", "24", "25", "29.97", "30", "50", "59.94", "60", "120"]
VIDEO_CODEC_LIST = ["libx264", "libx265", "h264_nvec", "hevc_nvenc"]
VIDEO_RESOLUTION_LIST = ["Original", "240p", "360p", "480p", "540p", "720p", "1080p", "1440p", "2160p"]

AVERAGING_METHODS = ["mean", "median"]
AVERAGING_METHOD = "mean"

ALIGNMENT_MODES = ["auto", "generic", "arcface", "mtcnn", "ffhq", "set1", "set2", "set3",]

FILTER_TYPES = ["By First Detected", "By Condition", "By Specific Face"]
POSITION_MODES = ["Any", "Left Most", "Right Most", "Top Most", "Bottom Most", "Center Most", "Biggest", "Smallest"]
AGE_MODES = ["Any", "Child", "Teen", "Adult", "Senior", "Youngest", "Eldest"]
GENDER_MODES = ["Any", "Male", "Female"]

BLEND_MODES = ["alpha blend", "laplacian", "poisson"]

SOURCE_FILE_TYPES = ["Image"]

DFM_MODELS = {}

WRITE_JSON = False

INTERPOLATION_MAP = OrderedDict([
    ("area", cv2.INTER_AREA),
    ("nearest", cv2.INTER_NEAREST),
    ("linear", cv2.INTER_LINEAR),
    ("cubic", cv2.INTER_CUBIC),
    ("lanczos4", cv2.INTER_LANCZOS4)
])

FRAME_CONTROL_DEFAULTS = OrderedDict([
    ('frame_upscale', False),
    ('keep_original_resolution', True),
])

FACE_CONTROL_DEFAULTS = OrderedDict([
    ('pre_blur_amount', 0),
    ('pre_sharpen_amount', 0),
    ('pre_sharpen_size', 2),
    ('whole_process_iteration', 1),
    ('source_forehead_influence', 1),

    ('pre_upscale', False),
    ('post_upscale', False),
    ('face_upscaler_opacity', 1.0),
    ('default_interpolation', "linear"),

    ('face_parse_from_source', False),
    ('face_parse_from_target', False),
    ('face_parse_invert', False),
    ('face_parse_erode', 0),
    ('mask_regions', MASK_REGIONS_DEFAULT),

    ('mask_erode', 0),
    ('mask_blur', 0),
    ('mask_smooth_radius', 0),
    ('mask_smooth_iteration', 0),

    ('crop_top', 10),
    ('crop_bott', 100),
    ('crop_left', 0),
    ('crop_right', 100),
    ('border_fade_amount', 20),

    ('use_color_match', False),
    ('blending_mode', BLEND_MODES[0]),

    ('median_blur', 0),
    ('jpeg_compression', 100),
    ('downsample', 1),

    ('occlusion_from_source', False),
    ('occlusion_from_target', False),

    ('dfm_src_mask', True),
    ('dfm_celeb_mask', True),

    ('landmark_scale', 1),
    ('face_alignment_method', ALIGNMENT_MODES[0]),
    ('re_align', False),
])

SETTINGS_DEFAULTS = OrderedDict([
    ('face_detection_size', 640),
    ('face_detection_threshold', 0.6),
    ('face_averaging_method', AVERAGING_METHODS[0]),
    ('face_similarity_threshold', 0.65),
    ('face_swapper_name', FACE_SWAPPER_LIST[0]),
    ('dfm_model_directory', dp.DFM_MODELS_DIRECTORY),
    ('occluder_model_path', dp.OCCLUDER_MODEL_PATH),
    ('face_upscaler_name', FACE_ENHANCER_LIST[0])
])

SOURCE_DEFAULTS = OrderedDict([
    ('source_type', "Image"),
    ('source_image_input', None),
    ('source_dfm_input', None),
    ('specific_image_input', None),
])

TARGET_SELECTION_DEFAULTS = OrderedDict([
    ('filter_type', 'By First Detected'),
    ('filter_condition_position', 'Any'),
    ('filter_condition_age_group', 'Any'),
    ('filter_condition_gender', 'Any'),
])

TARGET_DEFAULTS = OrderedDict([
    ('target_type', 'Image'),
    ('target_image_input', None),
    ('target_video_input', None),
    ('target_directory_input', None),
    ('target_video_start_frame', 0),
    ('target_video_end_frame', 0)
])

OUTPUT_DEFAULTS = OrderedDict([
    ('output_directory', DEFAULT_OUTPUT_DIRECTORY),
    ('output_name', 'Result'),
    ('use_datetime_suffix', True),

    ('extract_format', "png"),
    ('extract_quality', 100),
    ('keep_extracted_sequence', False),

    ('video_container', ".mp4"),
    ('video_fps', "Original"),
    ('video_resolution', 'Original'),

    ('video_quality', 100),
    ('video_codec', 'libx264'),
    ('merge_audio', True)

])
