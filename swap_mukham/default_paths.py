import os

# download custom ffmpeg binary if needed
ffmpeg_binary_path = "./assets/ffmpeg/ffmpeg"
FFMPEG_PATH = ffmpeg_binary_path if os.path.exists(ffmpeg_binary_path) else "ffmpeg"

RETINAFACE_PATH = "./assets/pretrained_models/retinaface_10g.onnx"
DETECTOR_PATH = "./assets/pretrained_models/det.onnx"
GENDERAGE_PATH = "./assets/pretrained_models/gender_age.onnx"

OCCLUDER_MODEL_PATH = "./assets/pretrained_models/mask/occluder.onnx"
FACE_PARSER_PATH = "./assets/pretrained_models/mask/faceparser.onnx"

DFM_MODELS_DIRECTORY = "./assets/pretrained_models/dfm/"

# see: swap_mukham/models/swapper/init.py
EMBEDDING_BASED_SWAPPERS = {
    "inswapper": {
        "backbone": "./assets/pretrained_models/swapper/arcface_w600k_r50.onnx",
        "Inswapper 128": "./assets/pretrained_models/swapper/inswapper_128.onnx",
    },

    "simswap": {
        "backbone": "./assets/pretrained_models/swapper/simswap_arcface_backbone.onnx",
        "Simswap": "./assets/pretrained_models/swapper/simswap.onnx",
        "Simswap 256": "./assets/pretrained_models/swapper/simswap_256_fix.onnx",
        "Simswap 512 beta": "./assets/pretrained_models/swapper/simswap_512_beta.onnx",
    },

    "simswap unofficial": {
        "backbone": "./assets/pretrained_models/swapper/simswap_arcface_backbone.onnx",
        "Simswap 512 unofficial": "./assets/pretrained_models/swapper/simswap_512_unofficial.onnx",
    },

    "ghost": {
        "backbone": "./assets/pretrained_models/swapper/ghost_arcface_backbone.onnx",
        "Ghost unet-1-block": "./assets/pretrained_models/swapper/ghost_unet_1_block.onnx",
        "Ghost unet-2-block": "./assets/pretrained_models/swapper/ghost_unet_2_block.onnx",
        "Ghost unet-3-block": "./assets/pretrained_models/swapper/ghost_unet_3_block.onnx",
    },
}

UPSCALERS = {
    "face": {
        "GFPGAN 1.4" : "./assets/pretrained_models/upscaler/GFPGANv1.4.onnx",
        "GFPGAN 1.3" : "./assets/pretrained_models/upscaler/GFPGANv1.3.onnx",
        "GFPGAN 1.2" : "./assets/pretrained_models/upscaler/GFPGANv1.2.onnx",

        "GPEN 512" : "./assets/pretrained_models/upscaler/GPEN-BFR-512.onnx",
        "GPEN 256" : "./assets/pretrained_models/upscaler/GPEN-BFR-256.onnx",

        "CodeFormer" : "./assets/pretrained_models/upscaler/codeformer.onnx",

        "RestoreFormer" : "./assets/pretrained_models/upscaler/restoreformer.onnx",
    },

    "generic": {
        "BSRGAN" : "./assets/pretrained_models/upscaler/bsrgan.onnx",

        "Realesrgan 2x" : "./assets/pretrained_models/upscaler/realesrganx2.onnx",
        "Realesrgan 4x" : "./assets/pretrained_models/upscaler/realesrganx4.onnx",
        "Realesrgan 8x" : "./assets/pretrained_models/upscaler/realesrganx8.onnx",

        "4x LSDIRCompact3_fp32" : "./assets/pretrained_models/upscaler/4xLSDIRCompact3_fp32.onnx",
        "4xLSDIRCompactC3_fp32" : "./assets/pretrained_models/upscaler/4xLSDIRCompactC3_fp32.onnx",
        "4xLSDIRCompactN3_fp32" : "./assets/pretrained_models/upscaler/4xLSDIRCompactN3_fp32.onnx",
        "4xLSDIRCompactR3_fp32" : "./assets/pretrained_models/upscaler/4xLSDIRCompactR3_fp32.onnx",

        "tghqface8x_opt.onnx" : "./assets/pretrained_models/face_upscaler/tghqface8x_opt.onnx",
    },
}
