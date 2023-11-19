import os

# download custom ffmpeg binary if needed
ffmpeg_binary_path = "./assets/ffmpeg/ffmpeg"
FFMPEG_PATH = ffmpeg_binary_path if os.path.exists(ffmpeg_binary_path) else "ffmpeg"

RETINAFACE_PATH = "./assets/pretrained_models/det_10g.onnx"
DETECTOR_PATH = "./assets/pretrained_models/detector.onnx"
GENDERAGE_PATH = "./assets/pretrained_models/gender_age.onnx"

OCCLUDER_MODEL_PATH = "./assets/pretrained_models/mask/occluder.onnx"
FACE_PARSER_PATH = "./assets/pretrained_models/mask/faceparser.onnx"

DFM_MODELS_DIRECTORY = "./assets/pretrained_models/dfm/"

# see: swap_mukham/models/swapper/init.py
EMBEDDING_BASED_SWAPPERS = {
    "inswapper": {
        "backbone": "./assets/pretrained_models/swapper/w600k_r50.onnx",
        "Inswapper 128": "./assets/pretrained_models/swapper/inswapper_128.onnx",
    },

    "simswap": {
        "backbone": "./assets/pretrained_models/swapper/simswap_arcface_backbone.onnx",
        "Simswap": "./assets/pretrained_models/swapper/simswap.onnx",
        "Simswap 224": "./assets/pretrained_models/swapper/simswap_224.onnx",
        "Simswap 256": "./assets/pretrained_models/swapper/simswap_256_fix.onnx",
        "Simswap 512 beta": "./assets/pretrained_models/swapper/simswap_512_beta.onnx",
        "Simswap 512 beta netr": "/mnt/B468F0EF68F0B0EA/AI_Tools/facefusion/.assets/models/simswap_512_beta_netr.onnx",
        "Simswap 512 unoff": "/mnt/B468F0EF68F0B0EA/AI_Tools/facefusion/.assets/models/simswap_512_unoff.onnx"
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
        "GFPGAN 1.4" : "./assets/pretrained_models/upscalers/GFPGANv1.4.onnx",
        "GFPGAN 1.3" : "./assets/pretrained_models/upscalers/GFPGANv1.3.onnx",
        "GFPGAN 1.2" : "./assets/pretrained_models/upscalers/GFPGANv1.2.onnx",

        "GPEN 512" : "./assets/pretrained_models/upscalers/GPEN-BFR-512.onnx",
        "GPEN 256" : "./assets/pretrained_models/upscalers/GPEN-BFR-256.onnx",

        "CodeFormer" : "./assets/pretrained_models/upscalers/codeformer.onnx",

        "RestoreFormer" : "./assets/pretrained_models/upscalers/restoreformer.onnx",
    },

    "generic": {
        "BSRGAN" : "./assets/pretrained_models/upscalers/bsrgan.onnx",

        "Realesrgan 2x" : "./assets/pretrained_models/upscalers/realesrganx2.onnx",
        "Realesrgan 4x" : "./assets/pretrained_models/upscalers/realesrganx4.onnx",
        "Realesrgan 8x" : "./assets/pretrained_models/upscalers/realesrganx8.onnx",

        "4x LSDIRCompact3_fp32" : "./assets/pretrained_models/upscalers/4xLSDIRCompact3_fp32.onnx",
        "4xLSDIRCompactC3_fp32" : "./assets/pretrained_models/upscalers/4xLSDIRCompactC3_fp32.onnx",
        "4xLSDIRCompactN3_fp32" : "./assets/pretrained_models/upscalers/4xLSDIRCompactN3_fp32.onnx",
        "4xLSDIRCompactR3_fp32" : "./assets/pretrained_models/upscalers/4xLSDIRCompactR3_fp32.onnx",

        "tghqface8x_opt.onnx" : "./assets/pretrained_models/face_upscalers/tghqface8x_opt.onnx",
    },
}
