import gradio as gr
from collections import OrderedDict
from swap_mukham import global_variables as gv


def create_face_control_inputs(cmpts):
    # Alignment
    with gr.Tab("Alignment"):
        with gr.Row():
            face_alignment_method = gr.Dropdown(
                gv.ALIGNMENT_MODES,
                value=gv.FACE_CONTROL_DEFAULTS['face_alignment_method'],
                multiselect=False,
                label="Method",
                interactive=True,
            )
            landmark_scale = gr.Slider(
                    label="Scale",
                    minimum=0,
                    maximum=2,
                    value=gv.FACE_CONTROL_DEFAULTS['landmark_scale'],
                    step=0.001,
                    interactive=True,
                )
        re_align = gr.Checkbox(
            label="Re-align with FFHQ",
            value=gv.FACE_CONTROL_DEFAULTS['re_align'],
            interactive=True,
        )

    # Enhance
    with gr.Tab("Enhance"):
        with gr.Column(variant="panel"):
            whole_process_iteration = gr.Slider(
                    label="Whole Process Iteration",
                    minimum=1,
                    maximum=4,
                    value=gv.FACE_CONTROL_DEFAULTS['whole_process_iteration'],
                    step=1,
                    interactive=True,
                )

        with gr.Column():
            source_forehead_influence = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.001,
                    value=gv.FACE_CONTROL_DEFAULTS['source_forehead_influence'],
                    interactive=True,
                    label="Source Forehead Influence",
                )

        with gr.Column():
            pre_blur_amount = gr.Slider(
                    label="Pre Blur",
                    minimum=0,
                    maximum=20,
                    value=gv.FACE_CONTROL_DEFAULTS['pre_blur_amount'],
                    step=0.001,
                    interactive=True,
                )
            pre_sharpen_amount = gr.Slider(
                label="Pre Sharpen",
                minimum=0,
                maximum=20,
                value=gv.FACE_CONTROL_DEFAULTS['pre_sharpen_amount'],
                step=0.001,
                interactive=True,
            )
            pre_sharpen_size = gr.Slider(
                label="Pre Sharpen Size",
                minimum=1,
                maximum=20,
                value=gv.FACE_CONTROL_DEFAULTS['pre_sharpen_size'],
                step=0.001,
                interactive=True,
                visible=False
                )

            with gr.Group():
                with gr.Row():
                    pre_upscale = gr.Checkbox(
                        label="Pre Upscale",
                        value=gv.FACE_CONTROL_DEFAULTS['pre_upscale'],
                        interactive=True,
                    )
                    post_upscale = gr.Checkbox(
                        label="Post Upscale",
                        value=gv.FACE_CONTROL_DEFAULTS['post_upscale'],
                        interactive=True,
                    )
                face_upscaler_opacity = gr.Slider(
                    label="Face Upscaler Opacity",
                    minimum=0,
                    maximum=1,
                    value=gv.FACE_CONTROL_DEFAULTS['face_upscaler_opacity'],
                    step=0.001,
                    interactive=True,
                )
                default_interpolation = gr.Dropdown(
                    list(gv.INTERPOLATION_MAP.keys()),
                    value=gv.FACE_CONTROL_DEFAULTS['default_interpolation'],
                    label="Default Interpolation",
                    multiselect=False,
                    interactive=True,
                )

    # Mask
    with gr.Tab("Mask"):
        with gr.Accordion("Face Parsing", open=False):
                with gr.Group():
                    with gr.Row():
                        face_parse_from_source = gr.Checkbox(
                            label="Parse from source",
                            value=gv.FACE_CONTROL_DEFAULTS['face_parse_from_source'],
                            interactive=True,
                        )
                        face_parse_from_target = gr.Checkbox(
                            label="Parse from target",
                            value=gv.FACE_CONTROL_DEFAULTS['face_parse_from_target'],
                            interactive=True,
                        )
                        face_parse_invert = gr.Checkbox(
                            label="Invert",
                            value=gv.FACE_CONTROL_DEFAULTS['face_parse_invert'],
                            interactive=True,
                        )
                    mask_regions = gr.Dropdown(
                        gv.MASK_REGIONS,
                        value=gv.FACE_CONTROL_DEFAULTS['mask_regions'],
                        label="Face Regions",
                        multiselect=True,
                        interactive=True,
                    )
                    face_parse_erode = gr.Slider(
                        label="Erode",
                        minimum=-500,
                        maximum=500,
                        value=gv.FACE_CONTROL_DEFAULTS['face_parse_erode'],
                        step=1,
                        interactive=True,
                    )

        with gr.Accordion("Face Occlusion", open=False):
            with gr.Group():
                with gr.Row():
                    occlusion_from_source = gr.Checkbox(
                        label="From source",
                        value=gv.FACE_CONTROL_DEFAULTS['occlusion_from_source'],
                        interactive=True,
                    )
                    occlusion_from_target = gr.Checkbox(
                        label="From target",
                        value=gv.FACE_CONTROL_DEFAULTS['occlusion_from_target'],
                        interactive=True,
                    )

        with gr.Accordion("DFM Mask", open=False):
            with gr.Group():
                with gr.Row():
                    dfm_src_mask = gr.Checkbox(
                            label="SRC Mask",
                            value=gv.FACE_CONTROL_DEFAULTS['dfm_src_mask'],
                            interactive=True,
                        )
                    dfm_celeb_mask = gr.Checkbox(
                            label="CELEB Mask",
                            value=gv.FACE_CONTROL_DEFAULTS['dfm_celeb_mask'],
                            interactive=True,
                        )

        with gr.Accordion("Mask Filter", open=False):
            with gr.Group():
                with gr.Row():
                    mask_smooth_radius = gr.Slider(
                        label="Smooth Radius",
                        minimum=0,
                        maximum=100,
                        value=gv.FACE_CONTROL_DEFAULTS['mask_smooth_radius'],
                        step=1,
                        interactive=True,
                    )

                    mask_smooth_iteration = gr.Slider(
                        label="Smooth Iteration",
                        minimum=0,
                        maximum=100,
                        value=gv.FACE_CONTROL_DEFAULTS['mask_smooth_iteration'],
                        step=1,
                        interactive=True,
                    )
                with gr.Row():
                    mask_erode = gr.Slider(
                        label="Erode",
                        minimum=0,
                        maximum=500,
                        value=gv.FACE_CONTROL_DEFAULTS['mask_erode'],
                        step=1,
                        interactive=True,
                    )

                    mask_blur = gr.Slider(
                        label="Blur",
                        minimum=0,
                        maximum=500,
                        value=gv.FACE_CONTROL_DEFAULTS['mask_blur'],
                        step=1,
                        interactive=True,
                    )

        with gr.Accordion("Border Crop", open=False):
            with gr.Group():
                with gr.Row():
                    crop_top = gr.Slider(
                        label="Top",
                        minimum=0,
                        maximum=99,
                        value=gv.FACE_CONTROL_DEFAULTS['crop_top'],
                        step=1,
                        interactive=True,
                    )
                    crop_bott = gr.Slider(
                        label="Bottom",
                        minimum=1,
                        maximum=100,
                        value=gv.FACE_CONTROL_DEFAULTS['crop_bott'],
                        step=1,
                        interactive=True,
                    )
                with gr.Row():
                    crop_left = gr.Slider(
                        label="Left",
                        minimum=0,
                        maximum=99,
                        value=gv.FACE_CONTROL_DEFAULTS['crop_left'],
                        step=1,
                        interactive=True,
                    )
                    crop_right = gr.Slider(
                        label="Right",
                        minimum=1,
                        maximum=100,
                        value=gv.FACE_CONTROL_DEFAULTS['crop_right'],
                        step=1,
                        interactive=True,
                    )

        with gr.Row():
            border_fade_amount = gr.Slider(
                label="Border Fade",
                minimum=0,
                maximum=100,
                value=gv.FACE_CONTROL_DEFAULTS['border_fade_amount'],
                step=1,
                interactive=True,
            )

        with gr.Column():
            blending_mode = gr.Dropdown(
                gv.BLEND_MODES,
                value=gv.FACE_CONTROL_DEFAULTS['blending_mode'],
                multiselect=False,
                label="Blend Mode",
                interactive=True,
            )

            use_color_match = gr.Checkbox(
                label="Color Match",
                value=gv.FACE_CONTROL_DEFAULTS['use_color_match'],
                interactive=True,
                visible=False, # TODO
            )

    # Degrade
    with gr.Tab("Degrade"):
        with gr.Column():
            median_blur = gr.Slider(
                label="Median Blur",
                minimum=0,
                maximum=100,
                value=gv.FACE_CONTROL_DEFAULTS['median_blur'],
                step=1,
                interactive=True,
            )

            jpeg_compression = gr.Slider(
                label="Compression",
                minimum=1,
                maximum=100,
                value=gv.FACE_CONTROL_DEFAULTS['jpeg_compression'],
                step=1,
                interactive=True,
            )

            downsample = gr.Slider(
                label="Downsample",
                minimum=0,
                maximum=1,
                value=gv.FACE_CONTROL_DEFAULTS['downsample'],
                step=0.001,
                interactive=True,
            )

    # order should match global_variables.FACE_CONTROL_DEFAULTS
    cmpts.face_control_components = [
        pre_blur_amount,
        pre_sharpen_amount,
        pre_sharpen_size,
        whole_process_iteration,
        source_forehead_influence,
        pre_upscale,
        post_upscale,
        face_upscaler_opacity,
        default_interpolation,
        face_parse_from_source,
        face_parse_from_target,
        face_parse_invert,
        face_parse_erode,
        mask_regions,
        mask_erode,
        mask_blur,
        mask_smooth_radius,
        mask_smooth_iteration,
        crop_top,
        crop_bott,
        crop_left,
        crop_right,
        border_fade_amount,
        use_color_match,
        blending_mode,
        median_blur,
        jpeg_compression,
        downsample,
        occlusion_from_source,
        occlusion_from_target,
        dfm_src_mask,
        dfm_celeb_mask,
        landmark_scale,
        face_alignment_method,
        re_align,
    ]

def create_frame_control_inputs(cmpts):
    with gr.Group():
        frame_upscale = gr.Checkbox(
            label="Upscale Frame",
            value=gv.FRAME_CONTROL_DEFAULTS['frame_upscale'],
            interactive=True,
        )
        keep_original_resolution = gr.Checkbox(
                label="Keep Original Resolution",
                value=gv.FRAME_CONTROL_DEFAULTS['keep_original_resolution'],
                interactive=True,
            )

    cmpts.frame_control_components = [
        frame_upscale,
        keep_original_resolution
    ]

