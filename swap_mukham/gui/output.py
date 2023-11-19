import gradio as gr
import swap_mukham.default_paths as dp
import swap_mukham.global_variables as gv
from swap_mukham.gui.custom_components import create_text_and_browse_button

def create_output_components(cmpts):
    cmpts.output_directory = create_text_and_browse_button(
        "Output Directory",
        "Browse",
        mode="directory",
        value=gv.OUTPUT_DEFAULTS['output_directory'],
        interactive=True
    )

    with gr.Group():
        cmpts.output_name = gr.Text(
            label="Output Name", value=gv.OUTPUT_DEFAULTS['output_name'], interactive=True
        )
        cmpts.use_datetime_suffix = gr.Checkbox(
            label="Suffix date-time", value=gv.OUTPUT_DEFAULTS['use_datetime_suffix'], interactive=True
        )

    with gr.Accordion("Video settings", open=True):
        with gr.Group():
            with gr.Row():
                cmpts.extract_format = gr.Dropdown(
                    ["jpg", "png"],
                    label="Extract format",
                    value=gv.OUTPUT_DEFAULTS['extract_format'],
                    interactive=True,
                )
                cmpts.extract_quality = gr.Slider(
                    label="Extract quality",
                    value=gv.OUTPUT_DEFAULTS['extract_quality'],
                    minimum=1,
                    maximum=100,
                    step=1
                )
            cmpts.keep_extracted_sequence = gr.Checkbox(
                label="Keep extracted sequence",
                value=gv.OUTPUT_DEFAULTS['keep_extracted_sequence'],
                interactive=True,
            )

        with gr.Group():
            with gr.Row():
                cmpts.video_container = gr.Dropdown(
                    [".mp4", ".gif"],
                    label="Container",
                    value=gv.OUTPUT_DEFAULTS['video_container'],
                    interactive=True,
                )

                cmpts.video_fps = gr.Dropdown(
                    gv.VIDEO_FPS_LIST,
                    label="Fps",
                    value=gv.OUTPUT_DEFAULTS['video_fps'],
                    interactive=True,
                )

                cmpts.video_resolution = gr.Dropdown(
                    gv.VIDEO_RESOLUTION_LIST,
                    label="Resolution",
                    value=gv.OUTPUT_DEFAULTS['video_resolution'],
                    interactive=True,
                )

        with gr.Group() as mp4_group:
            with gr.Row():
                cmpts.video_codec = gr.Dropdown(
                    gv.VIDEO_CODEC_LIST,
                    label="Codec",
                    value=gv.OUTPUT_DEFAULTS['video_codec'],
                    interactive=True,
                )
                cmpts.video_quality = gr.Slider(
                        label="Encode quality",
                        value=gv.OUTPUT_DEFAULTS['video_quality'],
                        minimum=1,
                        maximum=100,
                        step=1
                    )
            cmpts.merge_audio = gr.Checkbox(
                label="Merge Audio", value=gv.OUTPUT_DEFAULTS['merge_audio'], interactive=True
            )

    cmpts.output_components = [
        cmpts.output_directory,
        cmpts.output_name,
        cmpts.use_datetime_suffix,

        cmpts.extract_format,
        cmpts.extract_quality,
        cmpts.keep_extracted_sequence,

        cmpts.video_container,
        cmpts.video_fps,
        cmpts.video_resolution,

        cmpts.video_quality,
        cmpts.video_codec,
        cmpts.merge_audio
    ]