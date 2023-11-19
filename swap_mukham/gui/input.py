import gradio as gr
import datetime
from collections import OrderedDict
import swap_mukham.global_variables as gv
from swap_mukham.gui.custom_components import create_multi_image_input, create_text_and_browse_button

def create_source_inputs(cmpts):
    with gr.Column():
        with gr.Group():
            cmpts.source_type = gr.Dropdown(
                ["Image", "DFM"],
                label="Source Type",
                value=gv.SOURCE_DEFAULTS['source_type'],
                multiselect=False,
                interactive=True,
            )
            with gr.Column():
                with gr.Column():
                    with gr.Row():
                        with gr.Row() as image_container:
                            cmpts.source_image_input = create_multi_image_input(default_values=gv.SOURCE_DEFAULTS['source_image_input'], label="Source Face")
                        with gr.Row(visible=False) as dfm_container:
                            cmpts.source_dfm_input = gr.Dropdown(list(gv.DFM_MODELS.keys()), label="Deepface live models", value=gv.SOURCE_DEFAULTS['source_dfm_input'])
                        with gr.Row(visible=False) as specific_row:
                            cmpts.specific_image_input = create_multi_image_input(default_values=gv.SOURCE_DEFAULTS['specific_image_input'], label="Specific Face")
                            cmpts.specific_row = specific_row

        # order should match global_variables.SOURCE_DEFAULTS
        cmpts.source_components = [
            cmpts.source_type,
            cmpts.source_image_input,
            cmpts.source_dfm_input,
            cmpts.specific_image_input,
        ]

    def source_container_visibility(source_type):
        if source_type == "Image":
            return gr.Row(visible=True), gr.Row(visible=False)
        elif source_type == "DFM":
            return gr.Row(visible=False), gr.Row(visible=True)
        else:
            return gr.Row(visible=False), gr.Row(visible=False)

    cmpts.source_type.change(
        source_container_visibility,
        [cmpts.source_type,],
        [image_container, dfm_container]
    )

    return


def create_filter_components(cmpts):
    with gr.Column():
        with gr.Group():
            cmpts.filter_type = gr.Dropdown(
                gv.FILTER_TYPES,
                label="Target Face Selection",
                value=gv.TARGET_SELECTION_DEFAULTS['filter_type'],
                multiselect=False,
                interactive=True,
            )

            with gr.Row(visible=False) as condition_row:
                cmpts.filter_condition_position = gr.Dropdown(gv.POSITION_MODES, value=gv.TARGET_SELECTION_DEFAULTS['filter_condition_position'], label="Position", interactive=True)
                cmpts.filter_condition_age_group = gr.Dropdown(gv.AGE_MODES, value=gv.TARGET_SELECTION_DEFAULTS['filter_condition_age_group'], label="Age", interactive=True)
                cmpts.filter_condition_gender = gr.Dropdown(gv.GENDER_MODES, value=gv.TARGET_SELECTION_DEFAULTS['filter_condition_gender'], label="Gender", interactive=True)

    # order should match global_variables.TARGET_SELECTION_DEFAULTS
    cmpts.target_selection_components = [
        cmpts.filter_type,
        cmpts.filter_condition_position,
        cmpts.filter_condition_age_group,
        cmpts.filter_condition_gender,
    ]

    def condition_container_visibility(filter_type):
        if filter_type == "By First Detected":
            return gr.Row(visible=False), gr.Row(visible=False)
        if filter_type == "By Condition":
            return gr.Row(visible=True), gr.Row(visible=False)
        if filter_type == "By Specific Face":
            return gr.Row(visible=False), gr.Row(visible=True)


    cmpts.filter_type.input(
        condition_container_visibility,
        [cmpts.filter_type,],
        [condition_row, cmpts.specific_row]
    )


# predefine and render in preview region
start_frame = gr.Number(label="Start Frame", step=1, container=False, interactive=False)
end_frame = gr.Number(label="End Frame", step=1, container=False, interactive=False)

def create_target_inputs(cmpts, local=False):
    target_types = ["Image", "Video"]
    if local:
        target_types.append("Directory")

    with gr.Column():
        with gr.Group():
            cmpts.target_type = gr.Dropdown(
                target_types,
                label="Target Type",
                value="Image",
                multiselect=False,
                interactive=True,
            )
            if local:
                with gr.Row(visible=True) as image_container:
                   cmpts.target_image_input = create_text_and_browse_button("Target Image", "Browse", show_gallery="IMAGE", mode="file", interactive=True)
                with gr.Row(visible=False) as video_container:
                    cmpts.target_video_input = create_text_and_browse_button("Target Video", "Browse", show_gallery="VIDEO", mode="file", interactive=True)
            else:
                with gr.Row(visible=True) as image_container:
                    cmpts.target_image_input = gr.Image(label="Target Image", type='filepath')
                with gr.Row(visible=False) as video_container:
                    cmpts.target_video_input = gr.Video(label="Target Video")
            with gr.Row(visible=False) as directory_container:
                cmpts.target_directory_input = create_text_and_browse_button("Target Directory", "Browse", mode="directory", interactive=True)


    def visibility(target_type):
        if target_type == "Image":
            return gr.Row(visible=True), gr.Row(visible=False), gr.Row(visible=False)
        elif target_type == "Video":
            return gr.Row(visible=False), gr.Row(visible=True), gr.Row(visible=False)
        elif target_type == "Directory":
            return gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=True)

    cmpts.target_type.change(
        visibility,
        [cmpts.target_type,],
        [image_container, video_container, directory_container]
    )

    cmpts.target_video_start_frame = start_frame
    cmpts.target_video_end_frame = end_frame

    cmpts.target_components = [
        cmpts.target_type,
        cmpts.target_image_input,
        cmpts.target_video_input,
        cmpts.target_directory_input,
        cmpts.target_video_start_frame,
        cmpts.target_video_end_frame,
    ]

    return


def set_target_updates(cmpts):
    cmpts.target_image_input.upload(
        lambda x: x,
        [cmpts.target_image_input],
        [cmpts.preview_image]
    )
