import os
import gradio as gr
from collections import OrderedDict
from swap_mukham.utils import io as io_util
from swap_mukham import global_variables as gv
from swap_mukham.gui.custom_components import create_text_and_browse_button

def create_settings_inputs(cmpts, SM):
    with gr.Accordion("Detection", open=False):
        cmpts.face_detection_size = gr.Number(
            label="Detection Size",
            value=gv.SETTINGS_DEFAULTS['face_detection_size'],
            interactive=True,
        )
        cmpts.face_detection_threshold = gr.Slider(
            label="Detection Threshold",
            minimum=0,
            maximum=1,
            step=0.01,
            value=gv.SETTINGS_DEFAULTS['face_detection_threshold'],
            interactive=True,
        )

    with gr.Accordion("Embedding", open=False):
        cmpts.face_averaging_method = gr.Dropdown(
            gv.AVERAGING_METHODS,
            label="Multi-Face Averaging Method",
            value=gv.SETTINGS_DEFAULTS['face_averaging_method'],
            interactive=True,
        )
        cmpts.face_similarity_threshold = gr.Slider(
            minimum=0,
            maximum=2,
            value=gv.SETTINGS_DEFAULTS['face_similarity_threshold'],
            interactive=True,
            label="Specific-Target Distance",
        )

    with gr.Accordion("Swapper", open=False):
        cmpts.face_swapper_name = gr.Dropdown(
            gv.FACE_SWAPPER_LIST,
            label="Name",
            value=gv.SETTINGS_DEFAULTS['face_swapper_name'],
            multiselect=False,
            interactive=True,
        )

    with gr.Accordion("DFM", open=False):
        cmpts.dfm_model_directory = create_text_and_browse_button(
            "Model Directory",
            "Browse",
            mode="directory",
            value=gv.SETTINGS_DEFAULTS['dfm_model_directory'],
            interactive=True,
        )
        with gr.Row():
            cmpts.refresh_dfm_model_directory = gr.Button("Fetch models", variant="secondary")
            cmpts.delete_all_dfm_sessions = gr.Button("Clear all DFM sessions", variant="secondary")

    with gr.Accordion("Occluder", open=False):
        cmpts.occluder_model_path = create_text_and_browse_button(
            "Model Path",
            "Browse",
            mode="file",
            value=gv.SETTINGS_DEFAULTS['occluder_model_path'],
            interactive=True,
        )

    with gr.Accordion("Face Upscaler", open=False):
        cmpts.face_upscaler_name = gr.Dropdown(
            gv.FACE_ENHANCER_LIST,
            label="Name",
            value=gv.SETTINGS_DEFAULTS['face_upscaler_name'],
            multiselect=False,
            interactive=True,
        )

    # order should match global_variables.SETTINGS_DEFAULTS
    cmpts.settings_components = [
        cmpts.face_detection_size,
        cmpts.face_detection_threshold,
        cmpts.face_averaging_method,
        cmpts.face_similarity_threshold,
        cmpts.face_swapper_name,
        cmpts.dfm_model_directory,
        cmpts.occluder_model_path,
        cmpts.face_upscaler_name,
    ]

    def update_dfm_list(dfm_model_directory):
        dfm_paths, dfm_names = io_util.get_files_by_extension(directory, ".dfm")
        gv.DFM_MODELS = dict(zip(dfm_names, dfm_paths))
        gr.Info("DFM model path updated")
        return gr.Dropdown(choices=list(dfm_names))


    cmpts.refresh_dfm_model_directory.click(
        update_dfm_list,
        [cmpts.dfm_model_directory],
        [cmpts.source_dfm_input],
    )

    def del_dfm_sessions():
        SM.delete_all_dfm_sessions()
        gr.Info("Cleared DFM sessions")


    cmpts.delete_all_dfm_sessions.click(
        fn=del_dfm_sessions,
    )
