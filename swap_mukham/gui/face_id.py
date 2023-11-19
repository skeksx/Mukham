import gradio as gr
from swap_mukham import global_variables as gv


def create_face_id_components(cmpts):
    cmpts.face_id = gr.Dropdown(gv.FACE_IDS, value=gv.FACE_IDS[0], label="ID", interactive=True, visible=False)

    def visibility(x):
        return gr.Dropdown(visible = (x == "By Specific Face"))

    cmpts.filter_type.input(
        visibility,
        [cmpts.filter_type,],
        [cmpts.face_id]
    )


def create_control_id_components(cmpts):
    with gr.Row():
        cmpts.use_control_id = gr.Checkbox(label="Use ID", value=False, interactive=True, visible=False)
        cmpts.control_id = gr.Dropdown(gv.FACE_IDS, value=gv.FACE_IDS[0], label="ID", interactive=True, container=False, visible=False)

    def visibility(x):
        return (gr.Checkbox(visible = (x == "By Specific Face")), gr.Dropdown(visible = (x == "By Specific Face")))

    cmpts.filter_type.input(
        visibility,
        [cmpts.filter_type,],
        [cmpts.use_control_id, cmpts.control_id]
    )


def set_on_face_id_updates(cmpts):
    def update_source_input_components(face_id):
        output = cmpts.source_data.get(face_id, None)
        if output is None:
            raise ValueError("Face ID not found")
        return list(output.values())

    def update_face_control_components(control_id):
        output = cmpts.face_control_data.get(control_id, None)
        if output is None:
            raise ValueError("Face ID not found")
        return list(output.values())

    cmpts.face_id.input(
        update_source_input_components,
        [cmpts.face_id],
        [*cmpts.source_components]
    )
    cmpts.control_id.input(
        update_face_control_components,
        [cmpts.control_id,],
        [*cmpts.face_control_components],
    )