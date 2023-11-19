import gradio as gr
from swap_mukham.utils import io as io_util
from swap_mukham.utils import video as video_util


def create_preview_components(cmpts, sm):
    with gr.Column():
        with gr.Row():
            cmpts.enable_realtime_update = gr.Checkbox(label="Realtime Update", value=False)
            cmpts.preview_type = gr.Dropdown(
                ["Swap", "Original", "Mask", "Landmark"],
                value="Swap",
                show_label=False,
                container=False,
                interactive=True,
            )
            # cmpts.preview_play_button = gr.Button("Play") # TODO
            cmpts.collect_face_button = gr.Button("Collect Faces", variant="secondary")

        with gr.Group():
            cmpts.collected_faces = gr.Gallery(show_label=False, object_fit="cover", columns=10, height=120, allow_preview=False, visible=False)
            with gr.Column(visible=False, variant="panel") as frame_slider_group:
                cmpts.frame_slider = gr.Slider(label="Frame", maximum=1, minimum=0, step=1, interactive=False)
                with gr.Row():
                    cmpts.set_start_frame = gr.Button("Set Start")
                    cmpts.target_video_start_frame.render()
                    cmpts.target_video_end_frame.render()
                    cmpts.set_end_frame = gr.Button("Set End")
            cmpts.preview_image = gr.Image(show_label=False, interactive=False)

        def get_slider_pos(x):
            return x

        def visibility(x):
            return gr.Column(visible=(x=="Video" or x=="Directory"))


        cmpts.target_type.change(
            visibility,
            [cmpts.target_type],
            [frame_slider_group]
        )

        cmpts.set_start_frame.click(
            get_slider_pos,
            [cmpts.frame_slider],
            [cmpts.target_video_start_frame]
        )

        cmpts.set_end_frame.click(
            get_slider_pos,
            [cmpts.frame_slider],
            [cmpts.target_video_end_frame]
        )

        def set_slider_min_max(path, data_type):
            minimum, maximum = 0, 0
            interactive = False
            if path:
                if data_type == "Video":
                    total_frames = video_util.get_video_info(path)[0]
                    if total_frames:
                        maximum = total_frames - 1
                        interactive = True
                    else:
                        gr.Warning("Invalid video path")
                elif data_type == "Directory":
                    if io_util.is_valid_directory(path):
                        image_paths = io_util.get_images_from_directory(path)
                        maximum = len(image_paths) - 1
                        interactive = True
                    else:
                        gr.Warning("Invalid directory path")
            return gr.Slider(minimum=minimum, maximum=maximum, value=0, interactive=interactive), minimum, maximum

        cmpts.target_video_input.change(
            set_slider_min_max,
            [cmpts.target_video_input, cmpts.target_type],
            [cmpts.frame_slider, cmpts.target_video_start_frame, cmpts.target_video_end_frame]
        )

        cmpts.target_directory_input.change(
            set_slider_min_max,
            [cmpts.target_directory_input, cmpts.target_type],
            [cmpts.frame_slider, cmpts.target_video_start_frame, cmpts.target_video_end_frame]
        )

        def get_collected_faces(image):
            if image is not None:
                gr.Info(f"Collecting faces...")
                faces = sm.collect_heads(image)
                if len(faces):
                    return gr.Gallery(visible=True, value=faces)
            gr.Info(f"No Face found")
            return gr.Gallery(visible=False, value=None)

        cmpts.collect_face_button.click(get_collected_faces, inputs=[cmpts.preview_image], outputs=[cmpts.collected_faces])