import torch # can delete if your stuff is set up correctly
import os
import sys
import cv2
import json
import time
import datetime
import argparse
import numpy as np
import gradio as gr

from swap_mukham import gui
from swap_mukham.sm import SwapMukham
from swap_mukham.utils import io as io_util
from swap_mukham import default_paths as dp
from swap_mukham import global_variables as gv
from swap_mukham.gui import components as cmpts
from swap_mukham.utils import image as image_util
from swap_mukham.utils import video as video_util
from swap_mukham.utils import device as device_util
from swap_mukham.execution import ThreadedExecutor


## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Swap-Mukham Face Swapper")
parser.add_argument("--output_directory", help="Default output directory", default=os.getcwd())
parser.add_argument("--local", action="store_true", help="Gradio widgets modified for local machine", default=False)
parser.add_argument('--device', choices=device_util.available_onnx_devices, help=f'Select device {device_util.available_onnx_devices}')

user_args = parser.parse_args()

gv.CANCEL_SIGNAL = False
gv.DEFAULT_OUTPUT_DIRECTORY = user_args.output_directory
gv.OUTPUT_DIRECTORY = gv.DEFAULT_OUTPUT_DIRECTORY
gv.OUTPUT_FILE = None
dfm_paths, dfm_names = io_util.get_files_by_extension(dp.DFM_MODELS_DIRECTORY, ".dfm")
gv.DFM_MODELS = dict(zip(dfm_names, dfm_paths))



SM = SwapMukham(device=user_args.device)



with gr.Blocks(css=gui.theme.css, theme=gr.themes.Default(**gui.theme.theme)) as ui:
    gr.HTML(gui.theme.intro)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Input"):
                    gui.input.create_source_inputs(cmpts)
                    with gr.Group():
                        gui.input.create_filter_components(cmpts)
                        gui.face_id.create_face_id_components(cmpts)
                    gui.input.create_target_inputs(cmpts, local=user_args.local)

                with gr.TabItem("Face Controls"):
                    gui.face_id.create_control_id_components(cmpts)
                    gui.controls.create_face_control_inputs(cmpts)

                with gr.Row(visible=False):
                    with gr.TabItem("Frame Controls"): # TODO
                        gui.controls.create_frame_control_inputs(cmpts)

                with gr.TabItem("Output"):
                    gui.output.create_output_components(cmpts)

                with gr.TabItem("Settings"):
                    max_threads = gr.Number(label="Max Threads", value=4, step=1, interactive=True)
                    gui.settings.create_settings_inputs(cmpts, SM)

        with gr.Column(scale=2):
            with gr.Row():
                swap_button = gr.Button("🗿 Swap", variant="primary")
                cancel_button = gr.Button("⛔ Cancel", variant="secondary")
            with gr.Row(variant="panel"):
                gui.preview.create_preview_components(cmpts, SM)
            with gr.Row():
                open_output_directory_button = gr.Button("📂 Output Directory", size="sm")
                open_output_file_button = gr.Button("🎞️ Output File", size="sm")




    def make_preview(preview_type, frame, target_image, masks, keypointss):
        if frame is None:
            return gr.Image()
        if masks is None or keypointss is None:
            return frame
        if preview_type == "Swap":
            return frame
        elif preview_type == "Original":
            return target_image
        elif preview_type == "Mask":
            mask = np.maximum.reduce(masks)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            color_image = np.zeros_like(frame, dtype=np.uint8)
            color_image[:, :, 2] = 200
            frame = frame.astype('f')
            overlay = frame + (color_image - frame) * mask * 0.7
            return overlay
        elif preview_type == "Landmark":
            overlay = frame.copy()
            colors = [(255,0,0),(255,255,0),(0,0,255),(0,255,255),(0,255,0)]
            size = max(*frame.shape[:2])
            inner = max(int(size * 0.002), 2)
            outer = max(int(size * 0.004), 4)
            for keypoints in keypointss:
                for points, color in zip(keypoints, colors):
                    x, y = int(points[0]), int(points[1])
                    overlay = cv2.circle(overlay, (x,y), outer, (20,20,20), -1)
                    overlay = cv2.circle(overlay, (x,y), inner, color, -1)
            return overlay

    def process_frame(frame_path):
        if not gv.CANCEL_SIGNAL:
            frame = cv2.imread(frame_path)
            output = SM.process_frame(frame, None)[0]
            cv2.imwrite(frame_path, output)

    def main_process(is_final_render, current_face_id, current_control_id, use_control_id, frame_index, preview_type, is_realtime, max_threads):
        gv.CANCEL_SIGNAL = False
        main_process_start_time = time.time()
        calculate_execution_time = lambda s: str(datetime.timedelta(seconds=time.time() - s))

        source_data = cmpts.processed_source_data
        target_data = cmpts.processed_target_data
        filter_data = cmpts.processed_filter_data
        settings_data = cmpts.processed_settings_data
        output_data = cmpts.processed_output_data
        face_control_data, frame_control_data = cmpts.processed_control_data

        target_image = target_data['target_image_input']
        target_video = target_data['target_video_input']
        target_directory = target_data['target_directory_input']
        target_type = target_data['target_type']
        output_name = output_data['output_name']
        current_source_data = source_data[current_face_id]

        if not is_final_render:
            if target_data['target_type'] == "Image" and target_image:
                target_image = cv2.imread(target_image)
            elif target_data['target_type'] == "Video" and target_video:
                video_path = target_video.replace('"', "").strip()
                target_image = video_util.get_single_video_frame(video_path, frame_index, swap_rgb=True)
            elif target_data['target_type'] == "Directory" and target_directory:
                image_paths = io_util.get_images_from_directory(target_directory)
                target_image = cv2.imread(image_paths[frame_index])
            else:
                return None

            if is_realtime:
                if not current_source_data['source_image_input'] and not current_source_data['source_dfm_input']:
                    gr.Warning("Source face not provided")
                    return None
                if filter_data['filter_type'] == "by specific face" and not current_source_data["specific_image_input"]:
                    gr.Warning("Specific face not provided")
                    return None

                SM.use_control_id = use_control_id
                SM.current_control_id = current_control_id
                if not SM.detector.detect(target_image, "Image", gui=gr):return
                SM.set_attributes(filter_data, settings_data, face_control_data, frame_control_data)
                SM.analyse_source_faces(source_data, current_face_id)
                frame, masks, keypointss = SM.process_frame(target_image, None)
                yield make_preview(preview_type, frame, target_image, masks, keypointss)
            else:
                yield make_preview(preview_type, target_image, target_image, None, None)

        else:
            SM.use_control_id = use_control_id
            SM.current_control_id = current_control_id
            SM.set_attributes(filter_data, settings_data, face_control_data, frame_control_data)
            SM.analyse_source_faces(source_data, current_face_id)

            if output_data['use_datetime_suffix']:
                output_name = io_util.add_datetime_to_filename(output_data['output_name'])



            if target_data['target_type'] == "Image":
                target_image = cv2.imread(target_image)

                if not SM.detector.detect(target_image, target_type, gui=gr):return
                frame, masks, keypointss = SM.process_frame(target_image, None)
                output_file_path = os.path.join(output_data['output_directory'], output_name + ".jpg")
                cv2.imwrite(output_file_path, frame)

                gv.OUTPUT_DIRECTORY = output_data['output_directory']
                gv.OUTPUT_FILE = output_file_path

                print(f"Completed in {calculate_execution_time(main_process_start_time)}")
                yield make_preview(preview_type, frame, target_image, masks, keypointss)



            elif target_data['target_type'] == "Video":
                start_time = time.time()
                video_path = target_video.replace('"', "").strip()

                temp_path = os.path.join(output_data['output_directory'], output_name)
                os.makedirs(temp_path, exist_ok=True)
                total_frames, original_fps, resolution, codec = video_util.get_video_info(video_path)
                if not SM.detector.detect(video_path, target_type, gui=gr):return

                fps = original_fps if output_data['video_fps'] == "Original" else min(float(output_data['video_fps']), original_fps)
                custom_resolution = None if output_data['video_resolution'] == "Original" else int(output_data['video_resolution'].replace("p", ""))

                extract_process_start_time = time.time()
                extract_success, temp_frame_paths = video_util.ffmpeg_extract_frames(
                    video_path,
                    temp_path,
                    start_frame=target_data['target_video_start_frame'],
                    end_frame=target_data['target_video_end_frame'],
                    custom_fps=fps,
                    custom_resolution=custom_resolution,
                    quality=output_data['extract_quality'],
                    name=f"frame_%d.{output_data['extract_format']}",
                )
                if not extract_success:
                    gr.Info("Video extraction failed")
                    return
                extract_process_time = calculate_execution_time(extract_process_start_time)

                frame_process_start_time = time.time()
                TE = ThreadedExecutor(process_frame, temp_frame_paths)
                TE.run(threads=max_threads, text="Processing Frames")
                frame_process_time = calculate_execution_time(frame_process_start_time)

                if gv.CANCEL_SIGNAL:
                    if not output_data['keep_extracted_sequence']:
                        io_util.remove_directory(temp_path)
                    return

                merge_process_start_time = time.time()

                if output_data['video_container'] == ".gif":
                    gif_success, final_result = video_util.ffmpeg_merge_frames_to_gif(
                        temp_path,
                        f"frame_%d.{output_data['extract_format']}",
                        os.path.join(output_data['output_directory'], output_name + ".gif"),
                        fps=fps,
                    )
                    if not gif_success:
                        gr.Info("GIF conversion failed!")
                else:
                    merge_success, merged_video = video_util.ffmpeg_merge_frames(
                        temp_path,
                        f"frame_%d.{output_data['extract_format']}",
                        os.path.join(output_data['output_directory'], output_name + "_without_audio" + ".mp4"),
                        fps=fps,
                        codec=output_data['video_codec'],
                        quality=output_data['video_quality'],
                    )

                    if not merge_success:
                        gr.Info("Merging Failed!")

                    if output_data['merge_audio'] and merge_success:
                        merge_success, merged_video_audio = video_util.ffmpeg_mux_audio(
                            video_path,
                            merged_video,
                            os.path.join(output_data['output_directory'], output_name + ".mp4"),
                            start_frame=target_data['target_video_start_frame'],
                            end_frame=target_data['target_video_end_frame'],
                        )
                        if merge_success:
                            final_result = merged_video_audio
                            io_util.remove_file(merged_video)
                        else:
                            gr.Info("Merging Audio Failed!")
                            final_result = merged_video
                    else:
                        final_result = merged_video

                if not output_data['keep_extracted_sequence']:
                    io_util.remove_directory(temp_path)

                gv.OUTPUT_DIRECTORY = output_data['output_directory']
                gv.OUTPUT_FILE = final_result

                merge_process_time = calculate_execution_time(merge_process_start_time)
                total_process_frames = len(temp_frame_paths)
                relative_fps = total_process_frames / (time.time() - main_process_start_time)
                gr.Info("Completed")

                print("  ---------------------------------SUMMARY----------------------------------  ")
                print(f"    Total Frames         : {total_process_frames}")
                print(f"    Threads              : {int(max_threads)}")
                print("  --------------------------------------------------------------------------  ")
                print(f"    Extraction Time      : {extract_process_time}")
                print(f"    Processing Time      : {frame_process_time}")
                print(f"    Merging Time         : {merge_process_time}")
                print(f"    Total Execution Time : {calculate_execution_time(main_process_start_time)}")
                print(f"    Relative FPS         : {relative_fps:.4f}")
                print("  --------------------------------------------------------------------------  ")
                print(f"    Output Path          : {final_result}")
                print("  --------------------------------------------------------------------------  ")



            elif target_data['target_type'] == "Directory":
                image_paths = io_util.get_images_from_directory(target_directory)
                slice_start = int(target_data['target_video_start_frame'])
                slice_end = int(target_data['target_video_end_frame']) + 1
                image_paths = image_paths[slice_start:slice_end]

                if not SM.detector.detect(image_paths, target_type, gui=gr):return
                temp_path = os.path.join(output_data['output_directory'], output_name)
                os.makedirs(temp_path, exist_ok=True)
                output_paths = io_util.copy_files_to_directory(image_paths, temp_path)

                TE = ThreadedExecutor(process_frame, output_paths)
                TE.run(threads=max_threads, text="Processing Images")

                gv.OUTPUT_DIRECTORY = temp_path
                gv.OUTPUT_FILE = None

                total_process_frames = len(output_paths)
                relative_fps = total_process_frames / (time.time() - main_process_start_time)
                gr.Info("Completed")

                print("  ---------------------------------SUMMARY----------------------------------  ")
                print(f"    Total Frames         : {total_process_frames}")
                print(f"    Threads              : {int(max_threads)}")
                print("  --------------------------------------------------------------------------  ")
                print(f"    Total Execution Time : {calculate_execution_time(main_process_start_time)}")
                print(f"    Relative FPS         : {relative_fps:.4f}")
                print("  --------------------------------------------------------------------------  ")
                print(f"    Output Path          : {temp_path}")
                print("  --------------------------------------------------------------------------  ")




    swap_inputs = [
        gr.State(True),
        cmpts.face_id,
        cmpts.control_id,
        cmpts.use_control_id,
        cmpts.frame_slider,
        cmpts.preview_type,
        cmpts.enable_realtime_update,
        max_threads,
    ]

    gui.face_id.set_on_face_id_updates(cmpts)

    gui.update.set_on_component_update(swap_button, "click").then(
        main_process,
        swap_inputs,
        [cmpts.preview_image]
    )

    realtime_swap_inputs = swap_inputs
    realtime_swap_inputs[0] = gr.State(False)

    cmpts.realtime_update_components.append(cmpts.preview_type)
    cmpts.realtime_update_components.append(cmpts.frame_slider)
    cmpts.realtime_update_components.append(cmpts.use_control_id)
    cmpts.realtime_update_components.append(cmpts.enable_realtime_update)
    cmpts.realtime_update_components.extend(cmpts.source_components)
    cmpts.realtime_update_components.extend(cmpts.target_components)
    cmpts.realtime_update_components.extend(cmpts.settings_components)
    cmpts.realtime_update_components.extend(cmpts.face_control_components)
    cmpts.realtime_update_components.extend(cmpts.target_selection_components)

    event_priorities = ["upload", "submit", "release", "input", "change"]
    for component in cmpts.realtime_update_components:
        for event in event_priorities:
            if hasattr(component, event):
                gui.update.set_on_component_update(component, event).then(
                    main_process,
                    realtime_swap_inputs,
                    [cmpts.preview_image]
                )
                if event == "upload":
                    gui.update.set_on_component_update(component, "clear").then(
                        main_process,
                        realtime_swap_inputs,
                        [cmpts.preview_image]
                    )
                break

    def cancel_process():
        gv.CANCEL_SIGNAL = True
        gr.Info("Cancelled")

    cancel_button.click(
        cancel_process
    )

    def open_output_directory():
        io_util.open_directory(gv.OUTPUT_DIRECTORY)

    def open_output_file():
        io_util.open_file(gv.OUTPUT_FILE)

    open_output_directory_button.click(
        fn=open_output_directory
    )
    open_output_file_button.click(
        fn=open_output_file
    )

if __name__ == "__main__":
    ui.queue(concurrency_count=2, api_open = False)
    ui.launch(show_api = False)
